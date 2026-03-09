use crate::providers::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, ProviderCapabilities, StreamChunk,
    StreamOptions, StreamResult,
};
use async_trait::async_trait;
use anyhow::{bail, Result};
use futures_util::{stream, StreamExt};
use std::path::PathBuf;
use tracing::{info, warn};

#[cfg(feature = "inference-native")]
use candle_core::{Device, Tensor};
#[cfg(feature = "inference-native")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "inference-native")]
use candle_transformers::generation::{LogitsProcessor, Sampling};
#[cfg(feature = "inference-native")]
use tokenizers::Tokenizer;
#[cfg(feature = "inference-native")]
use std::sync::{Arc, Mutex, OnceLock};

// --- Handle different model types ---
#[cfg(feature = "inference-native")]
enum ModelType {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
}

#[cfg(feature = "inference-native")]
struct NativeModel {
    model: ModelType,
    tokenizer: Tokenizer,
    device: Device,
    loaded_path: PathBuf,
}

#[cfg(feature = "inference-native")]
static NATIVE_MODEL_CACHE: OnceLock<Arc<Mutex<Option<NativeModel>>>> = OnceLock::new();

#[cfg(feature = "inference-native")]
pub fn is_model_loaded() -> bool {
    if let Some(cache) = NATIVE_MODEL_CACHE.get() {
        if let Ok(lock) = cache.lock() {
            return lock.is_some();
        }
    }
    false
}

#[cfg(not(feature = "inference-native"))]
pub fn is_model_loaded() -> bool {
    false
}

const MAX_GENERATION_TOKENS: usize = 2048;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;

pub struct LlamaNativeProvider {
    model_path: PathBuf,
    #[cfg(feature = "inference-native")]
    model_cache: Arc<Mutex<Option<NativeModel>>>,
}

impl LlamaNativeProvider {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            #[cfg(feature = "inference-native")]
            model_cache: NATIVE_MODEL_CACHE.get_or_init(|| Arc::new(Mutex::new(None))).clone(),
        }
    }
}

#[cfg(feature = "inference-native")]
fn load_model(model_path: &std::path::Path, device: &Device) -> Result<(ModelType, Tokenizer)> {
    info!("Loading GGUF model: {}", model_path.display());
    let start = std::time::Instant::now();

    let mut file = std::fs::File::open(model_path)?;
    let mut content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {}", e))?;

    let arch = content.metadata.get("general.architecture")
        .map(|v| format!("{:?}", v).trim_matches('"').to_string())
        .unwrap_or_else(|| String::from("llama"));
    
    let name = content.metadata.get("general.name")
        .map(|v| format!("{:?}", v).trim_matches('"').to_string())
        .unwrap_or_else(|| String::from("unknown"));

    info!("Model '{}' (arch: {}, tensors: {}) detected", name, arch, content.tensor_infos.len());

    let model = match arch.as_str() {
        "qwen2" | "qwen3" => {
            // Remap qwen3 keys to qwen2 for compatibility if needed
            if arch == "qwen3" {
                info!("Remapping qwen3 metadata to qwen2 for structural compatibility");
                let mut new_metadata = std::collections::HashMap::new();
                for (k, v) in content.metadata.iter() {
                    if k.starts_with("qwen3.") {
                        new_metadata.insert(k.replace("qwen3.", "qwen2."), v.clone());
                    }
                }
                content.metadata.extend(new_metadata);
            }
            let weights = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow::anyhow!("Qwen build failed: {}", e))?;
            ModelType::Qwen2(weights)
        }
        _ => {
            let weights = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow::anyhow!("Llama build failed: {}", e))?;
            ModelType::Llama(weights)
        }
    };

    info!("Weights loaded in {:.1}s", start.elapsed().as_secs_f32());

    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("tokenizer.json not found at: {}", tokenizer_path.display());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    Ok((model, tokenizer))
}

#[cfg(feature = "inference-native")]
fn find_eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let vocab = tokenizer.get_vocab(true);
    let candidates = ["<|im_end|>", "<|end_of_text|>", "</s>", "<|endoftext|>"];
    let mut eos_ids = Vec::new();
    for c in &candidates {
        if let Some(id) = vocab.get(*c) { eos_ids.push(*id); }
    }
    if eos_ids.is_empty() { eos_ids.push(2); }
    eos_ids
}

#[cfg(feature = "inference-native")]
fn generate_text(
    model: &mut ModelType,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    temperature: f64,
    max_tokens: usize,
) -> Result<String> {
    let tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens = tokens.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    let sampling = if temperature <= 0.0 { Sampling::ArgMax } else { Sampling::TopP { p: 0.9, temperature } };
    let seed = std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_millis() as u64;
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = match model {
        ModelType::Llama(m) => m.forward(&input, 0)?,
        ModelType::Qwen2(m) => m.forward(&input, 0)?,
    };
    let logits = logits.squeeze(0)?;

    let mut next_token = logits_processor.sample(&logits)?;
    let mut all_tokens: Vec<u32> = vec![next_token];
    let eos_ids = find_eos_token_ids(tokenizer);

    for i in 0..max_tokens {
        if eos_ids.contains(&next_token) { break; }
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = match model {
            ModelType::Llama(m) => m.forward(&input, prompt_len + i)?,
            ModelType::Qwen2(m) => m.forward(&input, prompt_len + i)?,
        };
        let logits = logits.squeeze(0)?;
        let logits = if REPEAT_PENALTY != 1.0 {
            let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(&logits, REPEAT_PENALTY, &all_tokens[start_at..])?
        } else { logits };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
    }

    Ok(tokenizer.decode(&all_tokens, true).map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?.trim().to_string())
}

fn build_chatml_prompt(system_prompt: Option<&str>, messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    if let Some(sys) = system_prompt {
        prompt.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", sys));
    }
    for msg in messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" | "tool" => {
                prompt.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", msg.role, msg.content));
            }
            _ => {}
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[async_trait]
impl Provider for LlamaNativeProvider {
    fn capabilities(&self) -> ProviderCapabilities { ProviderCapabilities { native_tool_calling: true, vision: false } }

    async fn chat_with_system(&self, system_prompt: Option<&str>, message: &str, _model: &str, temp: f64) -> Result<String> {
        #[cfg(not(feature = "inference-native"))] bail!("Native inference not enabled.");
        #[cfg(feature = "inference-native")] {
            let mut model_lock = self.model_cache.lock().map_err(|e| anyhow::anyhow!("Lock failed: {}", e))?;
            if model_lock.as_ref().map_or(true, |m| m.loaded_path != self.model_path) {
                let device = Device::Cpu;
                let (model, tokenizer) = load_model(&self.model_path, &device)?;
                *model_lock = Some(NativeModel { model, tokenizer, device, loaded_path: self.model_path.clone() });
            }
            let native = model_lock.as_mut().unwrap();
            let prompt = build_chatml_prompt(system_prompt, &[ChatMessage { role: "user".into(), content: message.into() }]);
            generate_text(&mut native.model, &native.tokenizer, &native.device, &prompt, temp, MAX_GENERATION_TOKENS)
        }
    }

    async fn chat(&self, req: ChatRequest<'_>, model: &str, temp: f64) -> Result<ChatResponse> {
        let text = self.chat_with_history(req.messages, model, temp).await?;
        Ok(ChatResponse { text: Some(text), tool_calls: vec![], usage: None, reasoning_content: None })
    }

    async fn chat_with_history(&self, messages: &[ChatMessage], _model: &str, temp: f64) -> Result<String> {
        #[cfg(feature = "inference-native")] {
            let mut model_lock = self.model_cache.lock().map_err(|e| anyhow::anyhow!("Lock failed: {}", e))?;
            if model_lock.as_ref().map_or(true, |m| m.loaded_path != self.model_path) {
                let device = Device::Cpu;
                let (model, tokenizer) = load_model(&self.model_path, &device)?;
                *model_lock = Some(NativeModel { model, tokenizer, device, loaded_path: self.model_path.clone() });
            }
            let native = model_lock.as_mut().unwrap();
            let prompt = build_chatml_prompt(None, messages);
            generate_text(&mut native.model, &native.tokenizer, &native.device, &prompt, temp, MAX_GENERATION_TOKENS)
        }
        #[cfg(not(feature = "inference-native"))] bail!("Native inference not enabled.");
    }

    async fn chat_with_tools(&self, msg: &[ChatMessage], _tools: &[serde_json::Value], model: &str, temp: f64) -> Result<ChatResponse> {
        let text = self.chat_with_history(msg, model, temp).await?;
        Ok(ChatResponse { text: Some(text), tool_calls: vec![], usage: None, reasoning_content: None })
    }

    fn supports_streaming(&self) -> bool { false }
    fn stream_chat_with_system(&self, _s: Option<&str>, _m: &str, _mo: &str, _t: f64, _o: StreamOptions) -> stream::BoxStream<'static, StreamResult<StreamChunk>> { stream::empty().boxed() }
}
