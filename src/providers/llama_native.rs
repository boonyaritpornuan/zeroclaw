use crate::providers::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, ProviderCapabilities, StreamChunk,
    StreamOptions, StreamResult,
};
use async_trait::async_trait;
use anyhow::{bail, Result};
use futures_util::{stream, StreamExt};
use std::path::PathBuf;
use tracing::info;

#[cfg(feature = "inference-native")]
use candle_core::{Device, Tensor};
#[cfg(feature = "inference-native")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "inference-native")]
use candle_transformers::models::quantized_llama::ModelWeights;
#[cfg(feature = "inference-native")]
use candle_transformers::generation::{LogitsProcessor, Sampling};
#[cfg(feature = "inference-native")]
use tokenizers::Tokenizer;
#[cfg(feature = "inference-native")]
use std::sync::{Arc, Mutex, OnceLock};

// --- Cached Model (singleton, swappable) ------------------------------------

#[cfg(feature = "inference-native")]
struct NativeModel {
    model: ModelWeights,
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

// --- Provider ---------------------------------------------------------------

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

// --- Model Loading ----------------------------------------------------------

#[cfg(feature = "inference-native")]
fn load_model(model_path: &std::path::Path, device: &Device) -> Result<(ModelWeights, Tokenizer)> {
    info!("Loading GGUF model: {}", model_path.display());
    let start = std::time::Instant::now();

    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {}", e))?;

    let arch = content.metadata.get("general.architecture")
        .map(|v| format!("{v:?}").trim_matches('"').to_string())
        .unwrap_or_else(|| String::from("unknown"));
    let name = content.metadata.get("general.name")
        .map(|v| format!("{v:?}").trim_matches('"').to_string())
        .unwrap_or_else(|| String::from("unknown"));

    info!("Model '{}' (arch: {}, tensors: {}) headers in {:.1}s",
        name, arch, content.tensor_infos.len(), start.elapsed().as_secs_f32());

    let model = ModelWeights::from_gguf(content, &mut file, device)
        .map_err(|e| anyhow::anyhow!("Failed to build model: {}", e))?;
    info!("Weights loaded in {:.1}s", start.elapsed().as_secs_f32());

    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("tokenizer.json not found at: {}", tokenizer_path.display());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("Model ready! Total: {:.1}s", start.elapsed().as_secs_f32());
    Ok((model, tokenizer))
}

// --- EOS Detection ----------------------------------------------------------

#[cfg(feature = "inference-native")]
fn find_eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let vocab = tokenizer.get_vocab(true);
    let candidates = [
        "<|im_end|>",
        "<|end_of_text|>",
        "</s>",
        "<|endoftext|>",
    ];
    let mut eos_ids = Vec::new();
    for c in &candidates {
        if let Some(id) = vocab.get(*c) {
            eos_ids.push(*id);
        }
    }
    if eos_ids.is_empty() {
        // Fallback: use token ID 2 (common EOS)
        eos_ids.push(2);
    }
    eos_ids
}

// --- Text Generation --------------------------------------------------------

#[cfg(feature = "inference-native")]
fn generate_text(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    temperature: f64,
    max_tokens: usize,
) -> Result<String> {
    let start = std::time::Instant::now();

    let tokens = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens = tokens.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();
    info!("Prompt: {} tokens, max gen: {}", prompt_len, max_tokens);

    let sampling = if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::TopP { p: 0.9, temperature }
    };
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    // Prefill: process entire prompt at once
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;

    let prefill_dt = start.elapsed();
    info!("Prefill: {:.2} tok/s ({:.1}s)", prompt_len as f64 / prefill_dt.as_secs_f64(), prefill_dt.as_secs_f64());

    let mut next_token = logits_processor.sample(&logits)?;
    let mut all_tokens: Vec<u32> = vec![next_token];

    let eos_ids = find_eos_token_ids(tokenizer);

    // Autoregressive generation loop
    let gen_start = std::time::Instant::now();
    for i in 0..max_tokens {
        if eos_ids.contains(&next_token) {
            info!("EOS token hit at step {}", i);
            break;
        }

        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_len + i)?;
        let logits = logits.squeeze(0)?;

        // Apply repeat penalty
        let logits = if REPEAT_PENALTY != 1.0 {
            let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &all_tokens[start_at..],
            )?
        } else {
            logits
        };

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
    }

    let gen_dt = gen_start.elapsed();
    let gen_count = all_tokens.len();
    info!("Generated {} tokens in {:.1}s ({:.2} tok/s)",
        gen_count, gen_dt.as_secs_f64(), gen_count as f64 / gen_dt.as_secs_f64());

    // Decode tokens to text
    let text = tokenizer.decode(&all_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    Ok(text.trim().to_string())
}

// --- Build prompt in ChatML format ------------------------------------------

fn build_chatml_prompt(system: Option<&str>, messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    if let Some(sys) = system {
        prompt.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", sys));
    }

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if system.is_none() {
                    prompt.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", msg.content));
                }
            }
            "user" => {
                prompt.push_str(&format!("<|im_start|>user\n{}\n<|im_end|>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|im_start|>assistant\n{}\n<|im_end|>\n", msg.content));
            }
            "tool" => {
                prompt.push_str(&format!("<|im_start|>tool\n{}\n<|im_end|>\n", msg.content));
            }
            _ => {}
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// --- Provider Implementation ------------------------------------------------

#[async_trait]
impl Provider for LlamaNativeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: false,
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        _model: &str,
        temperature: f64,
    ) -> Result<String> {
        #[cfg(not(feature = "inference-native"))]
        {
            bail!("Native inference is not enabled. Recompile with: cargo build --features inference-native")
        }

        #[cfg(feature = "inference-native")]
        {
            info!("Native inference for model: {}", self.model_path.display());

            let mut model_lock = self.model_cache.lock()
                .map_err(|e| anyhow::anyhow!("Lock failed: {}", e))?;

            // Load or swap model if needed
            let needs_load = match model_lock.as_ref() {
                None => true,
                Some(m) => m.loaded_path != self.model_path,
            };

            if needs_load {
                if model_lock.is_some() {
                    info!("Swapping model: unloading previous, loading {}", self.model_path.display());
                }
                let device = Device::Cpu;
                let (model, tokenizer) = load_model(&self.model_path, &device)?;
                *model_lock = Some(NativeModel {
                    model,
                    tokenizer,
                    device,
                    loaded_path: self.model_path.clone(),
                });
            }

            let native = model_lock.as_mut().unwrap();

            // Build ChatML prompt
            let prompt = if let Some(sys) = system_prompt {
                format!("<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", sys, message)
            } else {
                format!("<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", message)
            };

            let result = generate_text(
                &mut native.model,
                &native.tokenizer,
                &native.device,
                &prompt,
                temperature,
                MAX_GENERATION_TOKENS,
            )?;

            Ok(result)
        }
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> Result<ChatResponse> {
        let text = self.chat_with_history(request.messages, model, temperature).await?;
        Ok(ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        })
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        _model: &str,
        temperature: f64,
    ) -> Result<String> {
        #[cfg(not(feature = "inference-native"))]
        {
            bail!("Native inference is not enabled. Recompile with: cargo build --features inference-native")
        }

        #[cfg(feature = "inference-native")]
        {
            let mut model_lock = self.model_cache.lock()
                .map_err(|e| anyhow::anyhow!("Lock failed: {}", e))?;

            let needs_load = match model_lock.as_ref() {
                None => true,
                Some(m) => m.loaded_path != self.model_path,
            };

            if needs_load {
                let device = Device::Cpu;
                let (model, tokenizer) = load_model(&self.model_path, &device)?;
                *model_lock = Some(NativeModel {
                    model,
                    tokenizer,
                    device,
                    loaded_path: self.model_path.clone(),
                });
            }

            let native = model_lock.as_mut().unwrap();
            let prompt = build_chatml_prompt(None, messages);

            generate_text(
                &mut native.model,
                &native.tokenizer,
                &native.device,
                &prompt,
                temperature,
                MAX_GENERATION_TOKENS,
            )
        }
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> Result<ChatResponse> {
        // For now, delegate to chat_with_history (tool calling will be parsed from text output)
        let text = self.chat_with_history(messages, model, temperature).await?;
        Ok(ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        })
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn stream_chat_with_system(
        &self,
        _system_prompt: Option<&str>,
        _message: &str,
        _model: &str,
        _temperature: f64,
        _options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        stream::empty().boxed()
    }
}
