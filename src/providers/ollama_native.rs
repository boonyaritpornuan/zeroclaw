use crate::multimodal;
use crate::providers::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse, Provider, ProviderCapabilities, StreamChunk, StreamError, StreamOptions, StreamResult, TokenUsage, ToolCall as ProviderToolCall,
};
use async_trait::async_trait;
use futures_util::{stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct OllamaNativeProvider {
    base_url: String,
    api_key: Option<String>,
    reasoning_enabled: Option<bool>,
}

// ─── Request Structures ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    options: Options,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OutgoingToolCall>>,
    /// For 'tool' role, Ollama expects the name of the tool being responded to.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

#[derive(Debug, Serialize)]
struct OutgoingToolCall {
    #[serde(rename = "type")]
    kind: String,
    function: OutgoingFunction,
}

#[derive(Debug, Serialize)]
struct OutgoingFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct Options {
    temperature: f64,
}

// ─── Response Structures ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ApiChatResponse {
    message: ResponseMessage,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    done: bool,
    /// Some experimental versions might include thinking in the top-level
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
    /// Some models return a "thinking" field with internal reasoning
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaToolCall {
    id: Option<String>,
    function: OllamaFunction,
}

#[derive(Debug, Deserialize)]
struct OllamaFunction {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl OllamaNativeProvider {
    fn normalize_base_url(raw_url: &str) -> String {
        let trimmed = raw_url.trim().trim_end_matches('/');
        if trimmed.is_empty() {
            return String::new();
        }
        trimmed
            .strip_suffix("/api")
            .unwrap_or(trimmed)
            .trim_end_matches('/')
            .to_string()
    }

    pub fn new(base_url: Option<&str>, api_key: Option<&str>) -> Self {
        Self::new_with_reasoning(base_url, api_key, None)
    }

    pub fn new_with_reasoning(
        base_url: Option<&str>,
        api_key: Option<&str>,
        reasoning_enabled: Option<bool>,
    ) -> Self {
        let api_key = api_key.and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });

        Self {
            base_url: Self::normalize_base_url(base_url.unwrap_or("http://localhost:11434")),
            api_key,
            reasoning_enabled,
        }
    }

    fn is_local_endpoint(&self) -> bool {
        reqwest::Url::parse(&self.base_url)
            .ok()
            .and_then(|url| url.host_str().map(|host| host.to_string()))
            .is_some_and(|host| matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1"))
    }

    fn http_client(&self) -> Client {
        crate::config::build_runtime_proxy_client_with_timeouts("provider.ollama-native", 300, 10)
    }

    fn resolve_request_details(&self, model: &str) -> anyhow::Result<(String, bool)> {
        let normalized_model = model.strip_suffix(":cloud").unwrap_or(model).to_string();
        let requests_cloud = model.ends_with(":cloud");

        if requests_cloud && self.is_local_endpoint() {
            anyhow::bail!(
                "Model '{}' requested cloud routing, but Ollama endpoint is local. Configure api_url with a remote Ollama endpoint.",
                model
            );
        }

        let should_auth = self.api_key.is_some() && !self.is_local_endpoint();
        Ok((normalized_model, should_auth))
    }

    fn parse_tool_arguments(arguments: &str) -> serde_json::Value {
        serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
    }

    fn build_chat_request(
        &self,
        messages: Vec<Message>,
        model: &str,
        temperature: f64,
        tools: Option<&[serde_json::Value]>,
        stream: bool,
    ) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages,
            stream,
            options: Options { temperature },
            think: self.reasoning_enabled,
            tools: tools.map(|t| t.to_vec()),
        }
    }

    fn convert_user_message_content(&self, content: &str) -> (Option<String>, Option<Vec<String>>) {
        let (cleaned, image_refs) = multimodal::parse_image_markers(content);
        if image_refs.is_empty() {
            return (Some(content.to_string()), None);
        }

        let images: Vec<String> = image_refs
            .iter()
            .filter_map(|reference| multimodal::extract_ollama_image_payload(reference))
            .collect();

        if images.is_empty() {
            return (Some(content.to_string()), None);
        }

        let cleaned = cleaned.trim();
        let content = if cleaned.is_empty() {
            None
        } else {
            Some(cleaned.to_string())
        };

        (content, Some(images))
    }

    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<Message> {
        let mut tool_name_by_id: HashMap<String, String> = HashMap::new();

        messages
            .iter()
            .map(|message| {
                if message.role == "assistant" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&message.content) {
                        if let Some(tool_calls_value) = value.get("tool_calls") {
                            if let Ok(parsed_calls) =
                                serde_json::from_value::<Vec<ProviderToolCall>>(
                                    tool_calls_value.clone(),
                                )
                            {
                                let outgoing_calls: Vec<OutgoingToolCall> = parsed_calls
                                    .into_iter()
                                    .map(|call| {
                                        tool_name_by_id.insert(call.id.clone(), call.name.clone());
                                        OutgoingToolCall {
                                            kind: "function".to_string(),
                                            function: OutgoingFunction {
                                                name: call.name,
                                                arguments: Self::parse_tool_arguments(
                                                    &call.arguments,
                                                ),
                                            },
                                        }
                                    })
                                    .collect();
                                let content = value
                                    .get("content")
                                    .and_then(serde_json::Value::as_str)
                                    .map(ToString::to_string);
                                return Message {
                                    role: "assistant".to_string(),
                                    content,
                                    images: None,
                                    tool_calls: Some(outgoing_calls),
                                    tool_name: None,
                                };
                            }
                        }
                    }
                }

                if message.role == "tool" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&message.content) {
                        let tool_name = value
                            .get("tool_name")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string)
                            .or_else(|| {
                                value
                                    .get("tool_call_id")
                                    .and_then(serde_json::Value::as_str)
                                    .and_then(|id| tool_name_by_id.get(id))
                                    .cloned()
                            });
                        let content = value
                            .get("content")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string)
                            .or_else(|| {
                                (!message.content.trim().is_empty())
                                    .then_some(message.content.clone())
                            });

                        return Message {
                            role: "tool".to_string(),
                            content,
                            images: None,
                            tool_calls: None,
                            tool_name,
                        };
                    }
                }

                if message.role == "user" {
                    let (content, images) = self.convert_user_message_content(&message.content);
                    return Message {
                        role: "user".to_string(),
                        content,
                        images,
                        tool_calls: None,
                        tool_name: None,
                    };
                }

                Message {
                    role: message.role.clone(),
                    content: Some(message.content.clone()),
                    images: None,
                    tool_calls: None,
                    tool_name: None,
                }
            })
            .collect()
    }

    fn extract_tool_name_and_args(&self, tc: &OllamaToolCall) -> (String, serde_json::Value) {
        let name = tc.function.name.clone();
        let mut args = tc.function.arguments.clone();

        // Handle nested wrapping often seen in models like Mistral and Qwen:
        // {"name": "tool_call", "arguments": {"name": "shell", "arguments": {...}}}
        if name == "tool_call" {
            if let Some(inner_name) = args.get("name").and_then(|v| v.as_str()) {
                if let Some(inner_args) = args.get("arguments") {
                    return (inner_name.to_string(), inner_args.clone());
                }
            }
        }

        // Handle prefixed names: "tool.shell" -> "shell"
        if let Some(stripped) = name.strip_prefix("tool.") {
            return (stripped.to_string(), args);
        }

        // Special case: "tool_calls" array within arguments (sometimes happens when model gets confused)
        if let Some(calls) = args.get("tool_calls").and_then(|v| v.as_array()) {
            if let Some(first) = calls.first() {
                if let Some(inner_name) = first.get("name").and_then(|v| v.as_str()) {
                    if let Some(inner_args) = first.get("arguments") {
                        return (inner_name.to_string(), inner_args.clone());
                    }
                }
            }
        }

        (name, args)
    }

    fn normalize_response_text(content: String) -> Option<String> {
        if content.trim().is_empty() {
            None
        } else {
            Some(content)
        }
    }

    fn fallback_text_for_empty_content(model: &str, thinking: Option<&str>) -> String {
        if let Some(thinking) = thinking.map(str::trim).filter(|value| !value.is_empty()) {
            let thinking_reply_excerpt: String = thinking.chars().take(200).collect();
            return format!(
                "I was thinking about this: {}... but I didn't complete my response. Could you try asking again?",
                thinking_reply_excerpt
            );
        }

        tracing::warn!(
            "Ollama returned empty or whitespace content with no tool calls for model '{}'",
            model
        );
        "I couldn't get a complete response from Ollama. Please try again or switch to a different model."
            .to_string()
    }
}

#[async_trait]
impl Provider for OllamaNativeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: true,
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let (normalized_model, should_auth) = self.resolve_request_details(model)?;
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(Message {
                role: "system".to_string(),
                content: Some(sys.to_string()),
                images: None,
                tool_calls: None,
                tool_name: None,
            });
        }
        let (user_content, user_images) = self.convert_user_message_content(message);
        messages.push(Message {
            role: "user".to_string(),
            content: user_content,
            images: user_images,
            tool_calls: None,
            tool_name: None,
        });

        let request = self.build_chat_request(messages, &normalized_model, temperature, None, false);
        let mut rb = self
            .http_client()
            .post(format!("{}/api/chat", self.base_url))
            .json(&request);
        if should_auth {
            if let Some(key) = self.api_key.as_ref() {
                rb = rb.bearer_auth(key);
            }
        }

        let response = rb.send().await?;
        if !response.status().is_success() {
            anyhow::bail!("Ollama API error: {}", response.status());
        }
        let api_resp: ApiChatResponse = response.json().await?;
        Ok(api_resp.message.content)
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let (normalized_model, should_auth) = self.resolve_request_details(model)?;
        let api_messages = self.convert_messages(messages);
        let request =
            self.build_chat_request(api_messages, &normalized_model, temperature, None, false);
        let mut rb = self
            .http_client()
            .post(format!("{}/api/chat", self.base_url))
            .json(&request);
        if should_auth {
            if let Some(key) = self.api_key.as_ref() {
                rb = rb.bearer_auth(key);
            }
        }

        let response = rb.send().await?;
        if !response.status().is_success() {
            anyhow::bail!("Ollama API error: {}", response.status());
        }
        let api_resp: ApiChatResponse = response.json().await?;
        Ok(api_resp.message.content)
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let (normalized_model, should_auth) = self.resolve_request_details(model)?;
        let api_messages = self.convert_messages(messages);
        let request = self.build_chat_request(
            api_messages,
            &normalized_model,
            temperature,
            Some(tools),
            false,
        );
        let mut rb = self
            .http_client()
            .post(format!("{}/api/chat", self.base_url))
            .json(&request);
        if should_auth {
            if let Some(key) = self.api_key.as_ref() {
                rb = rb.bearer_auth(key);
            }
        }

        let response = rb.send().await?;
        if !response.status().is_success() {
            anyhow::bail!("Ollama API error: {}", response.status());
        }
        let api_resp: ApiChatResponse = response.json().await?;

        let usage = Some(TokenUsage {
            input_tokens: api_resp.prompt_eval_count,
            output_tokens: api_resp.eval_count,
        });

        if !api_resp.message.tool_calls.is_empty() {
            let tool_calls = api_resp
                .message
                .tool_calls
                .into_iter()
                .map(|tc| {
                    let (name, args) = self.extract_tool_name_and_args(&tc);
                    ProviderToolCall {
                        id: tc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        name,
                        arguments: args.to_string(),
                    }
                })
                .collect();

            return Ok(ChatResponse {
                text: Some(api_resp.message.content),
                tool_calls,
                usage,
                reasoning_content: api_resp.message.thinking.or(api_resp.thinking),
            });
        }

        let text = if let Some(content) = Self::normalize_response_text(api_resp.message.content) {
            content
        } else {
            Self::fallback_text_for_empty_content(
                &normalized_model,
                api_resp.message.thinking.as_deref(),
            )
        };

        Ok(ChatResponse {
            text: Some(text),
            tool_calls: vec![],
            usage,
            reasoning_content: api_resp.message.thinking.or(api_resp.thinking),
        })
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        if let Some(specs) = request.tools {
            if !specs.is_empty() {
                let tools: Vec<serde_json::Value> = specs
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": s.name,
                                "description": s.description,
                                "parameters": s.parameters
                            }
                        })
                    })
                    .collect();
                return self
                    .chat_with_tools(request.messages, &tools, model, temperature)
                    .await;
            }
        }

        let text = self
            .chat_with_history(request.messages, model, temperature)
            .await?;
        Ok(ChatResponse {
            text: Some(text),
            tool_calls: vec![],
            usage: None,
            reasoning_content: None,
        })
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn stream_chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
        let (normalized_model, should_auth) = match self.resolve_request_details(model) {
            Ok(v) => v,
            Err(e) => {
                return stream::once(async move { Err(StreamError::Provider(e.to_string())) })
                    .boxed()
            }
        };

        let api_messages = self.convert_messages(messages);
        let request =
            self.build_chat_request(api_messages, &normalized_model, temperature, None, true);
        let url = format!("{}/api/chat", self.base_url);
        let client = self.http_client();
        let api_key = self.api_key.clone();

        let s = stream::unfold(
            (client, url, request, api_key, should_auth, None, Vec::new()),
            |(client, url, request, api_key, should_auth, mut response, mut buffer)| async move {
                if response.is_none() {
                    let mut rb = client.post(&url).json(&request);
                    if should_auth {
                        if let Some(key) = api_key.as_ref() {
                            rb = rb.bearer_auth(key);
                        }
                    }
                    match rb.send().await {
                        Ok(resp) => {
                            if !resp.status().is_success() {
                                return Some((
                                    Err(StreamError::Provider(format!(
                                        "Ollama API error: {}",
                                        resp.status()
                                    ))),
                                    (client, url, request, api_key, should_auth, None, buffer),
                                ));
                            }
                            response = Some(resp);
                        }
                        Err(e) => {
                            return Some((
                                Err(StreamError::Http(e)),
                                (client, url, request, api_key, should_auth, None, buffer),
                            ))
                        }
                    }
                }

                let mut resp = response.unwrap();
                loop {
                    if let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes = buffer.drain(..pos + 1).collect::<Vec<u8>>();
                        let line = String::from_utf8_lossy(&line_bytes);
                        if line.trim().is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<ApiChatResponse>(&line) {
                            Ok(api_resp) => {
                                let mut delta = api_resp.message.content;
                                // If thinking is present in delta, prepend it or handle it.
                                // Ollama usually streams thinking if supported by model.
                                if let Some(thinking) = api_resp.message.thinking {
                                    delta = format!("<think>{}</think>{}", thinking, delta);
                                }

                                let mut chunk = StreamChunk::delta(delta);
                                if options.count_tokens {
                                    chunk = chunk.with_token_estimate();
                                }
                                if api_resp.done {
                                    chunk.is_final = true;
                                    if let (Some(p), Some(e)) =
                                        (api_resp.prompt_eval_count, api_resp.eval_count)
                                    {
                                        chunk.token_count = (p + e) as usize;
                                    }
                                }
                                return Some((
                                    Ok(chunk),
                                    (
                                        client,
                                        url,
                                        request,
                                        api_key,
                                        should_auth,
                                        Some(resp),
                                        buffer,
                                    ),
                                ));
                            }
                            Err(e) => {
                                return Some((
                                    Err(StreamError::Json(e)),
                                    (client, url, request, api_key, should_auth, None, buffer),
                                ))
                            }
                        }
                    }

                    match resp.chunk().await {
                        Ok(Some(chunk_bytes)) => buffer.extend_from_slice(&chunk_bytes),
                        Ok(None) => return None,
                        Err(e) => {
                            return Some((
                                Err(StreamError::Http(e)),
                                (client, url, request, api_key, should_auth, None, buffer),
                            ))
                        }
                    }
                }
            },
        );

        s.boxed()
    }
}
