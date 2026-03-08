use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Duration,
};

use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{ai::ExternalToolDefinition, config::McpConfig, error::AppError, mask::mask_sensitive};

pub fn validate_mcp_config(cfg: &McpConfig) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    if cfg
        .endpoint
        .as_deref()
        .unwrap_or_default()
        .trim()
        .is_empty()
        && cfg.command.as_deref().unwrap_or_default().trim().is_empty()
    {
        return Err(AppError::Config(
            "mcp.enabled=true requires at least one of mcp.endpoint or mcp.command".to_string(),
        ));
    }
    Ok(())
}

pub fn mcp_summary(cfg: &McpConfig) -> String {
    if !cfg.enabled {
        return "disabled".to_string();
    }
    let endpoint = cfg
        .endpoint
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(mask_sensitive);
    let command = cfg
        .command
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(mask_sensitive);
    if let Some(value) = endpoint {
        return format!("enabled, endpoint={value}");
    }
    if let Some(value) = command {
        return format!("enabled, command={value}, args={}", cfg.args.len());
    }
    "enabled".to_string()
}

#[derive(Debug, Clone)]
struct McpToolInfo {
    ai_name: String,
    remote_name: String,
    description: String,
    parameters: Value,
}

pub struct McpManager {
    enabled: bool,
    timeout: Duration,
    tools: Vec<McpToolInfo>,
    tool_index: HashMap<String, usize>,
    transport: Option<McpTransport>,
    summary: String,
}

enum McpTransport {
    Stdio(StdioMcpClient),
    Http(HttpMcpClient),
}

struct StdioMcpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

struct HttpMcpClient {
    client: Client,
    endpoint: String,
    next_id: u64,
}

impl McpManager {
    pub fn connect(cfg: &McpConfig) -> Result<Self, AppError> {
        if !cfg.enabled {
            return Ok(Self {
                enabled: false,
                timeout: Duration::from_secs(cfg.timeout_seconds.unwrap_or(10)),
                tools: Vec::new(),
                tool_index: HashMap::new(),
                transport: None,
                summary: "disabled".to_string(),
            });
        }

        let timeout = Duration::from_secs(cfg.timeout_seconds.unwrap_or(10));
        let mut transport = if let Some(command) =
            cfg.command.as_deref().filter(|v| !v.trim().is_empty())
        {
            McpTransport::Stdio(StdioMcpClient::start(command, &cfg.args, &cfg.env)?)
        } else if let Some(endpoint) = cfg.endpoint.as_deref().filter(|v| !v.trim().is_empty()) {
            McpTransport::Http(HttpMcpClient::new(endpoint, timeout)?)
        } else {
            return Err(AppError::Config(
                "mcp.enabled=true requires mcp.command or mcp.endpoint".to_string(),
            ));
        };

        let initialize_result = request_mcp(
            &mut transport,
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MachineClaw",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
            timeout,
        )?;

        notify_mcp(&mut transport, "notifications/initialized", json!({}))?;

        let tools_result = request_mcp(&mut transport, "tools/list", json!({}), timeout)?;
        let tools = extract_tools(&tools_result);
        let mut tool_index = HashMap::new();
        for (idx, item) in tools.iter().enumerate() {
            tool_index.insert(item.ai_name.clone(), idx);
        }

        let summary = format!(
            "enabled, transport={}, tools={}, init={}",
            transport_name(&transport),
            tools.len(),
            summarize_initialize(&initialize_result)
        );
        Ok(Self {
            enabled: true,
            timeout,
            tools,
            tool_index,
            transport: Some(transport),
            summary,
        })
    }

    pub fn summary(&self) -> String {
        self.summary.clone()
    }

    pub fn external_tool_definitions(&self) -> Vec<ExternalToolDefinition> {
        self.tools
            .iter()
            .map(|tool| ExternalToolDefinition {
                name: tool.ai_name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect()
    }

    pub fn has_ai_tool(&self, name: &str) -> bool {
        self.tool_index.contains_key(name)
    }

    pub fn call_ai_tool(&mut self, name: &str, raw_arguments: &str) -> Result<String, AppError> {
        if !self.enabled {
            return Err(AppError::Runtime("mcp is disabled".to_string()));
        }
        let tool_idx = self.tool_index.get(name).copied().ok_or_else(|| {
            AppError::Runtime(format!("mcp tool not found: {}", mask_sensitive(name)))
        })?;
        let Some(transport) = self.transport.as_mut() else {
            return Err(AppError::Runtime("mcp transport not connected".to_string()));
        };
        let tool = &self.tools[tool_idx];
        let args = if raw_arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(raw_arguments).unwrap_or_else(|_| json!({}))
        };

        let result = request_mcp(
            transport,
            "tools/call",
            json!({
                "name": tool.remote_name,
                "arguments": args
            }),
            self.timeout,
        )?;
        Ok(extract_tool_call_text(&result))
    }
}

impl Drop for McpManager {
    fn drop(&mut self) {
        if let Some(McpTransport::Stdio(client)) = self.transport.as_mut() {
            let _ = client.child.kill();
            let _ = client.child.wait();
        }
    }
}

impl StdioMcpClient {
    fn start(
        command: &str,
        args: &[String],
        env: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self, AppError> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (key, value) in env {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().map_err(|err| {
            AppError::Runtime(format!(
                "failed to start MCP command {}: {err}",
                mask_sensitive(command)
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AppError::Runtime("failed to open MCP stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::Runtime("failed to open MCP stdout".to_string()))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }
}

impl HttpMcpClient {
    fn new(endpoint: &str, timeout: Duration) -> Result<Self, AppError> {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| AppError::Runtime(format!("failed to build MCP http client: {err}")))?;
        Ok(Self {
            client,
            endpoint: endpoint.to_string(),
            next_id: 1,
        })
    }
}

fn request_mcp(
    transport: &mut McpTransport,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<Value, AppError> {
    match transport {
        McpTransport::Stdio(client) => stdio_request(client, method, params, timeout),
        McpTransport::Http(client) => http_request(client, method, params),
    }
}

fn notify_mcp(transport: &mut McpTransport, method: &str, params: Value) -> Result<(), AppError> {
    match transport {
        McpTransport::Stdio(client) => stdio_notify(client, method, params),
        McpTransport::Http(client) => {
            let body = json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params
            });
            client
                .client
                .post(&client.endpoint)
                .json(&body)
                .send()
                .map_err(|err| AppError::Runtime(format!("MCP http notification failed: {err}")))?;
            Ok(())
        }
    }
}

fn stdio_request(
    client: &mut StdioMcpClient,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<Value, AppError> {
    let id = client.next_id;
    client.next_id += 1;
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    });
    write_mcp_frame(&mut client.stdin, &payload)?;

    let started = std::time::Instant::now();
    loop {
        if started.elapsed() > timeout {
            return Err(AppError::Runtime(format!(
                "MCP request timeout: method={method}"
            )));
        }
        let frame = read_mcp_frame(&mut client.stdout)?;
        if let Some(resp_id) = frame.get("id")
            && resp_id == &json!(id)
        {
            if let Some(err) = frame.get("error") {
                return Err(AppError::Runtime(format!(
                    "MCP returned error for method {}: {}",
                    method,
                    mask_sensitive(&err.to_string())
                )));
            }
            return Ok(frame.get("result").cloned().unwrap_or_else(|| json!({})));
        }
    }
}

fn stdio_notify(client: &mut StdioMcpClient, method: &str, params: Value) -> Result<(), AppError> {
    let payload = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    });
    write_mcp_frame(&mut client.stdin, &payload)
}

fn http_request(
    client: &mut HttpMcpClient,
    method: &str,
    params: Value,
) -> Result<Value, AppError> {
    let id = client.next_id;
    client.next_id += 1;
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    });
    let resp = client
        .client
        .post(&client.endpoint)
        .json(&payload)
        .send()
        .map_err(|err| AppError::Runtime(format!("MCP http request failed: {err}")))?;
    let body: Value = resp
        .json()
        .map_err(|err| AppError::Runtime(format!("failed to parse MCP response: {err}")))?;
    if let Some(err) = body.get("error") {
        return Err(AppError::Runtime(format!(
            "MCP returned error for method {}: {}",
            method,
            mask_sensitive(&err.to_string())
        )));
    }
    Ok(body.get("result").cloned().unwrap_or_else(|| json!({})))
}

fn write_mcp_frame(writer: &mut ChildStdin, payload: &Value) -> Result<(), AppError> {
    let body = payload.to_string();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer
        .write_all(header.as_bytes())
        .and_then(|_| writer.write_all(body.as_bytes()))
        .and_then(|_| writer.flush())
        .map_err(|err| AppError::Runtime(format!("failed to write MCP frame: {err}")))
}

fn read_mcp_frame(reader: &mut BufReader<ChildStdout>) -> Result<Value, AppError> {
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|err| AppError::Runtime(format!("failed to read MCP header: {err}")))?;
        if n == 0 {
            return Err(AppError::Runtime(
                "MCP stdout closed unexpectedly".to_string(),
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = value
                .trim()
                .parse::<usize>()
                .map_err(|err| AppError::Runtime(format!("invalid MCP Content-Length: {err}")))?;
        }
    }
    if content_length == 0 {
        return Err(AppError::Runtime(
            "invalid MCP frame: missing Content-Length".to_string(),
        ));
    }
    let mut body = vec![0_u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|err| AppError::Runtime(format!("failed to read MCP body: {err}")))?;
    let value = serde_json::from_slice::<Value>(&body)
        .map_err(|err| AppError::Runtime(format!("failed to parse MCP json: {err}")))?;
    Ok(value)
}

fn extract_tools(result: &Value) -> Vec<McpToolInfo> {
    let tools = result
        .get("tools")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut output = Vec::new();
    for tool in tools {
        let Some(remote_name) = tool.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let description = tool
            .get("description")
            .and_then(|value| value.as_str())
            .unwrap_or("MCP tool")
            .to_string();
        let params = tool
            .get("inputSchema")
            .or_else(|| tool.get("input_schema"))
            .or_else(|| tool.get("parameters"))
            .cloned()
            .unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": true
                })
            });
        let sanitized = sanitize_tool_name(remote_name);
        output.push(McpToolInfo {
            ai_name: format!("mcp__{sanitized}"),
            remote_name: remote_name.to_string(),
            description: format!("MCP tool {remote_name}: {description}"),
            parameters: params,
        });
    }
    output
}

fn summarize_initialize(result: &Value) -> String {
    let protocol = result
        .get("protocolVersion")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let server = result
        .get("serverInfo")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    format!("{server}/{protocol}")
}

fn transport_name(transport: &McpTransport) -> &'static str {
    match transport {
        McpTransport::Stdio(_) => "stdio",
        McpTransport::Http(_) => "http",
    }
}

fn sanitize_tool_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_matches('_').to_string()
}

fn extract_tool_call_text(result: &Value) -> String {
    if let Some(content) = result.get("content").and_then(|value| value.as_array()) {
        let texts = content
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(|value| value.as_str())
                    .map(|value| value.trim().to_string())
            })
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if !texts.is_empty() {
            return texts.join("\n");
        }
    }
    if let Some(text) = result.get("text").and_then(|value| value.as_str()) {
        return text.to_string();
    }
    if let Some(value) = result.get("structuredContent") {
        return value.to_string();
    }
    result.to_string()
}
