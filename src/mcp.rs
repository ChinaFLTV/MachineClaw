use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Duration,
};

use reqwest::{
    blocking::{Client, Response},
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Value, json};

use crate::{
    ai::ExternalToolDefinition,
    config::{McpConfig, McpServerConfig},
    error::AppError,
    mask::mask_sensitive,
    tls::ensure_rustls_crypto_provider,
};

const DEFAULT_MCP_TIMEOUT_SECONDS: u64 = 10;

#[derive(Debug, Clone)]
struct NormalizedServerConfig {
    name: String,
    transport: McpTransportMode,
    endpoint: Option<String>,
    command: Option<String>,
    args: Vec<String>,
    env: BTreeMap<String, String>,
    headers: BTreeMap<String, String>,
    timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpTransportMode {
    Stdio,
    Http,
}

#[derive(Debug, Clone)]
struct McpGlobalDefaults {
    transport: Option<String>,
    timeout_seconds: Option<u64>,
    headers: BTreeMap<String, String>,
    auth_type: Option<String>,
    auth_token: Option<String>,
}

pub fn validate_mcp_config(cfg: &McpConfig) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let _ = normalized_server_configs(cfg)?;
    Ok(())
}

pub fn mcp_summary(cfg: &McpConfig) -> String {
    if !cfg.enabled {
        return "disabled".to_string();
    }
    let servers = match normalized_server_configs(cfg) {
        Ok(servers) => servers,
        Err(err) => {
            return format!(
                "enabled, invalid config: {}",
                mask_sensitive(&err.to_string())
            );
        }
    };
    if servers.is_empty() {
        return "enabled, servers=0".to_string();
    }
    let mut previews = Vec::new();
    for server in servers.iter().take(3) {
        match server.transport {
            McpTransportMode::Http => {
                if let Some(endpoint) = server.endpoint.as_deref() {
                    previews.push(format!("{}:http={}", server.name, mask_sensitive(endpoint)));
                }
            }
            McpTransportMode::Stdio => {
                if let Some(command) = server.command.as_deref() {
                    previews.push(format!(
                        "{}:stdio={} args={}",
                        server.name,
                        mask_sensitive(command),
                        server.args.len()
                    ));
                }
            }
        }
    }
    if previews.is_empty() {
        return format!("enabled, servers={}", servers.len());
    }
    format!(
        "enabled, servers={}, {}",
        servers.len(),
        previews.join("; ")
    )
}

#[derive(Debug, Clone)]
struct McpToolInfo {
    ai_name: String,
    remote_name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone)]
struct ToolBinding {
    connection_idx: usize,
    remote_name: String,
}

struct McpConnection {
    name: String,
    timeout: Duration,
    transport: McpTransport,
}

pub struct McpManager {
    enabled: bool,
    tools: Vec<McpToolInfo>,
    tool_index: HashMap<String, ToolBinding>,
    connections: Vec<McpConnection>,
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
    headers: HeaderMap,
    session_id: Option<String>,
    next_id: u64,
}

impl McpManager {
    pub fn connect(cfg: &McpConfig) -> Result<Self, AppError> {
        if !cfg.enabled {
            return Ok(Self {
                enabled: false,
                tools: Vec::new(),
                tool_index: HashMap::new(),
                connections: Vec::new(),
                summary: "disabled".to_string(),
            });
        }

        let server_configs = normalized_server_configs(cfg)?;
        if server_configs.is_empty() {
            return Ok(Self {
                enabled: true,
                tools: Vec::new(),
                tool_index: HashMap::new(),
                connections: Vec::new(),
                summary: "enabled, servers=0".to_string(),
            });
        }

        let total_servers = server_configs.len();
        let mut connections = Vec::<McpConnection>::new();
        let mut tools = Vec::<McpToolInfo>::new();
        let mut tool_index = HashMap::<String, ToolBinding>::new();
        let mut errors = Vec::<String>::new();

        for server in server_configs {
            let connect_outcome = connect_one_server(&server);
            let (transport, initialize_result, tools_result) = match connect_outcome {
                Ok(value) => value,
                Err(err) => {
                    errors.push(format!(
                        "{}: {}",
                        server.name,
                        mask_sensitive(&err.to_string())
                    ));
                    continue;
                }
            };

            let connection_idx = connections.len();
            let server_tools = extract_tools(&server.name, &tools_result);
            for tool in &server_tools {
                tool_index.insert(
                    tool.ai_name.clone(),
                    ToolBinding {
                        connection_idx,
                        remote_name: tool.remote_name.clone(),
                    },
                );
            }
            tools.extend(server_tools);
            connections.push(McpConnection {
                name: server.name.clone(),
                timeout: server.timeout,
                transport,
            });

            let _ = initialize_result;
        }

        if connections.is_empty() {
            let reason = if errors.is_empty() {
                "no MCP server available".to_string()
            } else {
                format!("no MCP server available: {}", errors.join(" | "))
            };
            return Err(AppError::Runtime(reason));
        }

        let mut detail = connections
            .iter()
            .map(|conn| format!("{}:{}", conn.name, transport_name(&conn.transport)))
            .collect::<Vec<_>>();
        detail.sort();

        let summary = if errors.is_empty() {
            format!(
                "enabled, servers={}/{}, tools={}, {}",
                connections.len(),
                total_servers,
                tools.len(),
                detail.join("; ")
            )
        } else {
            format!(
                "enabled, servers={}/{}, tools={}, {} (partial failures: {})",
                connections.len(),
                total_servers,
                tools.len(),
                detail.join("; "),
                errors.join(" | ")
            )
        };

        Ok(Self {
            enabled: true,
            tools,
            tool_index,
            connections,
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

    pub fn resolve_ai_tool_target(&self, name: &str) -> Option<(String, String)> {
        let binding = self.tool_index.get(name)?;
        let connection = self.connections.get(binding.connection_idx)?;
        Some((connection.name.clone(), binding.remote_name.clone()))
    }

    pub fn call_ai_tool(&mut self, name: &str, raw_arguments: &str) -> Result<String, AppError> {
        if !self.enabled {
            return Err(AppError::Runtime("mcp is disabled".to_string()));
        }
        let Some(binding) = self.tool_index.get(name).cloned() else {
            return Err(AppError::Runtime(format!(
                "mcp tool not found: {}",
                mask_sensitive(name)
            )));
        };
        let Some(connection) = self.connections.get_mut(binding.connection_idx) else {
            return Err(AppError::Runtime("mcp connection not found".to_string()));
        };
        let args = parse_mcp_tool_arguments(raw_arguments)?;

        let result = request_mcp(
            &mut connection.transport,
            "tools/call",
            json!({
                "name": binding.remote_name,
                "arguments": args
            }),
            connection.timeout,
        )?;
        Ok(extract_tool_call_text(&result))
    }
}

impl Drop for McpManager {
    fn drop(&mut self) {
        for connection in &mut self.connections {
            if let McpTransport::Stdio(client) = &mut connection.transport {
                let _ = client.child.kill();
                let _ = client.child.wait();
            }
        }
    }
}

impl StdioMcpClient {
    fn start(
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
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
    fn new(
        endpoint: &str,
        headers: &BTreeMap<String, String>,
        timeout: Duration,
    ) -> Result<Self, AppError> {
        ensure_rustls_crypto_provider();
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| AppError::Runtime(format!("failed to build MCP http client: {err}")))?;
        let headers = parse_http_headers(headers)?;
        Ok(Self {
            client,
            endpoint: endpoint.to_string(),
            headers,
            session_id: None,
            next_id: 1,
        })
    }
}

fn normalized_server_configs(cfg: &McpConfig) -> Result<Vec<NormalizedServerConfig>, AppError> {
    let defaults = McpGlobalDefaults {
        transport: cfg.transport.clone(),
        timeout_seconds: cfg.timeout_seconds,
        headers: cfg.headers.clone(),
        auth_type: cfg.auth_type.clone(),
        auth_token: cfg.auth_token.clone(),
    };
    if !cfg.servers.is_empty() {
        let mut keys = cfg.servers.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        let mut out = Vec::new();
        for name in keys {
            let Some(item) = cfg.servers.get(&name) else {
                continue;
            };
            if let Some(server) = normalize_server(name, item, &defaults)? {
                out.push(server);
            }
        }
        return Ok(out);
    }

    let has_legacy_target = non_empty_trimmed(cfg.server_url.as_deref()).is_some()
        || non_empty_trimmed(cfg.endpoint.as_deref()).is_some()
        || non_empty_trimmed(cfg.command.as_deref()).is_some();
    if !has_legacy_target {
        return Ok(Vec::new());
    }

    let legacy = McpServerConfig {
        enabled: true,
        transport: cfg.transport.clone(),
        server_url: cfg.server_url.clone(),
        endpoint: cfg.endpoint.clone(),
        command: cfg.command.clone(),
        args: cfg.args.clone(),
        env: cfg.env.clone(),
        headers: cfg.headers.clone(),
        auth_type: cfg.auth_type.clone(),
        auth_token: cfg.auth_token.clone(),
        timeout_seconds: cfg.timeout_seconds,
    };
    Ok(
        match normalize_server("root".to_string(), &legacy, &defaults)? {
            Some(item) => vec![item],
            None => Vec::new(),
        },
    )
}

fn normalize_server(
    name: String,
    server: &McpServerConfig,
    defaults: &McpGlobalDefaults,
) -> Result<Option<NormalizedServerConfig>, AppError> {
    if !server.enabled {
        return Ok(None);
    }
    let endpoint = non_empty_trimmed(server.server_url.as_deref())
        .or_else(|| non_empty_trimmed(server.endpoint.as_deref()));
    let command = non_empty_trimmed(server.command.as_deref());
    let transport = resolve_transport_mode(
        &name,
        server
            .transport
            .as_deref()
            .or(defaults.transport.as_deref()),
        endpoint.is_some(),
        command.is_some(),
    )?;
    let timeout_seconds = server
        .timeout_seconds
        .or(defaults.timeout_seconds)
        .unwrap_or(DEFAULT_MCP_TIMEOUT_SECONDS);
    if timeout_seconds == 0 {
        return Err(AppError::Config(format!(
            "mcp server '{}' timeout-seconds must be greater than 0",
            name
        )));
    }
    let mut headers = defaults.headers.clone();
    headers.extend(server.headers.clone());
    if transport == McpTransportMode::Http {
        apply_auth_to_headers(
            &name,
            &mut headers,
            server
                .auth_type
                .as_deref()
                .or(defaults.auth_type.as_deref()),
            server
                .auth_token
                .as_deref()
                .or(defaults.auth_token.as_deref()),
        )?;
    }
    Ok(Some(NormalizedServerConfig {
        name,
        transport,
        endpoint,
        command,
        args: server.args.clone(),
        env: server.env.clone(),
        headers,
        timeout: Duration::from_secs(timeout_seconds),
    }))
}

fn connect_one_server(
    server: &NormalizedServerConfig,
) -> Result<(McpTransport, Value, Value), AppError> {
    let mut transport = match server.transport {
        McpTransportMode::Stdio => {
            let Some(command) = server.command.as_deref() else {
                return Err(AppError::Config(format!(
                    "mcp server '{}' transport=stdio requires command",
                    server.name
                )));
            };
            McpTransport::Stdio(StdioMcpClient::start(command, &server.args, &server.env)?)
        }
        McpTransportMode::Http => {
            let Some(endpoint) = server.endpoint.as_deref() else {
                return Err(AppError::Config(format!(
                    "mcp server '{}' transport=http requires endpoint or server-url",
                    server.name
                )));
            };
            McpTransport::Http(HttpMcpClient::new(
                endpoint,
                &server.headers,
                server.timeout,
            )?)
        }
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
        server.timeout,
    )?;

    notify_mcp(&mut transport, "notifications/initialized", json!({}))?;
    let tools_result = request_mcp(&mut transport, "tools/list", json!({}), server.timeout)?;
    Ok((transport, initialize_result, tools_result))
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
            let resp = send_http_request(client, &body, "notification", false)?;
            update_http_session_id(client, resp.headers());
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
    let resp = send_http_request(client, &payload, method, true)?;
    let headers = resp.headers().clone();
    let body = parse_mcp_http_response(resp)?;
    update_http_session_id(client, &headers);
    if let Some(err) = body.get("error") {
        return Err(AppError::Runtime(format!(
            "MCP returned error for method {}: {}",
            method,
            mask_sensitive(&err.to_string())
        )));
    }
    Ok(body.get("result").cloned().unwrap_or_else(|| json!({})))
}

fn send_http_request(
    client: &HttpMcpClient,
    payload: &Value,
    method: &str,
    expect_response: bool,
) -> Result<Response, AppError> {
    let mut request_builder = client.client.post(&client.endpoint).json(payload);
    request_builder = request_builder.headers(client.headers.clone());
    if let Some(session_id) = client.session_id.as_deref() {
        request_builder = request_builder.header("Mcp-Session-Id", session_id);
    }
    let response = request_builder
        .send()
        .map_err(|err| AppError::Runtime(format!("MCP http request failed: {err}")))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        return Err(AppError::Runtime(format!(
            "MCP http request failed: status={}, body={}, method={}",
            status,
            mask_sensitive(&trim_text_preview(&body, 1200)),
            method
        )));
    }
    if !expect_response {
        return Ok(response);
    }
    Ok(response)
}

fn update_http_session_id(client: &mut HttpMcpClient, headers: &HeaderMap) {
    if let Some(value) = headers
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        client.session_id = Some(value.to_string());
    }
}

fn parse_mcp_http_response(resp: Response) -> Result<Value, AppError> {
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let body = resp
        .text()
        .map_err(|err| AppError::Runtime(format!("failed to read MCP response body: {err}")))?;
    if body.trim().is_empty() {
        return Ok(json!({}));
    }
    if content_type.contains("text/event-stream") {
        return parse_sse_jsonrpc_payload(&body);
    }
    serde_json::from_str::<Value>(&body)
        .map_err(|err| AppError::Runtime(format!("failed to parse MCP response: {err}")))
}

fn parse_sse_jsonrpc_payload(raw: &str) -> Result<Value, AppError> {
    let mut data_lines: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim_end_matches('\r');
        if trimmed.is_empty() {
            if let Some(value) = parse_sse_data_block(&data_lines) {
                return Ok(value);
            }
            data_lines.clear();
            continue;
        }
        if trimmed.starts_with(':') {
            continue;
        }
        if let Some(data) = trimmed.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
    if let Some(value) = parse_sse_data_block(&data_lines) {
        return Ok(value);
    }
    Err(AppError::Runtime(
        "failed to parse MCP SSE response: no valid JSON-RPC payload".to_string(),
    ))
}

fn parse_sse_data_block(data_lines: &[String]) -> Option<Value> {
    if data_lines.is_empty() {
        return None;
    }
    let payload = data_lines.join("\n");
    let trimmed = payload.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(trimmed).ok()?;
    if parsed.get("result").is_some() || parsed.get("error").is_some() {
        return Some(parsed);
    }
    let nested = parsed
        .get("data")
        .and_then(|value| value.as_object())
        .and_then(|_| parsed.get("data").cloned())?;
    if nested.get("result").is_some() || nested.get("error").is_some() {
        return Some(nested);
    }
    None
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

fn non_empty_trimmed(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn resolve_transport_mode(
    server_name: &str,
    transport_raw: Option<&str>,
    has_endpoint: bool,
    has_command: bool,
) -> Result<McpTransportMode, AppError> {
    if let Some(raw) = non_empty_trimmed(transport_raw) {
        let normalized = raw.to_ascii_lowercase();
        return match normalized.as_str() {
            "http" => {
                if has_endpoint {
                    Ok(McpTransportMode::Http)
                } else {
                    Err(AppError::Config(format!(
                        "mcp server '{}' transport=http requires endpoint or server-url",
                        server_name
                    )))
                }
            }
            "stdio" => {
                if has_command {
                    Ok(McpTransportMode::Stdio)
                } else {
                    Err(AppError::Config(format!(
                        "mcp server '{}' transport=stdio requires command",
                        server_name
                    )))
                }
            }
            _ => Err(AppError::Config(format!(
                "mcp server '{}' has invalid transport '{}', expected one of: http, stdio",
                server_name, raw
            ))),
        };
    }
    if has_command {
        return Ok(McpTransportMode::Stdio);
    }
    if has_endpoint {
        return Ok(McpTransportMode::Http);
    }
    Err(AppError::Config(format!(
        "mcp server '{}' requires endpoint or command",
        server_name
    )))
}

fn apply_auth_to_headers(
    server_name: &str,
    headers: &mut BTreeMap<String, String>,
    auth_type: Option<&str>,
    auth_token: Option<&str>,
) -> Result<(), AppError> {
    let normalized_auth_type = non_empty_trimmed(auth_type).map(|value| value.to_ascii_lowercase());
    if let Some(auth_type_value) = normalized_auth_type.as_deref()
        && auth_type_value != "bearer"
    {
        return Err(AppError::Config(format!(
            "mcp server '{}' has invalid auth-type '{}', expected: bearer",
            server_name, auth_type_value
        )));
    }
    let token = non_empty_trimmed(auth_token);
    if token.is_none() {
        return Ok(());
    }
    if headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization"))
    {
        return Ok(());
    }
    let token = token.unwrap_or_default();
    let auth_value = if token.to_ascii_lowercase().starts_with("bearer ") {
        token
    } else {
        format!("Bearer {token}")
    };
    headers.insert("Authorization".to_string(), auth_value);
    Ok(())
}

fn parse_http_headers(raw_headers: &BTreeMap<String, String>) -> Result<HeaderMap, AppError> {
    let mut headers = HeaderMap::new();
    for (raw_key, raw_value) in raw_headers {
        let key = raw_key.trim();
        if key.is_empty() {
            continue;
        }
        let value = raw_value.trim();
        let header_name = HeaderName::from_bytes(key.as_bytes())
            .map_err(|err| AppError::Config(format!("invalid MCP header name '{}': {err}", key)))?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            AppError::Config(format!("invalid MCP header value for '{}': {err}", key))
        })?;
        headers.insert(header_name, header_value);
    }
    if !headers.contains_key(ACCEPT) {
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
    }
    if !headers.contains_key(CONTENT_TYPE) {
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }
    Ok(headers)
}

fn extract_tools(server_name: &str, result: &Value) -> Vec<McpToolInfo> {
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
        let sanitized_server = sanitize_tool_name(server_name);
        let sanitized_tool = sanitize_tool_name(remote_name);
        output.push(McpToolInfo {
            ai_name: format!("mcp__{sanitized_server}__{sanitized_tool}"),
            remote_name: remote_name.to_string(),
            description: format!("MCP[{server_name}] tool {remote_name}: {description}"),
            parameters: params,
        });
    }
    output
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
    let normalized = out.trim_matches('_').to_string();
    if normalized.is_empty() {
        return "tool".to_string();
    }
    normalized
}

fn parse_mcp_tool_arguments(raw_arguments: &str) -> Result<Value, AppError> {
    let trimmed = raw_arguments.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }
    let parsed = serde_json::from_str::<Value>(trimmed).map_err(|err| {
        AppError::Runtime(format!(
            "invalid MCP tool arguments JSON: {}, raw={}",
            err,
            mask_sensitive(&trim_text_preview(trimmed, 240))
        ))
    })?;
    if !parsed.is_object() {
        return Err(AppError::Runtime(
            "invalid MCP tool arguments JSON: expected a JSON object".to_string(),
        ));
    }
    Ok(parsed)
}

fn trim_text_preview(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let trimmed: String = text.chars().take(max_chars).collect();
    format!("{}...", trimmed)
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        McpGlobalDefaults, McpManager, McpTransportMode, mcp_summary, non_empty_trimmed,
        normalize_server, parse_mcp_tool_arguments, resolve_transport_mode, validate_mcp_config,
    };
    use crate::config::{McpConfig, McpServerConfig};

    #[test]
    fn parse_mcp_tool_arguments_accepts_empty_input() {
        let parsed = parse_mcp_tool_arguments("").expect("empty input should be allowed");
        assert_eq!(parsed.to_string(), "{}");
    }

    #[test]
    fn parse_mcp_tool_arguments_rejects_invalid_json() {
        let err = parse_mcp_tool_arguments("{not-json").expect_err("invalid json must fail");
        assert!(err.to_string().contains("invalid MCP tool arguments JSON"));
    }

    #[test]
    fn parse_mcp_tool_arguments_rejects_non_object_json() {
        let err = parse_mcp_tool_arguments("[1,2,3]").expect_err("non-object json must fail");
        assert!(
            err.to_string()
                .contains("invalid MCP tool arguments JSON: expected a JSON object")
        );
    }

    #[test]
    fn normalize_server_supports_server_url_and_bearer_auth() {
        let server = McpServerConfig {
            enabled: true,
            transport: Some("http".to_string()),
            server_url: Some("https://example.com/mcp".to_string()),
            endpoint: None,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            headers: BTreeMap::from([("X-Trace-Id".to_string(), "abc".to_string())]),
            auth_type: None,
            auth_token: Some("token-value".to_string()),
            timeout_seconds: Some(12),
        };
        let defaults = McpGlobalDefaults {
            transport: None,
            timeout_seconds: None,
            headers: BTreeMap::new(),
            auth_type: None,
            auth_token: None,
        };

        let normalized = normalize_server("remote".to_string(), &server, &defaults)
            .expect("normalize should pass")
            .expect("server should stay enabled");
        assert_eq!(normalized.transport, McpTransportMode::Http);
        assert_eq!(
            normalized.endpoint.as_deref(),
            Some("https://example.com/mcp")
        );
        assert_eq!(
            normalized.headers.get("Authorization").map(String::as_str),
            Some("Bearer token-value")
        );
        assert_eq!(
            normalized.headers.get("X-Trace-Id").map(String::as_str),
            Some("abc")
        );
    }

    #[test]
    fn resolve_transport_mode_prefers_stdio_when_both_fields_are_present() {
        let transport = resolve_transport_mode("srv", None, true, true).expect("transport");
        assert_eq!(transport, McpTransportMode::Stdio);
    }

    #[test]
    fn validate_mcp_config_rejects_invalid_transport_value() {
        let mut servers = BTreeMap::new();
        servers.insert(
            "bad".to_string(),
            McpServerConfig {
                enabled: true,
                transport: Some("stream".to_string()),
                server_url: Some("https://example.com/mcp".to_string()),
                endpoint: None,
                command: None,
                args: vec![],
                env: BTreeMap::new(),
                headers: BTreeMap::new(),
                auth_type: None,
                auth_token: None,
                timeout_seconds: None,
            },
        );
        let cfg = McpConfig {
            enabled: true,
            transport: None,
            server_url: None,
            endpoint: None,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
            auth_type: None,
            auth_token: None,
            timeout_seconds: None,
            servers,
        };
        let err = validate_mcp_config(&cfg).expect_err("invalid transport must fail");
        assert!(err.to_string().contains("invalid transport"));
    }

    #[test]
    fn validate_mcp_config_allows_enabled_without_any_server() {
        let cfg = McpConfig {
            enabled: true,
            ..McpConfig::default()
        };
        validate_mcp_config(&cfg).expect("enabled mcp without servers should be allowed");
        assert_eq!(mcp_summary(&cfg), "enabled, servers=0");
    }

    #[test]
    fn connect_allows_enabled_without_any_server() {
        let cfg = McpConfig {
            enabled: true,
            ..McpConfig::default()
        };
        let manager = McpManager::connect(&cfg).expect("connect should allow empty mcp server set");
        assert_eq!(manager.summary(), "enabled, servers=0");
        assert!(manager.external_tool_definitions().is_empty());
    }

    #[test]
    fn mcp_summary_uses_root_name_for_legacy_single_server() {
        let cfg = McpConfig {
            enabled: true,
            endpoint: Some("http://127.0.0.1:8080/mcp".to_string()),
            ..McpConfig::default()
        };
        let summary = mcp_summary(&cfg);
        assert!(summary.contains("root:http="));
        assert!(!summary.contains("default:http="));
    }

    #[test]
    fn non_empty_trimmed_skips_blank_strings() {
        assert_eq!(non_empty_trimmed(Some("  ")), None);
        assert_eq!(
            non_empty_trimmed(Some(" value ")),
            Some("value".to_string())
        );
    }
}
