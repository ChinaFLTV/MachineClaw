use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Duration,
};

use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{
    ai::ExternalToolDefinition,
    config::{McpConfig, McpServerConfig},
    error::AppError,
    mask::mask_sensitive,
};

#[derive(Debug, Clone)]
struct NormalizedServerConfig {
    name: String,
    endpoint: Option<String>,
    command: Option<String>,
    args: Vec<String>,
    env: BTreeMap<String, String>,
    timeout: Duration,
}

pub fn validate_mcp_config(cfg: &McpConfig) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let servers = normalized_server_configs(cfg);
    if servers.is_empty() {
        return Err(AppError::Config(
            "mcp.enabled=true requires at least one configured server".to_string(),
        ));
    }
    for server in servers {
        if server.endpoint.is_none() && server.command.is_none() {
            return Err(AppError::Config(format!(
                "mcp server '{}' requires endpoint or command",
                server.name
            )));
        }
    }
    Ok(())
}

pub fn mcp_summary(cfg: &McpConfig) -> String {
    if !cfg.enabled {
        return "disabled".to_string();
    }
    let servers = normalized_server_configs(cfg);
    if servers.is_empty() {
        return "enabled, servers=0".to_string();
    }
    let mut previews = Vec::new();
    for server in servers.iter().take(3) {
        if let Some(endpoint) = server.endpoint.as_deref() {
            previews.push(format!("{}:http={}", server.name, mask_sensitive(endpoint)));
        } else if let Some(command) = server.command.as_deref() {
            previews.push(format!(
                "{}:stdio={} args={}",
                server.name,
                mask_sensitive(command),
                server.args.len()
            ));
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

        let server_configs = normalized_server_configs(cfg);
        if server_configs.is_empty() {
            return Err(AppError::Config(
                "mcp.enabled=true requires at least one configured server".to_string(),
            ));
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
        let args = if raw_arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(raw_arguments).unwrap_or_else(|_| json!({}))
        };

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

fn normalized_server_configs(cfg: &McpConfig) -> Vec<NormalizedServerConfig> {
    if !cfg.servers.is_empty() {
        let mut keys = cfg.servers.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        return keys
            .into_iter()
            .filter_map(|name| {
                let item = cfg.servers.get(&name)?;
                normalize_server(name, item, cfg.timeout_seconds)
            })
            .collect();
    }

    let legacy = McpServerConfig {
        enabled: true,
        endpoint: cfg.endpoint.clone(),
        command: cfg.command.clone(),
        args: cfg.args.clone(),
        env: cfg.env.clone(),
        timeout_seconds: cfg.timeout_seconds,
    };
    normalize_server("default".to_string(), &legacy, cfg.timeout_seconds)
        .map(|item| vec![item])
        .unwrap_or_default()
}

fn normalize_server(
    name: String,
    server: &McpServerConfig,
    global_timeout: Option<u64>,
) -> Option<NormalizedServerConfig> {
    if !server.enabled {
        return None;
    }
    let endpoint = server
        .endpoint
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);
    let command = server
        .command
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);
    if endpoint.is_none() && command.is_none() {
        return None;
    }
    let timeout = server.timeout_seconds.or(global_timeout).unwrap_or(10);
    Some(NormalizedServerConfig {
        name,
        endpoint,
        command,
        args: server.args.clone(),
        env: server.env.clone(),
        timeout: Duration::from_secs(timeout),
    })
}

fn connect_one_server(
    server: &NormalizedServerConfig,
) -> Result<(McpTransport, Value, Value), AppError> {
    let mut transport = if let Some(command) = server.command.as_deref() {
        McpTransport::Stdio(StdioMcpClient::start(command, &server.args, &server.env)?)
    } else if let Some(endpoint) = server.endpoint.as_deref() {
        McpTransport::Http(HttpMcpClient::new(endpoint, server.timeout)?)
    } else {
        return Err(AppError::Config(format!(
            "mcp server '{}' requires endpoint or command",
            server.name
        )));
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
        .map_err(|err| AppError::Runtime(format!("MCP http request failed: {err}")));
    let resp = resp?;
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
