use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufRead, BufReader, ErrorKind, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread::{self, sleep},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use reqwest::{
    Url,
    blocking::{Client, Response},
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};

use crate::{
    ai::ExternalToolDefinition,
    config::{McpConfig, McpServerConfig, expand_tilde},
    error::AppError,
    mask::mask_sensitive,
    tls::ensure_rustls_crypto_provider,
};

const DEFAULT_MCP_TIMEOUT_SECONDS: u64 = 10;
const MCP_STDIO_POLL_INTERVAL: Duration = Duration::from_millis(20);
const MCP_SERVERS_FILE_NAME: &str = "servers.json";
const MCP_WRITE_LOCK_TIMEOUT: Duration = Duration::from_secs(2);
const MCP_WRITE_LOCK_STALE_AFTER: Duration = Duration::from_secs(120);
const MCP_WRITE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Clone)]
pub struct McpServerRecord {
    pub name: String,
    pub config: McpServerConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
struct McpServersFile {
    #[serde(default, rename = "mcpServers", alias = "mcp_servers")]
    mcp_servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug)]
struct McpWriteLockGuard {
    lock_path: PathBuf,
}

impl Drop for McpWriteLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

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
    Sse,
}

pub fn validate_mcp_config(cfg: &McpConfig, config_path: &Path) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let _ = normalized_server_configs(cfg, config_path)?;
    Ok(())
}

pub fn mcp_summary(cfg: &McpConfig, config_path: &Path) -> String {
    if !cfg.enabled {
        return "disabled".to_string();
    }
    let servers = match normalized_server_configs(cfg, config_path) {
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
            McpTransportMode::Sse => {
                if let Some(endpoint) = server.endpoint.as_deref() {
                    previews.push(format!("{}:sse={}", server.name, mask_sensitive(endpoint)));
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

pub fn load_mcp_server_records(
    cfg: &McpConfig,
    config_path: &Path,
) -> Result<Vec<McpServerRecord>, AppError> {
    let file_path = resolve_mcp_servers_file_path(cfg, config_path);
    let map = load_mcp_server_map_from_file(file_path.as_path())?;
    let mut names = map.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut records = Vec::with_capacity(names.len());
    for name in names {
        if let Some(config) = map.get(&name) {
            records.push(McpServerRecord {
                name,
                config: config.clone(),
            });
        }
    }
    Ok(records)
}

pub fn save_mcp_server_records(
    cfg: &McpConfig,
    config_path: &Path,
    records: &[McpServerRecord],
) -> Result<PathBuf, AppError> {
    let mut map = BTreeMap::<String, McpServerConfig>::new();
    for record in records {
        let name = record.name.trim();
        if name.is_empty() {
            return Err(AppError::Config(
                "MCP server name must not be empty".to_string(),
            ));
        }
        if map
            .insert(name.to_string(), record.config.clone())
            .is_some()
        {
            return Err(AppError::Config(format!(
                "duplicated MCP server name: {name}"
            )));
        }
    }
    let _ = normalized_server_configs_from_map(&map)?;
    let file_path = resolve_mcp_servers_file_path(cfg, config_path);
    write_mcp_server_map_to_file(file_path.as_path(), &map)?;
    Ok(file_path)
}

pub fn resolve_mcp_servers_file_path(cfg: &McpConfig, config_path: &Path) -> PathBuf {
    let expanded = expand_tilde(&cfg.dir);
    let base_path = if expanded.is_absolute() {
        expanded
    } else {
        let parent = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(default_config_base_dir);
        parent.join(expanded)
    };
    if base_path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        base_path
    } else {
        base_path.join(MCP_SERVERS_FILE_NAME)
    }
}

fn default_config_base_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[derive(Debug, Clone)]
struct McpToolInfo {
    ai_name: String,
    server_name: String,
    remote_name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone)]
pub struct McpServiceStatus {
    pub name: String,
    pub transport: String,
    pub target: String,
    pub args_count: usize,
    pub timeout_seconds: u64,
    pub available: bool,
    pub tool_count: usize,
    pub summary: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct McpToolStatus {
    pub server_name: String,
    pub ai_name: String,
    pub remote_name: String,
    pub description: String,
    pub available: bool,
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
    tool_statuses: Vec<McpToolStatus>,
    service_statuses: Vec<McpServiceStatus>,
    tool_index: HashMap<String, ToolBinding>,
    connections: Vec<McpConnection>,
    summary: String,
}

enum McpTransport {
    Stdio(StdioMcpClient),
    Http(HttpMcpClient),
    Sse(LegacySseMcpClient),
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

struct LegacySseMcpClient {
    client: Client,
    sse_endpoint: String,
    headers: HeaderMap,
    message_endpoint: Option<String>,
    session_id: Option<String>,
    event_rx: Option<Receiver<Result<LegacySseEvent, String>>>,
    event_stop_tx: Option<mpsc::Sender<()>>,
    event_thread: Option<thread::JoinHandle<()>>,
    next_id: u64,
}

#[derive(Debug)]
enum LegacySseEvent {
    Endpoint(String),
    Message(Value),
}

#[derive(Debug)]
struct SseFrame {
    event: Option<String>,
    data_lines: Vec<String>,
}

impl McpManager {
    pub fn pending(summary: String) -> Self {
        Self {
            enabled: true,
            tools: Vec::new(),
            tool_statuses: Vec::new(),
            service_statuses: Vec::new(),
            tool_index: HashMap::new(),
            connections: Vec::new(),
            summary,
        }
    }

    pub fn pending_with_config(cfg: &McpConfig, config_path: &Path, summary: String) -> Self {
        let mut manager = Self::pending(summary);
        let service_statuses = normalized_server_configs(cfg, config_path)
            .map(|servers| {
                servers
                    .into_iter()
                    .map(|server| McpServiceStatus {
                        name: server.name.clone(),
                        transport: transport_mode_name(server.transport).to_string(),
                        target: server_target_display(&server),
                        args_count: server.args.len(),
                        timeout_seconds: server.timeout.as_secs(),
                        available: false,
                        tool_count: 0,
                        summary: Some(format!(
                            "{}, availability=checking",
                            server_summary(&server)
                        )),
                        error: None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        manager.service_statuses = service_statuses;
        manager
    }

    pub fn connect(cfg: &McpConfig, config_path: &Path) -> Result<Self, AppError> {
        if !cfg.enabled {
            return Ok(Self {
                enabled: false,
                tools: Vec::new(),
                tool_statuses: Vec::new(),
                service_statuses: Vec::new(),
                tool_index: HashMap::new(),
                connections: Vec::new(),
                summary: "disabled".to_string(),
            });
        }

        let server_configs = normalized_server_configs(cfg, config_path)?;
        if server_configs.is_empty() {
            return Ok(Self {
                enabled: true,
                tools: Vec::new(),
                tool_statuses: Vec::new(),
                service_statuses: Vec::new(),
                tool_index: HashMap::new(),
                connections: Vec::new(),
                summary: "enabled, servers=0".to_string(),
            });
        }

        let total_servers = server_configs.len();
        let mut connections = Vec::<McpConnection>::new();
        let mut tools = Vec::<McpToolInfo>::new();
        let mut tool_statuses = Vec::<McpToolStatus>::new();
        let mut service_statuses = Vec::<McpServiceStatus>::new();
        let mut tool_index = HashMap::<String, ToolBinding>::new();
        let mut used_tool_names = HashSet::<String>::new();
        let mut tool_name_suffixes = HashMap::<String, usize>::new();
        let mut errors = Vec::<String>::new();

        for server in server_configs {
            let connect_outcome = connect_one_server(&server);
            let (transport, initialize_result, tools_result) = match connect_outcome {
                Ok(value) => value,
                Err(err) => {
                    let masked_error = mask_sensitive(&err.to_string());
                    errors.push(format!("{}: {}", server.name, masked_error));
                    service_statuses.push(McpServiceStatus {
                        name: server.name.clone(),
                        transport: transport_mode_name(server.transport).to_string(),
                        target: server_target_display(&server),
                        args_count: server.args.len(),
                        timeout_seconds: server.timeout.as_secs(),
                        available: false,
                        tool_count: 0,
                        summary: Some(server_summary(&server)),
                        error: Some(masked_error),
                    });
                    continue;
                }
            };

            let connection_idx = connections.len();
            let mut server_tools = extract_tools(&server.name, &tools_result);
            ensure_unique_ai_tool_names(
                &mut server_tools,
                &mut used_tool_names,
                &mut tool_name_suffixes,
            );
            let server_tool_count = server_tools.len();
            for tool in &server_tools {
                tool_statuses.push(McpToolStatus {
                    server_name: tool.server_name.clone(),
                    ai_name: tool.ai_name.clone(),
                    remote_name: tool.remote_name.clone(),
                    description: tool.description.clone(),
                    available: true,
                });
            }
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
            service_statuses.push(McpServiceStatus {
                name: server.name.clone(),
                transport: transport_mode_name(server.transport).to_string(),
                target: server_target_display(&server),
                args_count: server.args.len(),
                timeout_seconds: server.timeout.as_secs(),
                available: true,
                tool_count: server_tool_count,
                summary: Some(server_summary(&server)),
                error: None,
            });
            connections.push(McpConnection {
                name: server.name.clone(),
                timeout: server.timeout,
                transport,
            });

            let _ = initialize_result;
        }

        if connections.is_empty() {
            let summary = if errors.is_empty() {
                "enabled, servers=0".to_string()
            } else {
                format!("enabled, servers=0, unavailable: {}", errors.join(" | "))
            };
            return Ok(Self {
                enabled: true,
                tools: Vec::new(),
                tool_statuses,
                service_statuses,
                tool_index: HashMap::new(),
                connections: Vec::new(),
                summary,
            });
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
            tool_statuses,
            service_statuses,
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

    pub fn service_statuses(&self) -> Vec<McpServiceStatus> {
        self.service_statuses.clone()
    }

    pub fn tool_statuses(&self) -> Vec<McpToolStatus> {
        self.tool_statuses.clone()
    }

    pub fn startup_failures(&self) -> Vec<String> {
        self.service_statuses
            .iter()
            .filter(|item| !item.available)
            .filter_map(|item| {
                item.error.as_ref().map(|detail| {
                    format!(
                        "server={}, transport={}, error={detail}",
                        item.name, item.transport
                    )
                })
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
                terminate_mcp_child(&mut client.child);
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
        set_stdio_nonblocking(&stdout)?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }
}

impl Drop for StdioMcpClient {
    fn drop(&mut self) {
        terminate_mcp_child(&mut self.child);
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

impl LegacySseMcpClient {
    fn new(
        endpoint: &str,
        headers: &BTreeMap<String, String>,
        connect_timeout: Duration,
    ) -> Result<Self, AppError> {
        ensure_rustls_crypto_provider();
        let client = Client::builder()
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|err| AppError::Runtime(format!("failed to build MCP sse client: {err}")))?;
        let headers = parse_http_headers(headers)?;
        Ok(Self {
            client,
            sse_endpoint: endpoint.to_string(),
            headers,
            message_endpoint: None,
            session_id: None,
            event_rx: None,
            event_stop_tx: None,
            event_thread: None,
            next_id: 1,
        })
    }

    fn ensure_connected(&mut self, timeout: Duration) -> Result<(), AppError> {
        if self.event_rx.is_some() && self.message_endpoint.is_some() {
            return Ok(());
        }
        self.open_event_stream(timeout)?;
        self.wait_for_endpoint(timeout)
    }

    fn open_event_stream(&mut self, timeout: Duration) -> Result<(), AppError> {
        self.stop_event_stream_worker();
        let mut request_builder = self
            .client
            .get(&self.sse_endpoint)
            .headers(self.headers.clone())
            .header(ACCEPT, "text/event-stream");
        request_builder = request_builder.timeout(timeout);
        if let Some(session_id) = self.session_id.as_deref() {
            request_builder = request_builder.header("Mcp-Session-Id", session_id);
        }
        let response = request_builder
            .send()
            .map_err(|err| AppError::Runtime(format!("MCP sse connect failed: {err}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(AppError::Runtime(format!(
                "MCP sse connect failed: status={}, body={}",
                status,
                mask_sensitive(&trim_text_preview(&body, 1200))
            )));
        }
        update_sse_session_id(&mut self.session_id, response.headers());

        let (tx, rx) = mpsc::channel::<Result<LegacySseEvent, String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let worker = thread::spawn(move || {
            let mut reader = BufReader::new(response);
            loop {
                if stop_rx.try_recv().is_ok() {
                    break;
                }
                match read_sse_frame(&mut reader) {
                    Ok(Some(frame)) => {
                        if let Some(parsed) = parse_legacy_sse_event(frame)
                            && tx.send(Ok(parsed)).is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = tx.send(Err("MCP SSE stream closed by remote".to_string()));
                        break;
                    }
                    Err(err) => {
                        let _ = tx.send(Err(format!("failed to read MCP SSE stream: {err}")));
                        break;
                    }
                }
            }
        });
        self.event_rx = Some(rx);
        self.event_stop_tx = Some(stop_tx);
        self.event_thread = Some(worker);
        self.message_endpoint = None;
        Ok(())
    }

    fn wait_for_endpoint(&mut self, timeout: Duration) -> Result<(), AppError> {
        let started = Instant::now();
        loop {
            if started.elapsed() > timeout {
                self.reset_event_stream();
                return Err(AppError::Runtime(
                    "MCP sse handshake timeout: endpoint event not received".to_string(),
                ));
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            let event = self.recv_event(remaining)?;
            match event {
                LegacySseEvent::Endpoint(raw_endpoint) => {
                    let resolved = resolve_sse_message_endpoint(&self.sse_endpoint, &raw_endpoint)?;
                    self.message_endpoint = Some(resolved);
                    return Ok(());
                }
                LegacySseEvent::Message(_) => {
                    // ignore non-endpoint events during handshake
                }
            }
        }
    }

    fn recv_event(&mut self, timeout: Duration) -> Result<LegacySseEvent, AppError> {
        let Some(rx) = self.event_rx.as_ref() else {
            return Err(AppError::Runtime(
                "MCP sse stream not initialized".to_string(),
            ));
        };
        match rx.recv_timeout(timeout) {
            Ok(Ok(event)) => Ok(event),
            Ok(Err(detail)) => {
                self.reset_event_stream();
                Err(AppError::Runtime(detail))
            }
            Err(RecvTimeoutError::Timeout) => {
                self.reset_event_stream();
                Err(AppError::Runtime("MCP sse receive timeout".to_string()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.reset_event_stream();
                Err(AppError::Runtime("MCP sse stream disconnected".to_string()))
            }
        }
    }

    fn reset_event_stream(&mut self) {
        self.stop_event_stream_worker();
        self.event_rx = None;
        self.event_stop_tx = None;
        self.event_thread = None;
        self.message_endpoint = None;
    }

    fn stop_event_stream_worker(&mut self) {
        if let Some(stop_tx) = self.event_stop_tx.take() {
            let _ = stop_tx.send(());
        }
        if let Some(worker) = self.event_thread.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for LegacySseMcpClient {
    fn drop(&mut self) {
        self.stop_event_stream_worker();
    }
}

fn normalized_server_configs(
    cfg: &McpConfig,
    config_path: &Path,
) -> Result<Vec<NormalizedServerConfig>, AppError> {
    let file_path = resolve_mcp_servers_file_path(cfg, config_path);
    let map = load_mcp_server_map_from_file(file_path.as_path())?;
    normalized_server_configs_from_map(&map)
}

fn normalized_server_configs_from_map(
    map: &BTreeMap<String, McpServerConfig>,
) -> Result<Vec<NormalizedServerConfig>, AppError> {
    let mut keys = map.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    let mut out = Vec::new();
    for name in keys {
        let Some(item) = map.get(&name) else {
            continue;
        };
        if let Some(server) = normalize_server(name, item)? {
            out.push(server);
        }
    }
    Ok(out)
}

fn load_mcp_server_map_from_file(
    path: &Path,
) -> Result<BTreeMap<String, McpServerConfig>, AppError> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Config(format!(
            "failed to read MCP servers file {}: {err}",
            path.display()
        ))
    })?;
    if raw.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    let parsed: McpServersFile = serde_json::from_str(&raw).map_err(|err| {
        AppError::Config(format!(
            "failed to parse MCP servers JSON {}: {err}",
            path.display()
        ))
    })?;
    for name in parsed.mcp_servers.keys() {
        if name.trim().is_empty() {
            return Err(AppError::Config(format!(
                "MCP server name must not be empty in {}",
                path.display()
            )));
        }
    }
    Ok(parsed.mcp_servers)
}

fn write_mcp_server_map_to_file(
    path: &Path,
    map: &BTreeMap<String, McpServerConfig>,
) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Config(format!(
                "failed to create MCP directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let _lock = acquire_mcp_write_lock(path)?;
    let existing = read_existing_mcp_servers_json(path)?;
    let merged = merge_mcp_servers_json(existing, map)?;
    let content = serde_json::to_string_pretty(&merged)
        .map_err(|err| AppError::Config(format!("failed to serialize MCP servers JSON: {err}")))?;
    write_mcp_servers_atomically(path, &content).map_err(|err| {
        AppError::Config(format!(
            "failed to write MCP servers file {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

fn acquire_mcp_write_lock(path: &Path) -> Result<McpWriteLockGuard, AppError> {
    let lock_path = mcp_write_lock_path(path);
    let start = Instant::now();
    loop {
        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                let _ = writeln!(
                    file,
                    "pid={} ts={}",
                    std::process::id(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                );
                return Ok(McpWriteLockGuard { lock_path });
            }
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                if is_stale_mcp_lock(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                if start.elapsed() >= MCP_WRITE_LOCK_TIMEOUT {
                    return Err(AppError::Config(format!(
                        "timeout waiting MCP config write lock: {}",
                        lock_path.display()
                    )));
                }
                sleep(MCP_WRITE_LOCK_POLL_INTERVAL);
            }
            Err(err) => {
                return Err(AppError::Config(format!(
                    "failed to create MCP write lock {}: {err}",
                    lock_path.display()
                )));
            }
        }
    }
}

fn mcp_write_lock_path(path: &Path) -> PathBuf {
    let parent = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_base_dir);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("servers.json");
    parent.join(format!(".{file_name}.lock"))
}

fn is_stale_mcp_lock(lock_path: &Path) -> bool {
    let Ok(meta) = fs::metadata(lock_path) else {
        return false;
    };
    let Ok(modified) = meta.modified() else {
        return false;
    };
    let Ok(age) = SystemTime::now().duration_since(modified) else {
        return false;
    };
    age >= MCP_WRITE_LOCK_STALE_AFTER
}

fn read_existing_mcp_servers_json(path: &Path) -> Result<Option<Value>, AppError> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Config(format!(
            "failed to read existing MCP servers file {}: {err}",
            path.display()
        ))
    })?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let value = serde_json::from_str::<Value>(&raw).map_err(|err| {
        AppError::Config(format!(
            "failed to parse existing MCP servers JSON {}: {err}",
            path.display()
        ))
    })?;
    Ok(Some(value))
}

fn merge_mcp_servers_json(
    existing: Option<Value>,
    map: &BTreeMap<String, McpServerConfig>,
) -> Result<Value, AppError> {
    let mut root = existing
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let existing_servers = root
        .get("mcpServers")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut merged_servers = Map::<String, Value>::new();
    for (name, cfg) in map {
        let mut merged_obj = existing_servers
            .get(name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for key in known_mcp_server_json_keys() {
            merged_obj.remove(*key);
        }
        let known = serde_json::to_value(cfg).map_err(|err| {
            AppError::Config(format!("failed to encode MCP server '{}': {err}", name))
        })?;
        let Some(known_obj) = known.as_object() else {
            return Err(AppError::Config(format!(
                "failed to encode MCP server '{}': expected object",
                name
            )));
        };
        for (key, value) in known_obj {
            merged_obj.insert(key.to_string(), value.clone());
        }
        merged_servers.insert(name.to_string(), Value::Object(merged_obj));
    }
    root.insert("mcpServers".to_string(), Value::Object(merged_servers));
    Ok(Value::Object(root))
}

fn known_mcp_server_json_keys() -> &'static [&'static str] {
    &[
        "enabled",
        "transport",
        "type",
        "server-url",
        "server_url",
        "serverUrl",
        "url",
        "endpoint",
        "cmd",
        "command",
        "args",
        "env",
        "headers",
        "auth-type",
        "auth_type",
        "authType",
        "auth-token",
        "auth_token",
        "authToken",
        "timeout-seconds",
        "timeout_seconds",
        "timeoutSeconds",
    ]
}

fn write_mcp_servers_atomically(path: &Path, content: &str) -> std::io::Result<()> {
    let parent = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_base_dir);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("servers.json");
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::write(&temp_path, content)?;
    if let Err(rename_err) = fs::rename(&temp_path, path) {
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return Err(std::io::Error::new(
                copy_err.kind(),
                format!(
                    "failed to replace MCP servers file {} (rename: {}; copy fallback: {})",
                    path.display(),
                    rename_err,
                    copy_err
                ),
            ));
        }
        let _ = fs::remove_file(&temp_path);
    }
    Ok(())
}

fn normalize_server(
    name: String,
    server: &McpServerConfig,
) -> Result<Option<NormalizedServerConfig>, AppError> {
    if !server.enabled {
        return Ok(None);
    }
    let endpoint = non_empty_trimmed(server.server_url.as_deref())
        .or_else(|| non_empty_trimmed(server.endpoint.as_deref()));
    let command = non_empty_trimmed(server.command.as_deref());
    let transport = resolve_transport_mode(
        &name,
        server.transport.as_deref(),
        endpoint.is_some(),
        command.is_some(),
    )?;
    let timeout_seconds = server
        .timeout_seconds
        .unwrap_or(DEFAULT_MCP_TIMEOUT_SECONDS);
    if timeout_seconds == 0 {
        return Err(AppError::Config(format!(
            "mcp server '{}' timeout-seconds must be greater than 0",
            name
        )));
    }
    let mut headers = server.headers.clone();
    if transport == McpTransportMode::Http || transport == McpTransportMode::Sse {
        apply_auth_to_headers(
            &name,
            &mut headers,
            server.auth_type.as_deref(),
            server.auth_token.as_deref(),
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
        McpTransportMode::Sse => {
            let Some(endpoint) = server.endpoint.as_deref() else {
                return Err(AppError::Config(format!(
                    "mcp server '{}' transport=sse requires endpoint or server-url",
                    server.name
                )));
            };
            McpTransport::Sse(LegacySseMcpClient::new(
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

    notify_mcp(
        &mut transport,
        "notifications/initialized",
        json!({}),
        server.timeout,
    )?;
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
        McpTransport::Sse(client) => sse_request(client, method, params, timeout),
    }
}

fn notify_mcp(
    transport: &mut McpTransport,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<(), AppError> {
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
        McpTransport::Sse(client) => sse_notify(client, method, params, timeout),
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

    let started = Instant::now();
    loop {
        if started.elapsed() > timeout {
            return Err(AppError::Runtime(format!(
                "MCP request timeout: method={method}"
            )));
        }
        let frame = read_mcp_frame(
            &mut client.stdout,
            &mut client.child,
            method,
            timeout,
            started,
        )?;
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

fn sse_request(
    client: &mut LegacySseMcpClient,
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

    client.ensure_connected(timeout)?;
    let immediate = send_sse_post_request(client, &payload, method, timeout)?;
    if let Some(body) = immediate
        && let Some(resp_id) = body.get("id")
        && resp_id == &json!(id)
    {
        if let Some(err) = body.get("error") {
            return Err(AppError::Runtime(format!(
                "MCP returned error for method {}: {}",
                method,
                mask_sensitive(&err.to_string())
            )));
        }
        return Ok(body.get("result").cloned().unwrap_or_else(|| json!({})));
    }

    let started = Instant::now();
    loop {
        if started.elapsed() > timeout {
            client.reset_event_stream();
            return Err(AppError::Runtime(format!(
                "MCP request timeout: method={method}"
            )));
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        let event = client.recv_event(remaining)?;
        let LegacySseEvent::Message(frame) = event else {
            continue;
        };
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

fn sse_notify(
    client: &mut LegacySseMcpClient,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<(), AppError> {
    let payload = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    });
    client.ensure_connected(timeout)?;
    let _ = send_sse_post_request(client, &payload, "notification", timeout)?;
    Ok(())
}

fn send_sse_post_request(
    client: &mut LegacySseMcpClient,
    payload: &Value,
    method: &str,
    timeout: Duration,
) -> Result<Option<Value>, AppError> {
    let Some(message_endpoint) = client.message_endpoint.as_deref() else {
        return Err(AppError::Runtime(
            "MCP sse handshake failed: message endpoint is missing".to_string(),
        ));
    };

    let mut request_builder = client
        .client
        .post(message_endpoint)
        .headers(client.headers.clone())
        .json(payload)
        .timeout(timeout);
    if let Some(session_id) = client.session_id.as_deref() {
        request_builder = request_builder.header("Mcp-Session-Id", session_id);
    }
    let response = request_builder.send().map_err(|err| {
        client.reset_event_stream();
        AppError::Runtime(format!("MCP sse post failed: {err}"))
    })?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        client.reset_event_stream();
        return Err(AppError::Runtime(format!(
            "MCP sse post failed: status={}, body={}, method={}",
            status,
            mask_sensitive(&trim_text_preview(&body, 1200)),
            method
        )));
    }
    update_sse_session_id(&mut client.session_id, response.headers());
    let body = match parse_mcp_http_response(response) {
        Ok(body) => body,
        Err(err) => {
            client.reset_event_stream();
            return Err(err);
        }
    };
    if body.as_object().is_some_and(|obj| obj.is_empty()) {
        return Ok(None);
    }
    Ok(Some(body))
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
    update_sse_session_id(&mut client.session_id, headers);
}

fn update_sse_session_id(session_id: &mut Option<String>, headers: &HeaderMap) {
    if let Some(value) = headers
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        *session_id = Some(value.to_string());
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
    let trimmed = raw.trim();
    if !trimmed.is_empty()
        && let Some(value) = parse_sse_data_block(&[trimmed.to_string()])
    {
        return Ok(value);
    }

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

fn read_sse_frame<R: BufRead>(reader: &mut R) -> std::io::Result<Option<SseFrame>> {
    let mut event: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            if event.is_none() && data_lines.is_empty() {
                return Ok(None);
            }
            break;
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with(':') {
            continue;
        }
        if let Some(raw_event) = trimmed.strip_prefix("event:") {
            event = Some(raw_event.trim().to_string());
            continue;
        }
        if let Some(data) = trimmed.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
    Ok(Some(SseFrame { event, data_lines }))
}

fn parse_legacy_sse_event(frame: SseFrame) -> Option<LegacySseEvent> {
    let normalized_event = frame
        .event
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    if normalized_event.as_deref() == Some("endpoint") {
        return parse_legacy_sse_endpoint_block(&frame.data_lines).map(LegacySseEvent::Endpoint);
    }
    if normalized_event.is_none() || normalized_event.as_deref() == Some("message") {
        return parse_sse_data_block(&frame.data_lines).map(LegacySseEvent::Message);
    }
    parse_sse_data_block(&frame.data_lines).map(LegacySseEvent::Message)
}

fn parse_legacy_sse_endpoint_block(data_lines: &[String]) -> Option<String> {
    if data_lines.is_empty() {
        return None;
    }
    let payload = data_lines.join("\n");
    let trimmed = payload.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(trimmed).ok();
    if let Some(Value::String(endpoint)) = parsed.as_ref() {
        return non_empty_trimmed(Some(endpoint));
    }
    if let Some(parsed_value) = parsed {
        for key in [
            "endpoint",
            "url",
            "uri",
            "messageEndpoint",
            "message_endpoint",
        ] {
            if let Some(raw) = parsed_value.get(key).and_then(|value| value.as_str())
                && let Some(cleaned) = non_empty_trimmed(Some(raw))
            {
                return Some(cleaned);
            }
        }
    }
    non_empty_trimmed(Some(trimmed))
}

fn resolve_sse_message_endpoint(
    base_endpoint: &str,
    raw_endpoint: &str,
) -> Result<String, AppError> {
    let endpoint = non_empty_trimmed(Some(raw_endpoint)).ok_or_else(|| {
        AppError::Runtime("MCP sse handshake failed: endpoint event is empty".to_string())
    })?;
    if let Ok(url) = Url::parse(&endpoint) {
        return Ok(url.to_string());
    }
    let base = Url::parse(base_endpoint).map_err(|err| {
        AppError::Runtime(format!(
            "MCP sse handshake failed: invalid base endpoint '{}': {err}",
            mask_sensitive(base_endpoint)
        ))
    })?;
    let joined = base.join(&endpoint).map_err(|err| {
        AppError::Runtime(format!(
            "MCP sse handshake failed: cannot resolve message endpoint '{}': {err}",
            mask_sensitive(&endpoint)
        ))
    })?;
    Ok(joined.to_string())
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

fn read_mcp_frame(
    reader: &mut BufReader<ChildStdout>,
    child: &mut Child,
    method: &str,
    timeout: Duration,
    started: Instant,
) -> Result<Value, AppError> {
    let timeout_error = || AppError::Runtime(format!("MCP request timeout: method={method}"));
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        let n = loop {
            if started.elapsed() > timeout {
                return Err(timeout_error());
            }
            match reader.read_line(&mut line) {
                Ok(n) => break n,
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    if let Some(status) = child.try_wait().map_err(|e| {
                        AppError::Runtime(format!("failed to inspect MCP process state: {e}"))
                    })? {
                        return Err(AppError::Runtime(format!(
                            "MCP process exited before responding: method={}, status={}",
                            method, status
                        )));
                    }
                    sleep(MCP_STDIO_POLL_INTERVAL);
                }
                Err(err) => {
                    return Err(AppError::Runtime(format!(
                        "failed to read MCP header: {err}"
                    )));
                }
            }
        };
        if n == 0 {
            if let Some(status) = child.try_wait().map_err(|e| {
                AppError::Runtime(format!("failed to inspect MCP process state: {e}"))
            })? {
                return Err(AppError::Runtime(format!(
                    "MCP process exited before responding: method={}, status={}",
                    method, status
                )));
            }
            if started.elapsed() > timeout {
                return Err(timeout_error());
            }
            sleep(MCP_STDIO_POLL_INTERVAL);
            continue;
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
    let mut offset = 0;
    while offset < body.len() {
        if started.elapsed() > timeout {
            return Err(timeout_error());
        }
        match reader.read(&mut body[offset..]) {
            Ok(0) => {
                if let Some(status) = child.try_wait().map_err(|e| {
                    AppError::Runtime(format!("failed to inspect MCP process state: {e}"))
                })? {
                    return Err(AppError::Runtime(format!(
                        "MCP process exited before full response body: method={}, status={}",
                        method, status
                    )));
                }
                sleep(MCP_STDIO_POLL_INTERVAL);
            }
            Ok(n) => {
                offset += n;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                if let Some(status) = child.try_wait().map_err(|e| {
                    AppError::Runtime(format!("failed to inspect MCP process state: {e}"))
                })? {
                    return Err(AppError::Runtime(format!(
                        "MCP process exited before full response body: method={}, status={}",
                        method, status
                    )));
                }
                sleep(MCP_STDIO_POLL_INTERVAL);
            }
            Err(err) => {
                return Err(AppError::Runtime(format!("failed to read MCP body: {err}")));
            }
        }
    }
    let value = serde_json::from_slice::<Value>(&body)
        .map_err(|err| AppError::Runtime(format!("failed to parse MCP json: {err}")))?;
    Ok(value)
}

fn terminate_mcp_child(child: &mut Child) {
    if let Ok(None) = child.try_wait() {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg(unix)]
fn set_stdio_nonblocking(stdout: &ChildStdout) -> Result<(), AppError> {
    use std::os::fd::AsRawFd;
    let fd = stdout.as_raw_fd();
    // Non-blocking stdout allows timeout-seconds to actually take effect for stdio MCP.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(AppError::Runtime(
            "failed to get MCP stdout flags for nonblocking mode".to_string(),
        ));
    }
    let set_result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if set_result < 0 {
        return Err(AppError::Runtime(
            "failed to set MCP stdout nonblocking mode".to_string(),
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_stdio_nonblocking(_stdout: &ChildStdout) -> Result<(), AppError> {
    Ok(())
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
            "http" | "streamable_http" => {
                if has_endpoint {
                    Ok(McpTransportMode::Http)
                } else {
                    Err(AppError::Config(format!(
                        "mcp server '{}' transport={} requires endpoint or server-url",
                        server_name, normalized
                    )))
                }
            }
            "sse" => {
                if has_endpoint {
                    Ok(McpTransportMode::Sse)
                } else {
                    Err(AppError::Config(format!(
                        "mcp server '{}' transport=sse requires endpoint or server-url",
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
                "mcp server '{}' has invalid transport '{}', expected one of: http, streamable_http, sse, stdio",
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
        let params = normalize_tool_parameters_schema(select_tool_parameters_source(&tool));
        let sanitized_server = sanitize_tool_name(server_name);
        let sanitized_tool = sanitize_tool_name(remote_name);
        output.push(McpToolInfo {
            ai_name: format!("mcp__{sanitized_server}__{sanitized_tool}"),
            server_name: server_name.to_string(),
            remote_name: remote_name.to_string(),
            description: format!("MCP[{server_name}] tool {remote_name}: {description}"),
            parameters: params,
        });
    }
    output
}

fn ensure_unique_ai_tool_names(
    tools: &mut [McpToolInfo],
    used_tool_names: &mut HashSet<String>,
    tool_name_suffixes: &mut HashMap<String, usize>,
) {
    for tool in tools {
        let base = tool.ai_name.clone();
        if used_tool_names.insert(base.clone()) {
            continue;
        }
        let mut suffix = *tool_name_suffixes.get(&base).unwrap_or(&2);
        loop {
            let candidate = format!("{base}_{suffix}");
            suffix += 1;
            if used_tool_names.insert(candidate.clone()) {
                tool.ai_name = candidate;
                tool_name_suffixes.insert(base, suffix);
                break;
            }
        }
    }
}

fn select_tool_parameters_source(tool: &Value) -> Option<&Value> {
    let source = tool
        .get("inputSchema")
        .or_else(|| tool.get("input_schema"))
        .or_else(|| tool.get("parameters"))?;
    if let Some(obj) = source.as_object() {
        if let Some(schema) = obj.get("jsonSchema") {
            return Some(schema);
        }
        if let Some(schema) = obj.get("schema") {
            return Some(schema);
        }
    }
    Some(source)
}

fn default_tool_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

fn normalize_tool_parameters_schema(raw: Option<&Value>) -> Value {
    let Some(value) = raw else {
        return default_tool_parameters_schema();
    };
    let mut schema = match value {
        Value::Object(map) => Value::Object(map.clone()),
        _ => return default_tool_parameters_schema(),
    };
    let Some(map) = schema.as_object_mut() else {
        return default_tool_parameters_schema();
    };

    // OpenAI function calling requires top-level JSON schema to be object.
    let root_is_object = match map.get("type") {
        Some(Value::String(kind)) => kind.eq_ignore_ascii_case("object"),
        Some(Value::Array(kinds)) => kinds.iter().any(|item| {
            item.as_str()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("object"))
        }),
        _ => false,
    };
    if !root_is_object || !matches!(map.get("type"), Some(Value::String(_))) {
        map.insert("type".to_string(), Value::String("object".to_string()));
    }
    if !map.get("properties").is_some_and(Value::is_object) {
        map.insert("properties".to_string(), Value::Object(Map::new()));
    }
    if map.get("required").is_some() && !map.get("required").is_some_and(Value::is_array) {
        map.remove("required");
    }
    if !map.contains_key("additionalProperties") {
        map.insert("additionalProperties".to_string(), Value::Bool(true));
    }
    schema
}

fn transport_name(transport: &McpTransport) -> &'static str {
    match transport {
        McpTransport::Stdio(_) => "stdio",
        McpTransport::Http(_) => "http",
        McpTransport::Sse(_) => "sse",
    }
}

fn transport_mode_name(mode: McpTransportMode) -> &'static str {
    match mode {
        McpTransportMode::Stdio => "stdio",
        McpTransportMode::Http => "http",
        McpTransportMode::Sse => "sse",
    }
}

fn server_target_display(server: &NormalizedServerConfig) -> String {
    match server.transport {
        McpTransportMode::Http | McpTransportMode::Sse => server
            .endpoint
            .as_deref()
            .map(mask_sensitive)
            .unwrap_or_else(|| "-".to_string()),
        McpTransportMode::Stdio => server
            .command
            .as_deref()
            .map(mask_sensitive)
            .unwrap_or_else(|| "-".to_string()),
    }
}

fn server_summary(server: &NormalizedServerConfig) -> String {
    format!(
        "tools=dynamic env={} headers={} timeout={}s",
        server.env.len(),
        server.headers.len(),
        server.timeout.as_secs()
    )
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

pub(crate) fn parse_json_object_arguments(raw_arguments: &str) -> Result<Value, String> {
    let trimmed = raw_arguments.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }
    let stripped = strip_markdown_code_fence(trimmed);
    let mut last_error: Option<String> = None;
    for candidate in [trimmed, stripped] {
        if let Some(parsed) = try_parse_object_candidate(candidate, &mut last_error) {
            return Ok(parsed);
        }
        if let Some(parsed) = try_parse_stringified_object_candidate(candidate, &mut last_error) {
            return Ok(parsed);
        }
    }
    for candidate in [stripped, trimmed] {
        if let Some(json_object) = extract_first_balanced_json_object(candidate)
            && let Some(parsed) = try_parse_object_candidate(json_object, &mut last_error)
        {
            return Ok(parsed);
        }
    }
    Err(last_error.unwrap_or_else(|| {
        "expected a strict JSON object with double-quoted keys/strings".to_string()
    }))
}

fn strip_markdown_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }
    let without_prefix = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```JSON")
        .trim_start_matches("```");
    without_prefix.trim().trim_end_matches("```").trim()
}

fn try_parse_object_candidate(candidate: &str, last_error: &mut Option<String>) -> Option<Value> {
    match serde_json::from_str::<Value>(candidate) {
        Ok(Value::Object(object)) => Some(Value::Object(object)),
        Ok(_) => {
            *last_error = Some("expected a strict JSON object".to_string());
            None
        }
        Err(err) => {
            *last_error = Some(err.to_string());
            None
        }
    }
}

fn try_parse_stringified_object_candidate(
    candidate: &str,
    last_error: &mut Option<String>,
) -> Option<Value> {
    let decoded = match serde_json::from_str::<Value>(candidate) {
        Ok(Value::String(inner)) => inner,
        Ok(_) => return None,
        Err(err) => {
            *last_error = Some(err.to_string());
            return None;
        }
    };
    try_parse_object_candidate(decoded.trim(), last_error)
}

fn extract_first_balanced_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaping = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaping {
                escaping = false;
                continue;
            }
            match ch {
                '\\' => escaping = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    let end = start + offset;
                    return Some(&text[start..=end]);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_mcp_tool_arguments(raw_arguments: &str) -> Result<Value, AppError> {
    parse_json_object_arguments(raw_arguments).map_err(|err| {
        AppError::Runtime(format!(
            "invalid MCP tool arguments JSON: {}; expected a strict JSON object with double-quoted keys/strings, raw={}",
            err,
            mask_sensitive(&trim_text_preview(raw_arguments.trim(), 240))
        ))
    })
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
    use serde_json::{Value, json};
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        fs,
        path::{Path, PathBuf},
        sync::mpsc,
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use super::{
        LegacySseMcpClient, McpManager, McpServerRecord, McpToolInfo, McpTransportMode,
        ensure_unique_ai_tool_names, extract_tools, mcp_summary, mcp_write_lock_path,
        non_empty_trimmed, normalize_server, parse_legacy_sse_endpoint_block,
        parse_mcp_tool_arguments, parse_sse_jsonrpc_payload, resolve_mcp_servers_file_path,
        resolve_sse_message_endpoint, resolve_transport_mode, save_mcp_server_records,
        validate_mcp_config, write_mcp_server_map_to_file, write_mcp_servers_atomically,
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
                .contains("invalid MCP tool arguments JSON: expected a strict JSON object")
        );
    }

    #[test]
    fn parse_mcp_tool_arguments_accepts_markdown_code_fence() {
        let parsed = parse_mcp_tool_arguments("```json\n{\"command\":\"pwd\"}\n```")
            .expect("code fence should be unwrapped");
        assert_eq!(parsed, json!({"command":"pwd"}));
    }

    #[test]
    fn parse_mcp_tool_arguments_accepts_json_string_wrapped_object() {
        let parsed = parse_mcp_tool_arguments("\"{\\\"command\\\":\\\"pwd\\\"}\"")
            .expect("stringified object should be decoded");
        assert_eq!(parsed, json!({"command":"pwd"}));
    }

    #[test]
    fn parse_mcp_tool_arguments_accepts_embedded_object_text() {
        let parsed =
            parse_mcp_tool_arguments("tool args => {\"command\":\"pwd\",\"mode\":\"read\"} end")
                .expect("embedded object should be extracted");
        assert_eq!(parsed, json!({"command":"pwd","mode":"read"}));
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
        let normalized = normalize_server("remote".to_string(), &server)
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
    fn resolve_transport_mode_accepts_sse_transport_mode() {
        let transport = resolve_transport_mode("srv", Some("sse"), true, false).expect("transport");
        assert_eq!(transport, McpTransportMode::Sse);
    }

    #[test]
    fn resolve_transport_mode_accepts_streamable_http_alias() {
        let transport =
            resolve_transport_mode("srv", Some("streamable_http"), true, false).expect("transport");
        assert_eq!(transport, McpTransportMode::Http);
    }

    #[test]
    fn parse_legacy_sse_endpoint_block_accepts_string_or_json() {
        let plain = parse_legacy_sse_endpoint_block(&["/messages?session=1".to_string()]);
        assert_eq!(plain.as_deref(), Some("/messages?session=1"));

        let json_endpoint =
            parse_legacy_sse_endpoint_block(
                &["{\"endpoint\":\"/messages?session=2\"}".to_string()],
            );
        assert_eq!(json_endpoint.as_deref(), Some("/messages?session=2"));
    }

    #[test]
    fn parse_sse_jsonrpc_payload_accepts_plain_json_body() {
        let parsed = parse_sse_jsonrpc_payload(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#)
            .expect("plain json body should parse");
        assert_eq!(
            parsed
                .get("result")
                .and_then(|value| value.get("ok"))
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn ensure_unique_ai_tool_names_adds_suffix_for_collisions() {
        let mut tools = vec![
            McpToolInfo {
                ai_name: "mcp__server__tool".to_string(),
                server_name: "s1".to_string(),
                remote_name: "tool-a".to_string(),
                description: "d".to_string(),
                parameters: json!({}),
            },
            McpToolInfo {
                ai_name: "mcp__server__tool".to_string(),
                server_name: "s1".to_string(),
                remote_name: "tool_b".to_string(),
                description: "d".to_string(),
                parameters: json!({}),
            },
            McpToolInfo {
                ai_name: "mcp__server__tool".to_string(),
                server_name: "s2".to_string(),
                remote_name: "tool c".to_string(),
                description: "d".to_string(),
                parameters: json!({}),
            },
        ];
        let mut used_tool_names = HashSet::new();
        let mut tool_name_suffixes = HashMap::new();
        ensure_unique_ai_tool_names(&mut tools, &mut used_tool_names, &mut tool_name_suffixes);
        let names = tools
            .iter()
            .map(|tool| tool.ai_name.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "mcp__server__tool".to_string(),
                "mcp__server__tool_2".to_string(),
                "mcp__server__tool_3".to_string()
            ]
        );
    }

    #[test]
    fn sse_recv_event_timeout_resets_stream_state() {
        let mut client = LegacySseMcpClient::new(
            "https://example.com/sse",
            &BTreeMap::new(),
            Duration::from_secs(1),
        )
        .expect("client");
        let (_tx, rx) = mpsc::channel();
        client.event_rx = Some(rx);
        client.message_endpoint = Some("https://example.com/messages".to_string());

        let err = client
            .recv_event(Duration::from_millis(1))
            .expect_err("timeout should fail");
        assert!(err.to_string().contains("MCP sse receive timeout"));
        assert!(client.event_rx.is_none());
        assert!(client.message_endpoint.is_none());
    }

    #[test]
    fn sse_reset_event_stream_stops_background_worker() {
        let mut client = LegacySseMcpClient::new(
            "https://example.com/sse",
            &BTreeMap::new(),
            Duration::from_secs(1),
        )
        .expect("client");
        let (_event_tx, event_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _ = stop_rx.recv_timeout(Duration::from_secs(1));
        });

        client.event_rx = Some(event_rx);
        client.event_stop_tx = Some(stop_tx);
        client.event_thread = Some(worker);
        client.message_endpoint = Some("https://example.com/messages".to_string());

        client.reset_event_stream();
        assert!(client.event_rx.is_none());
        assert!(client.event_stop_tx.is_none());
        assert!(client.event_thread.is_none());
        assert!(client.message_endpoint.is_none());
    }

    #[test]
    fn resolve_sse_message_endpoint_supports_relative_paths() {
        let resolved =
            resolve_sse_message_endpoint("https://example.com/mcp/sse", "/messages?session=abc")
                .expect("relative endpoint should resolve");
        assert_eq!(resolved, "https://example.com/messages?session=abc");
    }

    #[test]
    fn extract_tools_normalizes_invalid_or_missing_schema_type() {
        let result = json!({
            "tools": [
                {
                    "name": "weibo_news",
                    "description": "news",
                    "inputSchema": {
                        "type": null
                    }
                }
            ]
        });
        let tools = extract_tools("real-time_news", &result);
        assert_eq!(tools.len(), 1);
        let schema = &tools[0].parameters;
        assert_eq!(schema.get("type").and_then(Value::as_str), Some("object"));
        assert!(schema.get("properties").is_some_and(Value::is_object));
        assert!(schema.get("additionalProperties").is_some());
    }

    #[test]
    fn extract_tools_accepts_nested_json_schema_source() {
        let result = json!({
            "tools": [
                {
                    "name": "headline",
                    "inputSchema": {
                        "jsonSchema": {
                            "properties": {
                                "keyword": { "type": "string" }
                            }
                        }
                    }
                }
            ]
        });
        let tools = extract_tools("real-time_news", &result);
        assert_eq!(tools.len(), 1);
        let schema = &tools[0].parameters;
        assert_eq!(schema.get("type").and_then(Value::as_str), Some("object"));
        assert_eq!(
            schema
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|props| props.get("keyword"))
                .and_then(|item| item.get("type"))
                .and_then(Value::as_str),
            Some("string")
        );
    }

    fn new_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("machineclaw-{prefix}-{stamp}"));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    fn mcp_cfg_with_dir(dir: &Path) -> McpConfig {
        McpConfig {
            enabled: true,
            mcp_availability_check_mode: "rsync".to_string(),
            dir: dir.display().to_string(),
        }
    }

    fn write_servers(
        cfg: &McpConfig,
        config_path: &Path,
        servers: BTreeMap<String, McpServerConfig>,
    ) {
        let file_path = resolve_mcp_servers_file_path(cfg, config_path);
        write_mcp_server_map_to_file(file_path.as_path(), &servers).expect("write servers json");
    }

    #[test]
    fn save_mcp_server_records_preserves_unknown_fields() {
        let dir = new_temp_dir("mcp-preserve-unknown");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        let file_path = resolve_mcp_servers_file_path(&cfg, &config_path);
        let raw = r#"{
  "theme": "dark",
  "mcpServers": {
    "demo": {
      "transport": "sse",
      "url": "https://example.com/sse",
      "x-meta": {
        "source": "imported"
      }
    }
  },
  "x-global": {
    "keep": true
  }
}"#;
        fs::write(&file_path, raw).expect("seed existing mcp json");
        let records = vec![McpServerRecord {
            name: "demo".to_string(),
            config: McpServerConfig {
                enabled: true,
                transport: Some("http".to_string()),
                server_url: Some("https://example.com/mcp".to_string()),
                endpoint: None,
                command: None,
                args: vec![],
                env: BTreeMap::new(),
                headers: BTreeMap::new(),
                auth_type: None,
                auth_token: None,
                timeout_seconds: Some(9),
            },
        }];

        save_mcp_server_records(&cfg, &config_path, &records).expect("save records");
        let updated = fs::read_to_string(&file_path).expect("read updated json");
        let parsed: Value = serde_json::from_str(&updated).expect("parse updated json");
        let root = parsed.as_object().expect("root object");
        assert_eq!(
            root.get("theme").and_then(Value::as_str),
            Some("dark"),
            "top-level unknown field should be preserved"
        );
        assert!(
            root.get("x-global").is_some(),
            "top-level extension fields should be preserved"
        );
        let server = root
            .get("mcpServers")
            .and_then(Value::as_object)
            .and_then(|servers| servers.get("demo"))
            .and_then(Value::as_object)
            .expect("demo server");
        assert!(
            !server.contains_key("transport"),
            "legacy alias key should be normalized away"
        );
        assert_eq!(server.get("type").and_then(Value::as_str), Some("http"));
        assert_eq!(
            server.get("timeoutSeconds").and_then(Value::as_u64),
            Some(9)
        );
        assert!(
            server.get("x-meta").is_some(),
            "server-level unknown extension fields should be preserved"
        );
    }

    #[test]
    fn save_mcp_server_records_cleans_up_lock_file() {
        let dir = new_temp_dir("mcp-lock-cleanup");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        let lock_path =
            mcp_write_lock_path(resolve_mcp_servers_file_path(&cfg, &config_path).as_path());
        let records = vec![McpServerRecord {
            name: "demo".to_string(),
            config: McpServerConfig {
                enabled: true,
                transport: Some("stdio".to_string()),
                command: Some("echo".to_string()),
                ..McpServerConfig::default()
            },
        }];
        save_mcp_server_records(&cfg, &config_path, &records).expect("save records");
        assert!(
            !lock_path.exists(),
            "lock file should be removed after write completes"
        );
    }

    #[test]
    fn write_mcp_servers_atomically_does_not_remove_existing_directory_target() {
        let dir = new_temp_dir("mcp-write-fallback");
        let target = dir.join("servers.json");
        fs::create_dir_all(&target).expect("create target directory");
        let err =
            write_mcp_servers_atomically(&target, "{}").expect_err("directory target must fail");
        assert!(
            err.to_string().contains("copy fallback")
                || err.to_string().contains("Is a directory")
                || err.to_string().contains("is a directory")
        );
        assert!(target.is_dir(), "target directory must stay untouched");
        let dangling_tmp = fs::read_dir(&dir)
            .expect("read dir")
            .flatten()
            .any(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                name.starts_with(".servers.json.") && name.ends_with(".tmp")
            });
        assert!(!dangling_tmp, "temporary files should be cleaned up");
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
        let dir = new_temp_dir("mcp-invalid-transport");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        write_servers(&cfg, &config_path, servers);
        let err = validate_mcp_config(&cfg, &config_path).expect_err("invalid transport must fail");
        assert!(err.to_string().contains("invalid transport"));
    }

    #[test]
    fn validate_mcp_config_allows_enabled_without_any_server() {
        let dir = new_temp_dir("mcp-empty-validate");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        validate_mcp_config(&cfg, &config_path)
            .expect("enabled mcp without servers should be allowed");
        assert_eq!(mcp_summary(&cfg, &config_path), "enabled, servers=0");
    }

    #[test]
    fn connect_allows_enabled_without_any_server() {
        let dir = new_temp_dir("mcp-empty-connect");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        let manager = McpManager::connect(&cfg, &config_path)
            .expect("connect should allow empty mcp server set");
        assert_eq!(manager.summary(), "enabled, servers=0");
        assert!(manager.external_tool_definitions().is_empty());
    }

    #[test]
    fn connect_degrades_when_all_servers_are_unavailable() {
        let mut servers = BTreeMap::new();
        servers.insert(
            "broken".to_string(),
            McpServerConfig {
                enabled: true,
                transport: Some("stdio".to_string()),
                server_url: None,
                endpoint: None,
                command: Some("__machineclaw_missing_mcp_command__".to_string()),
                args: vec![],
                env: BTreeMap::new(),
                headers: BTreeMap::new(),
                auth_type: None,
                auth_token: None,
                timeout_seconds: Some(1),
            },
        );
        let dir = new_temp_dir("mcp-degrade");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        write_servers(&cfg, &config_path, servers);
        let manager = McpManager::connect(&cfg, &config_path)
            .expect("all unavailable servers should not block chat startup");
        assert!(
            manager
                .summary()
                .contains("enabled, servers=0, unavailable:")
        );
        assert_eq!(manager.startup_failures().len(), 1);
        assert!(manager.external_tool_definitions().is_empty());
    }

    #[test]
    fn mcp_summary_uses_json_server_name() {
        let dir = new_temp_dir("mcp-summary");
        let config_path = dir.join("claw.toml");
        let cfg = mcp_cfg_with_dir(&dir);
        let mut servers = BTreeMap::new();
        servers.insert(
            "local".to_string(),
            McpServerConfig {
                enabled: true,
                transport: Some("http".to_string()),
                server_url: Some("http://127.0.0.1:8080/mcp".to_string()),
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
        write_servers(&cfg, &config_path, servers);
        let summary = mcp_summary(&cfg, &config_path);
        assert!(summary.contains("local:http="));
        assert!(!summary.contains("root:http="));
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
