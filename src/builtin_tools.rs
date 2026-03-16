use std::{
    collections::VecDeque,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use glob::{MatchOptions, Pattern, glob_with};
use regex::Regex;
use reqwest::{
    blocking::Client,
    header::{ACCEPT, USER_AGENT},
};
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    ai::ExternalToolDefinition, config::BuiltinToolsConfig, mcp::parse_json_object_arguments,
};

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const DEFAULT_TOOL_LIMIT: usize = 100;
const DEFAULT_WEB_RESULT_LIMIT: usize = 5;
const MAX_LINE_PREVIEW_CHARS: usize = 500;
const WEB_SEARCH_USER_AGENT: &str =
    "MachineClaw/1.0 (+https://github.com/machineclaw; builtin-web-search)";
const MAX_WEB_SNIPPET_CHARS: usize = 280;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinToolKind {
    View,
    Ls,
    Glob,
    Grep,
    WebSearch,
    Think,
    Edit,
    Replace,
    NotebookEditCell,
    Task,
    Architect,
}

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    #[serde(alias = "path")]
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ListFilesArgs {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    recursive: bool,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GlobSearchArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GrepSearchArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    case_sensitive: bool,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ThinkArgs {
    thought: String,
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    #[serde(alias = "path")]
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
    #[serde(default)]
    apply: bool,
}

#[derive(Debug, Deserialize)]
struct ReplaceArgs {
    #[serde(alias = "path")]
    file_path: String,
    content: String,
    #[serde(default)]
    apply: bool,
}

#[derive(Debug, Deserialize)]
struct NotebookEditCellArgs {
    #[serde(alias = "path")]
    file_path: String,
    cell_index: usize,
    content: String,
    #[serde(default)]
    apply: bool,
}

#[derive(Debug, Deserialize)]
struct TaskArgs {
    description: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    acceptance_criteria: Option<String>,
    #[serde(default)]
    evidence: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArchitectArgs {
    problem: String,
}

pub fn external_tool_definitions(cfg: &BuiltinToolsConfig) -> Vec<ExternalToolDefinition> {
    if !cfg.enabled {
        return Vec::new();
    }
    let mut tools = vec![
        ExternalToolDefinition {
            name: "View".to_string(),
            description: "Read local file content with line offset/limit support.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "file_path":{"type":"string","description":"Absolute or relative path to file"},
                    "offset":{"type":"integer","minimum":0},
                    "limit":{"type":"integer","minimum":1}
                },
                "required":["file_path"],
                "additionalProperties": false
            }),
        },
        ExternalToolDefinition {
            name: "LS".to_string(),
            description: "List files and directories under a path.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","description":"Absolute or relative directory path"},
                    "recursive":{"type":"boolean"},
                    "include_hidden":{"type":"boolean"},
                    "limit":{"type":"integer","minimum":1}
                },
                "additionalProperties": false
            }),
        },
        ExternalToolDefinition {
            name: "GlobTool".to_string(),
            description: "Find paths by glob pattern.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string"},
                    "path":{"type":"string","description":"Base path for relative pattern"},
                    "limit":{"type":"integer","minimum":1}
                },
                "required":["pattern"],
                "additionalProperties": false
            }),
        },
        ExternalToolDefinition {
            name: "GrepTool".to_string(),
            description: "Search file contents with regex pattern recursively.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string"},
                    "path":{"type":"string","description":"Base path to search"},
                    "glob":{"type":"string","description":"Optional file path glob filter"},
                    "case_sensitive":{"type":"boolean"},
                    "limit":{"type":"integer","minimum":1}
                },
                "required":["pattern"],
                "additionalProperties": false
            }),
        },
        ExternalToolDefinition {
            name: "Task".to_string(),
            description: "Plan a sub-task with objective and acceptance criteria.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "description":{"type":"string"},
                    "task_id":{"type":"string","description":"Stable task identifier for updates"},
                    "status":{"type":"string","enum":["running","done","failed","blocked"]},
                    "acceptance_criteria":{"type":"string"},
                    "evidence":{"type":"string"}
                },
                "required":["description"],
                "additionalProperties": false
            }),
        },
        ExternalToolDefinition {
            name: "Architect".to_string(),
            description: "Draft architecture options and tradeoffs for a coding task.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "problem":{"type":"string"}
                },
                "required":["problem"],
                "additionalProperties": false
            }),
        },
    ];
    if cfg.web_search_enabled {
        tools.push(ExternalToolDefinition {
            name: "WebSearch".to_string(),
            description: "Search the web for recent/public information.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "query":{"type":"string"},
                    "max_results":{"type":"integer","minimum":1}
                },
                "required":["query"],
                "additionalProperties": false
            }),
        });
    }
    if cfg.write_tools_enabled {
        tools.push(ExternalToolDefinition {
            name: "Edit".to_string(),
            description: "Replace text in a file. Requires apply=true.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "file_path":{"type":"string"},
                    "old_string":{"type":"string"},
                    "new_string":{"type":"string"},
                    "replace_all":{"type":"boolean"},
                    "apply":{"type":"boolean"}
                },
                "required":["file_path","old_string","new_string"],
                "additionalProperties": false
            }),
        });
        tools.push(ExternalToolDefinition {
            name: "Replace".to_string(),
            description: "Replace full file content. Requires apply=true.".to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "file_path":{"type":"string"},
                    "content":{"type":"string"},
                    "apply":{"type":"boolean"}
                },
                "required":["file_path","content"],
                "additionalProperties": false
            }),
        });
        tools.push(ExternalToolDefinition {
            name: "NotebookEditCell".to_string(),
            description: "Edit source content of one Jupyter notebook cell. Requires apply=true."
                .to_string(),
            parameters: json!({
                "type":"object",
                "properties":{
                    "file_path":{"type":"string"},
                    "cell_index":{"type":"integer","minimum":0},
                    "content":{"type":"string"},
                    "apply":{"type":"boolean"}
                },
                "required":["file_path","cell_index","content"],
                "additionalProperties": false
            }),
        });
    }
    tools
}

pub fn tool_names(cfg: &BuiltinToolsConfig) -> Vec<String> {
    external_tool_definitions(cfg)
        .into_iter()
        .map(|item| item.name)
        .collect()
}

pub fn is_builtin_tool(name: &str) -> bool {
    parse_tool_kind(name).is_some()
}

pub fn is_silent_tool(name: &str) -> bool {
    matches!(parse_tool_kind(name), Some(BuiltinToolKind::Think))
}

pub fn execute_tool(
    name: &str,
    raw_arguments: &str,
    cfg: &BuiltinToolsConfig,
) -> Result<String, String> {
    if !cfg.enabled {
        return Err("builtin tools are disabled (ai.tools.builtin.enabled=false)".to_string());
    }
    let Some(kind) = parse_tool_kind(name) else {
        return Err(format!("unsupported builtin tool: {name}"));
    };
    let payload = match kind {
        BuiltinToolKind::View => execute_read_file(raw_arguments, cfg)?,
        BuiltinToolKind::Ls => execute_list_files(raw_arguments, cfg)?,
        BuiltinToolKind::Glob => execute_glob_search(raw_arguments, cfg)?,
        BuiltinToolKind::Grep => execute_grep_search(raw_arguments, cfg)?,
        BuiltinToolKind::WebSearch => execute_web_search(raw_arguments, cfg)?,
        BuiltinToolKind::Think => execute_think(raw_arguments)?,
        BuiltinToolKind::Edit => execute_edit(raw_arguments, cfg)?,
        BuiltinToolKind::Replace => execute_replace(raw_arguments, cfg)?,
        BuiltinToolKind::NotebookEditCell => execute_notebook_edit_cell(raw_arguments, cfg)?,
        BuiltinToolKind::Task => execute_task(raw_arguments)?,
        BuiltinToolKind::Architect => execute_architect(raw_arguments)?,
    };
    Ok(payload.to_string())
}

fn parse_tool_kind(name: &str) -> Option<BuiltinToolKind> {
    let normalized = name.trim().to_ascii_lowercase().replace(['_', '-'], "");
    match normalized.as_str() {
        "view" | "readfile" => Some(BuiltinToolKind::View),
        "ls" | "listfiles" => Some(BuiltinToolKind::Ls),
        "globtool" | "globsearch" => Some(BuiltinToolKind::Glob),
        "greptool" | "grepsearch" => Some(BuiltinToolKind::Grep),
        "websearch" => Some(BuiltinToolKind::WebSearch),
        "think" => Some(BuiltinToolKind::Think),
        "edit" | "editfile" => Some(BuiltinToolKind::Edit),
        "replace" | "replacefile" => Some(BuiltinToolKind::Replace),
        "notebookeditcell" => Some(BuiltinToolKind::NotebookEditCell),
        "task" => Some(BuiltinToolKind::Task),
        "architect" => Some(BuiltinToolKind::Architect),
        _ => None,
    }
}

fn execute_read_file(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    let args: ReadFileArgs = parse_tool_args(raw_arguments)?;
    let file_path = resolve_user_path(args.file_path.as_str(), cfg, false)?;
    let (bytes, truncated_by_bytes) =
        read_file_with_limit(file_path.as_path(), cfg.max_read_bytes)?;
    let content = String::from_utf8_lossy(bytes.as_slice()).to_string();
    let lines = content.lines().collect::<Vec<_>>();
    let start = args.offset.unwrap_or(0).min(lines.len());
    let line_limit = args.limit.unwrap_or(DEFAULT_READ_LINE_LIMIT).max(1);
    let end = start.saturating_add(line_limit).min(lines.len());
    let selected = if start >= end {
        String::new()
    } else {
        lines[start..end].join("\n")
    };
    Ok(json!({
        "ok": true,
        "tool": "View",
        "path": file_path.display().to_string(),
        "offset": start,
        "returned_lines": end.saturating_sub(start),
        "total_lines": lines.len(),
        "truncated_by_bytes": truncated_by_bytes,
        "content": selected
    }))
}

fn execute_list_files(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    let args: ListFilesArgs = parse_tool_args(raw_arguments)?;
    let path_raw = args.path.unwrap_or_else(|| ".".to_string());
    let base = resolve_user_path(path_raw.as_str(), cfg, false)?;
    if !base.is_dir() {
        return Err(format!("path is not a directory: {}", base.display()));
    }
    let max = normalized_limit(args.limit, cfg.max_search_results, DEFAULT_TOOL_LIMIT);
    let mut rows = Vec::<Value>::new();
    let mut truncated = false;
    if args.recursive {
        let mut queue = VecDeque::from([base.clone()]);
        while let Some(dir) = queue.pop_front() {
            let mut children = read_dir_sorted(dir.as_path())?;
            for child in children.drain(..) {
                if rows.len() >= max {
                    truncated = true;
                    break;
                }
                let file_name = child.file_name().to_string_lossy().to_string();
                if !args.include_hidden && file_name.starts_with('.') {
                    continue;
                }
                let path = child.path();
                let file_type = child.file_type().map_err(|err| {
                    format!("failed to inspect file type {}: {err}", path.display())
                })?;
                let kind = file_kind_from_file_type(file_type);
                rows.push(json!({
                    "path": path.display().to_string(),
                    "name": file_name,
                    "kind": kind
                }));
                if file_type.is_dir() {
                    queue.push_back(path);
                }
            }
            if rows.len() >= max {
                break;
            }
        }
    } else {
        let children = read_dir_sorted(base.as_path())?;
        for child in children {
            if rows.len() >= max {
                truncated = true;
                break;
            }
            let file_name = child.file_name().to_string_lossy().to_string();
            if !args.include_hidden && file_name.starts_with('.') {
                continue;
            }
            let path = child.path();
            let file_type = child
                .file_type()
                .map_err(|err| format!("failed to inspect file type {}: {err}", path.display()))?;
            rows.push(json!({
                "path": path.display().to_string(),
                "name": file_name,
                "kind": file_kind_from_file_type(file_type)
            }));
        }
    }
    Ok(json!({
        "ok": true,
        "tool": "LS",
        "path": base.display().to_string(),
        "recursive": args.recursive,
        "count": rows.len(),
        "truncated": truncated,
        "entries": rows
    }))
}

fn execute_glob_search(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    let args: GlobSearchArgs = parse_tool_args(raw_arguments)?;
    let base = resolve_user_path(args.path.as_deref().unwrap_or("."), cfg, false)?;
    let pattern_raw = args.pattern.trim();
    if pattern_raw.is_empty() {
        return Err("pattern is empty".to_string());
    }
    let full_pattern = if Path::new(pattern_raw).is_absolute() {
        pattern_raw.to_string()
    } else {
        base.join(pattern_raw).to_string_lossy().to_string()
    };
    let max = normalized_limit(args.limit, cfg.max_search_results, DEFAULT_TOOL_LIMIT);
    let options = MatchOptions {
        case_sensitive: true,
        require_literal_separator: false,
        require_literal_leading_dot: false,
    };
    let mut matches = Vec::<String>::new();
    let mut truncated = false;
    let iter = glob_with(full_pattern.as_str(), options)
        .map_err(|err| format!("invalid glob pattern '{}': {err}", pattern_raw))?;
    let workspace_root = if cfg.workspace_only {
        Some(canonical_workspace_root()?)
    } else {
        None
    };
    for item in iter {
        if matches.len() >= max {
            truncated = true;
            break;
        }
        let Ok(path) = item else {
            continue;
        };
        if let Some(workspace) = workspace_root.as_ref() {
            let allowed = is_path_within_workspace(path.as_path(), workspace).unwrap_or(false);
            if !allowed {
                continue;
            }
        }
        matches.push(path.display().to_string());
    }
    Ok(json!({
        "ok": true,
        "tool": "GlobTool",
        "base_path": base.display().to_string(),
        "pattern": pattern_raw,
        "count": matches.len(),
        "truncated": truncated,
        "matches": matches
    }))
}

fn execute_grep_search(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    let args: GrepSearchArgs = parse_tool_args(raw_arguments)?;
    if args.pattern.trim().is_empty() {
        return Err("pattern is empty".to_string());
    }
    let base = resolve_user_path(args.path.as_deref().unwrap_or("."), cfg, false)?;
    if !base.is_dir() {
        return Err(format!("path is not a directory: {}", base.display()));
    }
    let regex_pattern = if args.case_sensitive {
        args.pattern.clone()
    } else {
        format!("(?i){}", args.pattern)
    };
    let regex = Regex::new(regex_pattern.as_str())
        .map_err(|err| format!("invalid regex pattern: {err}"))?;
    let file_filter = match args.glob.as_deref() {
        Some(raw) if !raw.trim().is_empty() => {
            Some(Pattern::new(raw).map_err(|err| format!("invalid glob filter: {err}"))?)
        }
        _ => None,
    };
    let max = normalized_limit(args.limit, cfg.max_search_results, DEFAULT_TOOL_LIMIT);
    let mut matches = Vec::<Value>::new();
    let mut truncated = false;
    let mut queue = VecDeque::from([base.clone()]);
    while let Some(dir) = queue.pop_front() {
        let children = read_dir_sorted(dir.as_path())?;
        for child in children {
            if matches.len() >= max {
                truncated = true;
                break;
            }
            let path = child.path();
            let file_type = match child.file_type() {
                Ok(value) => value,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                queue.push_back(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            if let Some(filter) = file_filter.as_ref() {
                let relative = path.strip_prefix(&base).unwrap_or(path.as_path());
                if !filter.matches_path(relative) {
                    continue;
                }
            }
            let (bytes, _) = match read_file_with_limit(path.as_path(), cfg.max_read_bytes) {
                Ok(content) => content,
                Err(_) => continue,
            };
            if bytes.contains(&0) {
                continue;
            }
            let content = String::from_utf8_lossy(bytes.as_slice()).to_string();
            for (line_idx, line) in content.lines().enumerate() {
                if matches.len() >= max {
                    break;
                }
                if let Some(found) = regex.find(line) {
                    matches.push(json!({
                        "path": path.display().to_string(),
                        "line": line_idx + 1,
                        "column": found.start() + 1,
                        "text": trim_line_preview(line)
                    }));
                }
            }
        }
        if matches.len() >= max {
            break;
        }
    }
    Ok(json!({
        "ok": true,
        "tool": "GrepTool",
        "base_path": base.display().to_string(),
        "pattern": args.pattern,
        "count": matches.len(),
        "truncated": truncated,
        "matches": matches
    }))
}

fn execute_web_search(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    if !cfg.web_search_enabled {
        return Err(
            "web search is disabled (ai.tools.builtin.web-search-enabled=false)".to_string(),
        );
    }
    let args: WebSearchArgs = parse_tool_args(raw_arguments)?;
    let query = args.query.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let max = normalized_limit(
        args.max_results,
        cfg.web_search_max_results,
        DEFAULT_WEB_RESULT_LIMIT,
    );
    let client = build_web_search_client(cfg.web_search_timeout_seconds)?;
    let mut errors = Vec::<String>::new();
    let mut source = "duckduckgo_instant";
    let mut results = match search_duckduckgo_instant(&client, query, max) {
        Ok(items) => items,
        Err(err) => {
            errors.push(err);
            Vec::new()
        }
    };
    if results.is_empty() {
        source = "duckduckgo_html";
        results = match search_duckduckgo_html(&client, query, max) {
            Ok(items) => items,
            Err(err) => {
                errors.push(err);
                Vec::new()
            }
        };
    }
    if results.is_empty() && !errors.is_empty() {
        return Err(format!("web search failed: {}", errors.join("; ")));
    }
    Ok(json!({
        "ok": true,
        "tool": "WebSearch",
        "query": query,
        "source": source,
        "count": results.len(),
        "truncated": results.len() >= max,
        "results": results
    }))
}

fn build_web_search_client(timeout_seconds: u64) -> Result<Client, String> {
    Client::builder()
        .connect_timeout(Duration::from_secs(timeout_seconds))
        .timeout(Duration::from_secs(timeout_seconds))
        .build()
        .map_err(|err| format!("failed to create http client: {err}"))
}

fn search_duckduckgo_instant(
    client: &Client,
    query: &str,
    max: usize,
) -> Result<Vec<Value>, String> {
    let url = reqwest::Url::parse_with_params(
        "https://api.duckduckgo.com/",
        &[
            ("q", query),
            ("format", "json"),
            ("no_redirect", "1"),
            ("no_html", "1"),
            ("skip_disambig", "0"),
        ],
    )
    .map_err(|err| format!("failed to build search url: {err}"))?;
    let resp = client
        .get(url)
        .header(USER_AGENT, WEB_SEARCH_USER_AGENT)
        .header(ACCEPT, "application/json")
        .send()
        .and_then(|res| res.error_for_status())
        .map_err(|err| format!("instant search request failed: {err}"))?;
    let body = resp
        .json::<Value>()
        .map_err(|err| format!("failed to parse instant search response: {err}"))?;
    let mut results = Vec::<Value>::new();
    append_duckduckgo_results(&body, &mut results, max);
    Ok(results)
}

fn search_duckduckgo_html(client: &Client, query: &str, max: usize) -> Result<Vec<Value>, String> {
    let url = reqwest::Url::parse_with_params("https://html.duckduckgo.com/html/", &[("q", query)])
        .map_err(|err| format!("failed to build html search url: {err}"))?;
    let html = client
        .get(url)
        .header(USER_AGENT, WEB_SEARCH_USER_AGENT)
        .header(ACCEPT, "text/html")
        .send()
        .and_then(|res| res.error_for_status())
        .map_err(|err| format!("html search request failed: {err}"))?
        .text()
        .map_err(|err| format!("failed to parse html search response: {err}"))?;
    parse_duckduckgo_html_results(html.as_str(), max)
}

fn parse_duckduckgo_html_results(html: &str, max: usize) -> Result<Vec<Value>, String> {
    let document = Html::parse_document(html);
    let result_selector = Selector::parse(".result")
        .map_err(|err| format!("invalid html selector .result: {err}"))?;
    let title_selector = Selector::parse("a.result__a")
        .map_err(|err| format!("invalid html selector a.result__a: {err}"))?;
    let snippet_selector = Selector::parse(".result__snippet")
        .map_err(|err| format!("invalid html selector .result__snippet: {err}"))?;
    let mut results = Vec::<Value>::new();
    for row in document.select(&result_selector) {
        if results.len() >= max {
            break;
        }
        let Some(anchor) = row.select(&title_selector).next() else {
            continue;
        };
        let href = anchor.value().attr("href").unwrap_or_default();
        let title = html_normalize_text(anchor.text().collect::<Vec<_>>().join(" ").as_str());
        if title.is_empty() {
            continue;
        }
        let url = normalize_search_result_url(href);
        let snippet = row
            .select(&snippet_selector)
            .next()
            .map(|node| html_normalize_text(node.text().collect::<Vec<_>>().join(" ").as_str()))
            .unwrap_or_default();
        results.push(json!({
            "title": trim_text_len(title.as_str(), MAX_WEB_SNIPPET_CHARS),
            "url": url,
            "snippet": trim_text_len(snippet.as_str(), MAX_WEB_SNIPPET_CHARS),
        }));
    }
    if results.is_empty() {
        for anchor in document.select(&title_selector) {
            if results.len() >= max {
                break;
            }
            let href = anchor.value().attr("href").unwrap_or_default();
            let title = html_normalize_text(anchor.text().collect::<Vec<_>>().join(" ").as_str());
            if title.is_empty() {
                continue;
            }
            results.push(json!({
                "title": trim_text_len(title.as_str(), MAX_WEB_SNIPPET_CHARS),
                "url": normalize_search_result_url(href),
                "snippet": ""
            }));
        }
    }
    Ok(results)
}

fn html_normalize_text(raw: &str) -> String {
    let no_tags = Regex::new(r"(?is)<[^>]+>")
        .map(|re| re.replace_all(raw, " ").to_string())
        .unwrap_or_else(|_| raw.to_string());
    let decoded = no_tags
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    Regex::new(r"\s+")
        .map(|re| re.replace_all(decoded.trim(), " ").to_string())
        .unwrap_or(decoded.trim().to_string())
}

fn normalize_search_result_url(raw_href: &str) -> String {
    let href = raw_href.trim();
    if href.is_empty() {
        return String::new();
    }
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }
    if href.starts_with("//") {
        return format!("https:{href}");
    }
    if href.starts_with('/')
        && let Ok(url) = reqwest::Url::parse(format!("https://duckduckgo.com{href}").as_str())
    {
        if let Some(uddg) = url
            .query_pairs()
            .find(|(key, _)| key == "uddg")
            .map(|(_, value)| value.to_string())
            && !uddg.trim().is_empty()
        {
            return uddg;
        }
        return url.to_string();
    }
    href.to_string()
}

fn trim_text_len(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    chars[..max_chars].iter().collect::<String>() + "..."
}

fn append_duckduckgo_results(root: &Value, output: &mut Vec<Value>, max: usize) {
    if output.len() >= max {
        return;
    }
    let abstract_text = root
        .get("AbstractText")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    let abstract_url = root
        .get("AbstractURL")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    let heading = root
        .get("Heading")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if !abstract_text.is_empty() {
        output.push(json!({
            "title": if heading.is_empty() { "Abstract".to_string() } else { heading },
            "url": abstract_url,
            "snippet": abstract_text
        }));
    }
    if output.len() >= max {
        return;
    }
    if let Some(related) = root.get("RelatedTopics").and_then(|v| v.as_array()) {
        append_related_topics(related, output, max);
    }
}

fn append_related_topics(related: &[Value], output: &mut Vec<Value>, max: usize) {
    for item in related {
        if output.len() >= max {
            break;
        }
        if let Some(nested) = item.get("Topics").and_then(|v| v.as_array()) {
            append_related_topics(nested, output, max);
            continue;
        }
        let text = item
            .get("Text")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim();
        let url = item
            .get("FirstURL")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim();
        if text.is_empty() {
            continue;
        }
        output.push(json!({
            "title": text.split(" - ").next().unwrap_or(text).to_string(),
            "url": url,
            "snippet": text
        }));
    }
}

fn execute_think(raw_arguments: &str) -> Result<Value, String> {
    let args: ThinkArgs = parse_tool_args(raw_arguments)?;
    let thought = args.thought.trim();
    if thought.is_empty() {
        return Err("thought is empty".to_string());
    }
    Ok(json!({
        "ok": true,
        "tool": "Think",
        "thought": thought
    }))
}

fn execute_task(raw_arguments: &str) -> Result<Value, String> {
    let args: TaskArgs = parse_tool_args(raw_arguments)?;
    let description = args.description.trim();
    if description.is_empty() {
        return Err("description is empty".to_string());
    }
    let task_id =
        normalize_task_id(args.task_id.as_deref()).unwrap_or_else(|| Uuid::new_v4().to_string());
    let task_status = normalize_task_status(args.status.as_deref());
    let acceptance = args
        .acceptance_criteria
        .as_deref()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .unwrap_or_default()
        .to_string();
    let evidence = args
        .evidence
        .as_deref()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .unwrap_or_default()
        .to_string();
    Ok(json!({
        "ok": true,
        "tool": "Task",
        "task_id": task_id,
        "task_status": task_status,
        "blocked": task_status == "blocked",
        "description": description,
        "acceptance_criteria": acceptance,
        "evidence": evidence,
        "plan": format!("Sub-task accepted: {description}. Keep task_id stable across updates and move status through running -> done/failed/blocked with objective evidence.")
    }))
}

fn execute_architect(raw_arguments: &str) -> Result<Value, String> {
    let args: ArchitectArgs = parse_tool_args(raw_arguments)?;
    let problem = args.problem.trim();
    if problem.is_empty() {
        return Err("problem is empty".to_string());
    }
    Ok(json!({
        "ok": true,
        "tool": "Architect",
        "advice": format!("Architecture prompt captured: {problem}. Compare at least two options, document tradeoffs, then implement incrementally.")
    }))
}

fn execute_edit(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    if !cfg.write_tools_enabled {
        return Err(
            "write tools are disabled (ai.tools.builtin.write-tools-enabled=false)".to_string(),
        );
    }
    let args: EditArgs = parse_tool_args(raw_arguments)?;
    if !args.apply {
        return Err("Edit requires apply=true to perform write operation".to_string());
    }
    if args.old_string.is_empty() {
        return Err("old_string must not be empty".to_string());
    }
    let path = resolve_user_path(args.file_path.as_str(), cfg, true)?;
    let original = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read file {}: {err}", path.display()))?;
    let occurrences = original.matches(args.old_string.as_str()).count();
    if occurrences == 0 {
        return Err("old_string not found in target file".to_string());
    }
    let next = if args.replace_all {
        original.replace(args.old_string.as_str(), args.new_string.as_str())
    } else {
        original.replacen(args.old_string.as_str(), args.new_string.as_str(), 1)
    };
    write_text_file_atomically(path.as_path(), next.as_str())?;
    Ok(json!({
        "ok": true,
        "tool": "Edit",
        "path": path.display().to_string(),
        "replace_all": args.replace_all,
        "replaced_count": if args.replace_all { occurrences } else { 1 }
    }))
}

fn execute_replace(raw_arguments: &str, cfg: &BuiltinToolsConfig) -> Result<Value, String> {
    if !cfg.write_tools_enabled {
        return Err(
            "write tools are disabled (ai.tools.builtin.write-tools-enabled=false)".to_string(),
        );
    }
    let args: ReplaceArgs = parse_tool_args(raw_arguments)?;
    if !args.apply {
        return Err("Replace requires apply=true to perform write operation".to_string());
    }
    let path = resolve_user_path(args.file_path.as_str(), cfg, true)?;
    write_text_file_atomically(path.as_path(), args.content.as_str())?;
    Ok(json!({
        "ok": true,
        "tool": "Replace",
        "path": path.display().to_string(),
        "bytes_written": args.content.len()
    }))
}

fn execute_notebook_edit_cell(
    raw_arguments: &str,
    cfg: &BuiltinToolsConfig,
) -> Result<Value, String> {
    if !cfg.write_tools_enabled {
        return Err(
            "write tools are disabled (ai.tools.builtin.write-tools-enabled=false)".to_string(),
        );
    }
    let args: NotebookEditCellArgs = parse_tool_args(raw_arguments)?;
    if !args.apply {
        return Err("NotebookEditCell requires apply=true to perform write operation".to_string());
    }
    let path = resolve_user_path(args.file_path.as_str(), cfg, true)?;
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read notebook {}: {err}", path.display()))?;
    let mut root: Value = serde_json::from_str(raw.as_str())
        .map_err(|err| format!("failed to parse notebook json: {err}"))?;
    let cells = root
        .get_mut("cells")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| "notebook missing 'cells' array".to_string())?;
    if args.cell_index >= cells.len() {
        return Err(format!(
            "cell_index {} out of range (cells={})",
            args.cell_index,
            cells.len()
        ));
    }
    let lines = args
        .content
        .lines()
        .map(|line| format!("{line}\n"))
        .map(Value::String)
        .collect::<Vec<_>>();
    let cell = cells
        .get_mut(args.cell_index)
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| "target cell is not a valid object".to_string())?;
    cell.insert("source".to_string(), Value::Array(lines));
    let rendered = serde_json::to_string_pretty(&root)
        .map_err(|err| format!("failed to serialize notebook json: {err}"))?;
    write_text_file_atomically(path.as_path(), rendered.as_str())?;
    Ok(json!({
        "ok": true,
        "tool": "NotebookEditCell",
        "path": path.display().to_string(),
        "cell_index": args.cell_index
    }))
}

fn parse_tool_args<T>(raw_arguments: &str) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let value = parse_json_object_arguments(raw_arguments)?;
    serde_json::from_value::<T>(value).map_err(|err| format!("invalid arguments: {err}"))
}

fn resolve_user_path(
    raw_path: &str,
    cfg: &BuiltinToolsConfig,
    for_write: bool,
) -> Result<PathBuf, String> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return Err("path is empty".to_string());
    }
    let cwd = normalize_path(
        std::env::current_dir().map_err(|err| format!("failed to get current directory: {err}"))?,
    );
    let joined = if Path::new(trimmed).is_absolute() {
        PathBuf::from(trimmed)
    } else {
        cwd.join(trimmed)
    };
    let normalized = normalize_path(joined);
    if !for_write && !normalized.exists() {
        return Err(format!("path does not exist: {}", normalized.display()));
    }
    if cfg.workspace_only {
        let workspace_root = canonical_workspace_root()?;
        let canonical_target = canonicalize_for_workspace_check(normalized.as_path())?;
        if !canonical_target.starts_with(&workspace_root) {
            return Err(format!(
                "path '{}' resolves outside workspace root '{}'",
                normalized.display(),
                workspace_root.display()
            ));
        }
    }
    Ok(normalized)
}

fn canonical_workspace_root() -> Result<PathBuf, String> {
    let cwd =
        std::env::current_dir().map_err(|err| format!("failed to get current directory: {err}"))?;
    fs::canonicalize(cwd.as_path())
        .map_err(|err| format!("failed to resolve workspace root {}: {err}", cwd.display()))
}

fn is_path_within_workspace(path: &Path, workspace_root: &Path) -> Result<bool, String> {
    let canonical = canonicalize_for_workspace_check(path)?;
    Ok(canonical.starts_with(workspace_root))
}

fn canonicalize_for_workspace_check(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return fs::canonicalize(path)
            .map_err(|err| format!("failed to resolve path {}: {err}", path.display()));
    }
    let mut missing_parts = Vec::<PathBuf>::new();
    let mut cursor = path.to_path_buf();
    while !cursor.exists() {
        let Some(name) = cursor.file_name() else {
            break;
        };
        missing_parts.push(PathBuf::from(name));
        let Some(parent) = cursor.parent() else {
            break;
        };
        cursor = parent.to_path_buf();
    }
    if !cursor.exists() {
        return Err(format!(
            "failed to resolve existing parent for {}",
            path.display()
        ));
    }
    let mut canonical = fs::canonicalize(cursor.as_path())
        .map_err(|err| format!("failed to resolve path {}: {err}", path.display()))?;
    for part in missing_parts.iter().rev() {
        canonical.push(part);
    }
    Ok(normalize_path(canonical))
}

fn read_file_with_limit(path: &Path, max_bytes: usize) -> Result<(Vec<u8>, bool), String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to read file {}: {err}", path.display()))?;
    let max = max_bytes.max(1);
    let mut reader = file.by_ref().take((max as u64).saturating_add(1));
    let mut buffer = Vec::with_capacity(max.min(8192));
    reader
        .read_to_end(&mut buffer)
        .map_err(|err| format!("failed to read file {}: {err}", path.display()))?;
    let truncated = buffer.len() > max;
    if truncated {
        buffer.truncate(max);
    }
    Ok((buffer, truncated))
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn read_dir_sorted(path: &Path) -> Result<Vec<std::fs::DirEntry>, String> {
    let mut entries = fs::read_dir(path)
        .map_err(|err| format!("failed to read directory {}: {err}", path.display()))?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.path());
    Ok(entries)
}

fn file_kind_from_file_type(file_type: std::fs::FileType) -> &'static str {
    if file_type.is_dir() {
        return "dir";
    }
    if file_type.is_file() {
        return "file";
    }
    if file_type.is_symlink() {
        return "symlink";
    }
    "other"
}

fn normalized_limit(requested: Option<usize>, config_limit: usize, fallback: usize) -> usize {
    let base = requested.unwrap_or(fallback).max(1);
    base.min(config_limit.max(1))
}

fn normalize_task_id(raw: Option<&str>) -> Option<String> {
    let value = raw?;
    let mut normalized = String::with_capacity(value.len().min(80));
    for ch in value.trim().chars() {
        if normalized.chars().count() >= 80 {
            break;
        }
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            normalized.push(ch);
        } else if !normalized.ends_with('-') {
            normalized.push('-');
        }
    }
    let trimmed = normalized.trim_matches('-').trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn normalize_task_status(raw: Option<&str>) -> &'static str {
    match raw
        .unwrap_or("running")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "done" | "success" | "completed" => "done",
        "failed" | "error" => "failed",
        "blocked" => "blocked",
        _ => "running",
    }
}

fn trim_line_preview(line: &str) -> String {
    let chars = line.chars().collect::<Vec<_>>();
    if chars.len() <= MAX_LINE_PREVIEW_CHARS {
        return line.to_string();
    }
    chars[..MAX_LINE_PREVIEW_CHARS].iter().collect::<String>() + "..."
}

fn write_text_file_atomically(path: &Path, content: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("failed to resolve parent directory for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|err| {
        format!(
            "failed to create parent directory {}: {err}",
            parent.display()
        )
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("file");
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let temp_path = parent.join(format!(".{}.{}.{}.tmp", file_name, std::process::id(), now));
    fs::write(temp_path.as_path(), content.as_bytes()).map_err(|err| {
        format!(
            "failed to write temporary file {}: {err}",
            temp_path.display()
        )
    })?;
    if let Err(rename_err) = fs::rename(temp_path.as_path(), path) {
        if let Err(copy_err) = fs::copy(temp_path.as_path(), path) {
            let _ = fs::remove_file(temp_path.as_path());
            return Err(format!(
                "failed to replace file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            ));
        }
        let _ = fs::remove_file(temp_path.as_path());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::json;

    use super::{
        execute_grep_search, execute_list_files, execute_tool, external_tool_definitions,
        is_silent_tool, parse_duckduckgo_html_results, read_file_with_limit, resolve_user_path,
    };
    use crate::config::BuiltinToolsConfig;

    fn create_workspace_temp_dir(case_name: &str) -> PathBuf {
        let dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("builtin-tools-tests")
            .join(format!("{case_name}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.as_path()).expect("create workspace temp dir");
        dir
    }

    #[cfg(unix)]
    #[test]
    fn resolve_user_path_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let cfg = BuiltinToolsConfig {
            workspace_only: true,
            ..BuiltinToolsConfig::default()
        };
        let workspace_dir = create_workspace_temp_dir("path-escape");
        let outside_dir =
            std::env::temp_dir().join(format!("machineclaw-outside-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(outside_dir.as_path()).expect("create outside dir");
        fs::write(outside_dir.join("secret.txt"), "secret").expect("write outside file");
        symlink(outside_dir.as_path(), workspace_dir.join("escape")).expect("create symlink");
        let relative = workspace_dir
            .join("escape")
            .join("secret.txt")
            .strip_prefix(std::env::current_dir().expect("cwd"))
            .expect("strip workspace prefix")
            .to_string_lossy()
            .to_string();
        let result = resolve_user_path(relative.as_str(), &cfg, false);
        fs::remove_dir_all(workspace_dir).ok();
        fs::remove_dir_all(outside_dir).ok();
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn list_recursive_does_not_follow_symlink_directories() {
        use std::os::unix::fs::symlink;

        let cfg = BuiltinToolsConfig {
            workspace_only: true,
            ..BuiltinToolsConfig::default()
        };
        let workspace_dir = create_workspace_temp_dir("ls-symlink");
        let outside_dir =
            std::env::temp_dir().join(format!("machineclaw-ls-outside-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(outside_dir.as_path()).expect("create outside dir");
        fs::write(outside_dir.join("outside.txt"), "outside").expect("write outside file");
        fs::write(workspace_dir.join("inside.txt"), "inside").expect("write inside file");
        symlink(outside_dir.as_path(), workspace_dir.join("link")).expect("create symlink");
        let relative = workspace_dir
            .strip_prefix(std::env::current_dir().expect("cwd"))
            .expect("strip workspace prefix")
            .to_string_lossy()
            .to_string();
        let payload = execute_list_files(
            json!({
                "path": relative,
                "recursive": true,
                "include_hidden": true,
                "limit": 100
            })
            .to_string()
            .as_str(),
            &cfg,
        )
        .expect("execute ls");
        let entries = payload
            .get("entries")
            .and_then(|v| v.as_array())
            .expect("entries array");
        let mut contains_link_entry = false;
        let mut contains_outside_through_link = false;
        for item in entries {
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let kind = item
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if path.ends_with("/link") {
                contains_link_entry = kind == "symlink";
            }
            if path.contains("/link/") {
                contains_outside_through_link = true;
            }
        }
        fs::remove_dir_all(workspace_dir).ok();
        fs::remove_dir_all(outside_dir).ok();
        assert!(contains_link_entry);
        assert!(!contains_outside_through_link);
    }

    #[cfg(unix)]
    #[test]
    fn grep_does_not_follow_symlink_directories() {
        use std::os::unix::fs::symlink;

        let cfg = BuiltinToolsConfig {
            workspace_only: true,
            ..BuiltinToolsConfig::default()
        };
        let workspace_dir = create_workspace_temp_dir("grep-symlink");
        let outside_dir =
            std::env::temp_dir().join(format!("machineclaw-grep-outside-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(outside_dir.as_path()).expect("create outside dir");
        fs::write(outside_dir.join("outside.txt"), "needle outside").expect("write outside file");
        fs::write(workspace_dir.join("inside.txt"), "needle inside").expect("write inside file");
        symlink(outside_dir.as_path(), workspace_dir.join("link")).expect("create symlink");
        let relative = workspace_dir
            .strip_prefix(std::env::current_dir().expect("cwd"))
            .expect("strip workspace prefix")
            .to_string_lossy()
            .to_string();
        let payload = execute_grep_search(
            json!({
                "pattern": "needle",
                "path": relative,
                "case_sensitive": false,
                "limit": 20
            })
            .to_string()
            .as_str(),
            &cfg,
        )
        .expect("execute grep");
        let matches = payload
            .get("matches")
            .and_then(|v| v.as_array())
            .expect("matches array");
        let mut seen_inside = false;
        let mut seen_outside = false;
        for item in matches {
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if path.ends_with("/inside.txt") {
                seen_inside = true;
            }
            if path.contains("/link/") {
                seen_outside = true;
            }
        }
        fs::remove_dir_all(workspace_dir).ok();
        fs::remove_dir_all(outside_dir).ok();
        assert!(seen_inside);
        assert!(!seen_outside);
    }

    #[test]
    fn read_file_with_limit_truncates_content() {
        let workspace_dir = create_workspace_temp_dir("read-limit");
        let file = workspace_dir.join("data.txt");
        fs::write(file.as_path(), "abcdefg").expect("write test file");
        let (bytes, truncated) = read_file_with_limit(file.as_path(), 4).expect("read with limit");
        fs::remove_dir_all(workspace_dir).ok();
        assert!(truncated);
        assert_eq!(bytes, b"abcd");
    }

    #[test]
    fn parse_duckduckgo_html_results_extracts_title_url_and_snippet() {
        let html = r#"
<html><body>
  <div class="result">
    <a class="result__a" href="/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fdoc">Example Doc</a>
    <a class="result__snippet">This is snippet.</a>
  </div>
</body></html>
"#;
        let results = parse_duckduckgo_html_results(html, 3).expect("parse html results");
        assert_eq!(results.len(), 1);
        let item = &results[0];
        assert_eq!(
            item.get("url").and_then(|v| v.as_str()),
            Some("https://example.com/doc")
        );
        assert_eq!(
            item.get("title").and_then(|v| v.as_str()),
            Some("Example Doc")
        );
        assert_eq!(
            item.get("snippet").and_then(|v| v.as_str()),
            Some("This is snippet.")
        );
    }

    #[test]
    fn external_tool_catalog_excludes_think_tool() {
        let cfg = BuiltinToolsConfig::default();
        let names = external_tool_definitions(&cfg)
            .into_iter()
            .map(|item| item.name)
            .collect::<Vec<_>>();
        assert!(!names.iter().any(|item| item.eq_ignore_ascii_case("think")));
    }

    #[test]
    fn think_tool_is_silent_for_compatibility_calls() {
        assert!(is_silent_tool("Think"));
        assert!(is_silent_tool("think"));
        assert!(!is_silent_tool("Task"));
    }

    #[test]
    fn task_tool_accepts_stable_id_and_running_status() {
        let payload = execute_tool(
            "Task",
            json!({
                "description": "sync docs",
                "task_id": " docs.sync#001 ",
                "status": "running",
                "acceptance_criteria": "all tests pass",
                "evidence": "cargo test"
            })
            .to_string()
            .as_str(),
            &BuiltinToolsConfig::default(),
        )
        .expect("task tool should execute");
        let parsed = serde_json::from_str::<serde_json::Value>(payload.as_str())
            .expect("parse task payload");
        assert_eq!(
            parsed.get("task_id").and_then(|v| v.as_str()),
            Some("docs-sync-001")
        );
        assert_eq!(
            parsed.get("task_status").and_then(|v| v.as_str()),
            Some("running")
        );
        assert_eq!(parsed.get("ok").and_then(|v| v.as_bool()), Some(true));
    }
}
