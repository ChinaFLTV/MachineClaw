use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::{error::AppError, mcp::parse_json_object_arguments};

const TASKS_DIR_NAME: &str = "tasks";
const TASK_FILE_PREFIX: &str = "tasks-";
const TASK_FILE_SUFFIX: &str = ".json";
const MAX_RESULT_PREVIEW_CHARS: usize = 600;
const TASK_ID_MAX_CHARS: usize = 80;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Running,
    #[default]
    Failed,
    Done,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedTask {
    #[serde(default)]
    pub task_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub session_name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub status: TaskStatus,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub result_preview: String,
    #[serde(default)]
    pub created_at_epoch_ms: u128,
    #[serde(default)]
    pub updated_at_epoch_ms: u128,
}

#[derive(Debug, Clone, Default)]
pub struct TaskSessionSummary {
    pub total: usize,
    pub running: usize,
    pub done: usize,
    pub failed: usize,
    pub blocked: usize,
    pub latest: Vec<PersistedTask>,
}

pub struct PersistTaskRequest<'a> {
    pub session_file: &'a Path,
    pub session_id: &'a str,
    pub session_name: &'a str,
    pub task_id: Option<&'a str>,
    pub description: &'a str,
    pub status: TaskStatus,
    pub source: &'a str,
    pub tool_call_id: &'a str,
    pub raw_payload: &'a str,
}

#[derive(Debug, Clone)]
pub struct TaskCallArgs {
    pub description: String,
    pub task_id: Option<String>,
}

pub fn extract_task_call_args(raw_arguments: &str) -> Option<TaskCallArgs> {
    let value = parse_json_object_arguments(raw_arguments).ok()?;
    let description = value.get("description")?.as_str()?.trim();
    if description.is_empty() {
        return None;
    }
    let task_id = value
        .get("task_id")
        .or_else(|| value.get("id"))
        .and_then(|item| item.as_str())
        .and_then(|item| sanitize_task_id(item).filter(|clean| !clean.is_empty()));
    Some(TaskCallArgs {
        description: description.to_string(),
        task_id,
    })
}

pub fn sanitize_task_id(raw: &str) -> Option<String> {
    let mut normalized = String::with_capacity(raw.len().min(TASK_ID_MAX_CHARS));
    for ch in raw.trim().chars() {
        if normalized.chars().count() >= TASK_ID_MAX_CHARS {
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

pub fn status_from_str(raw: &str) -> Option<TaskStatus> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "running" | "in_progress" | "in-progress" | "todo" | "pending" => Some(TaskStatus::Running),
        "done" | "success" | "completed" => Some(TaskStatus::Done),
        "failed" | "error" => Some(TaskStatus::Failed),
        "blocked" => Some(TaskStatus::Blocked),
        _ => None,
    }
}

pub fn infer_task_status_from_payload(raw_payload: &str) -> TaskStatus {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw_payload) else {
        return TaskStatus::Failed;
    };
    if let Some(status) = value
        .get("task_status")
        .or_else(|| value.get("status"))
        .and_then(|item| item.as_str())
        .and_then(status_from_str)
    {
        return status;
    }
    if value
        .get("blocked")
        .and_then(|item| item.as_bool())
        .unwrap_or(false)
    {
        return TaskStatus::Blocked;
    }
    if value
        .get("ok")
        .and_then(|item| item.as_bool())
        .unwrap_or(false)
    {
        return TaskStatus::Done;
    }
    TaskStatus::Failed
}

pub fn persist_task_record(request: PersistTaskRequest<'_>) -> Result<PersistedTask, AppError> {
    let task_dir = resolve_tasks_dir(request.session_file);
    fs::create_dir_all(task_dir.as_path()).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create task directory {}: {err}",
            task_dir.display()
        ))
    })?;
    let now = now_epoch_ms();
    let task_id = request
        .task_id
        .and_then(sanitize_task_id)
        .or_else(|| task_id_from_payload(request.raw_payload))
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let file_path = task_dir.join(format!("{TASK_FILE_PREFIX}{task_id}{TASK_FILE_SUFFIX}"));
    let existing = read_task_record(file_path.as_path());
    let record = PersistedTask {
        task_id: task_id.to_string(),
        session_id: request.session_id.trim().to_string(),
        session_name: request.session_name.trim().to_string(),
        description: request.description.trim().to_string(),
        status: request.status,
        source: request.source.trim().to_string(),
        tool_call_id: request.tool_call_id.trim().to_string(),
        result_preview: trim_chars(request.raw_payload.trim(), MAX_RESULT_PREVIEW_CHARS),
        created_at_epoch_ms: existing
            .as_ref()
            .map(|item| item.created_at_epoch_ms)
            .unwrap_or(now),
        updated_at_epoch_ms: now,
    };
    let encoded = serde_json::to_string_pretty(&record)
        .map_err(|err| AppError::Runtime(format!("failed to encode task record: {err}")))?;
    write_string_atomically(file_path.as_path(), encoded.as_str())?;
    Ok(record)
}

pub fn list_tasks_for_session(
    session_file: &Path,
    session_id: &str,
    limit: usize,
) -> Result<Vec<PersistedTask>, AppError> {
    let task_dir = resolve_tasks_dir(session_file);
    if !task_dir.exists() {
        return Ok(Vec::new());
    }
    let mut items = Vec::<PersistedTask>::new();
    for entry in fs::read_dir(task_dir.as_path()).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read task directory {}: {err}",
            task_dir.display()
        ))
    })? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !is_task_file(path.as_path()) {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path.as_path()) else {
            continue;
        };
        let Ok(item) = serde_json::from_str::<PersistedTask>(&raw) else {
            continue;
        };
        if item.session_id.trim() != session_id.trim() {
            continue;
        }
        items.push(item);
    }
    items.sort_by(|a, b| {
        b.updated_at_epoch_ms
            .cmp(&a.updated_at_epoch_ms)
            .then_with(|| b.created_at_epoch_ms.cmp(&a.created_at_epoch_ms))
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
    let cap = limit.max(1);
    if items.len() > cap {
        items.truncate(cap);
    }
    Ok(items)
}

pub fn summarize_tasks_for_session(
    session_file: &Path,
    session_id: &str,
    latest_limit: usize,
) -> Result<TaskSessionSummary, AppError> {
    let task_dir = resolve_tasks_dir(session_file);
    if !task_dir.exists() {
        return Ok(TaskSessionSummary::default());
    }
    let all = list_tasks_for_session(session_file, session_id, usize::MAX)?;
    let mut summary = TaskSessionSummary {
        total: all.len(),
        ..TaskSessionSummary::default()
    };
    for item in &all {
        match item.status {
            TaskStatus::Running => summary.running += 1,
            TaskStatus::Done => summary.done += 1,
            TaskStatus::Failed => summary.failed += 1,
            TaskStatus::Blocked => summary.blocked += 1,
        }
    }
    summary.latest = all
        .into_iter()
        .take(latest_limit.max(1))
        .collect::<Vec<_>>();
    Ok(summary)
}

pub fn resolve_tasks_dir(session_file: &Path) -> PathBuf {
    let parent = session_file.parent().unwrap_or_else(|| Path::new("."));
    if parent
        .file_name()
        .and_then(|item| item.to_str())
        .is_some_and(|name| name == "sessions")
    {
        let base = parent.parent().unwrap_or(parent);
        return base.join(TASKS_DIR_NAME);
    }
    parent.join(TASKS_DIR_NAME)
}

pub fn status_to_label(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Running => "running",
        TaskStatus::Done => "done",
        TaskStatus::Failed => "failed",
        TaskStatus::Blocked => "blocked",
    }
}

pub fn augment_tool_payload_with_task(payload: &str, record: &PersistedTask) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return payload.to_string();
    };
    let Some(object) = value.as_object_mut() else {
        return payload.to_string();
    };
    object.insert("task_id".to_string(), json!(record.task_id));
    object.insert(
        "task_status".to_string(),
        json!(status_to_label(record.status)),
    );
    object.insert("task_session_id".to_string(), json!(record.session_id));
    value.to_string()
}

fn is_task_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|item| item.to_str()) else {
        return false;
    };
    name.starts_with(TASK_FILE_PREFIX) && name.ends_with(TASK_FILE_SUFFIX)
}

fn task_id_from_payload(raw_payload: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(raw_payload).ok()?;
    value
        .get("task_id")
        .or_else(|| value.get("id"))
        .and_then(|item| item.as_str())
        .and_then(sanitize_task_id)
}

fn read_task_record(path: &Path) -> Option<PersistedTask> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<PersistedTask>(raw.as_str()).ok()
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn trim_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "..."
}

fn write_string_atomically(path: &Path, content: &str) -> Result<(), AppError> {
    let parent = path.parent().ok_or_else(|| {
        AppError::Runtime(format!(
            "failed to resolve task file parent for {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create task directory {}: {err}",
            parent.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("task");
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        now_epoch_ms()
    ));
    fs::write(temp_path.as_path(), content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write temporary task file {}: {err}",
            temp_path.display()
        ))
    })?;
    if let Err(rename_err) = fs::rename(temp_path.as_path(), path) {
        if let Err(copy_err) = fs::copy(temp_path.as_path(), path) {
            let _ = fs::remove_file(temp_path.as_path());
            return Err(AppError::Runtime(format!(
                "failed to replace task file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            )));
        }
        let _ = fs::remove_file(temp_path.as_path());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        PersistTaskRequest, TaskStatus, extract_task_call_args, infer_task_status_from_payload,
        list_tasks_for_session, persist_task_record, resolve_tasks_dir, sanitize_task_id,
        status_from_str, status_to_label, summarize_tasks_for_session,
    };

    #[test]
    fn extract_task_call_args_reads_optional_task_id() {
        let args = extract_task_call_args(r#"{"description":"build","task_id":" job.001 "}"#)
            .expect("task args should parse");
        assert_eq!(args.description, "build");
        assert_eq!(args.task_id.as_deref(), Some("job-001"));
    }

    #[test]
    fn sanitize_task_id_filters_dangerous_chars() {
        assert_eq!(
            sanitize_task_id("../job alpha$beta").as_deref(),
            Some("job-alpha-beta")
        );
        assert!(sanitize_task_id("   ").is_none());
    }

    #[test]
    fn status_from_str_supports_running_aliases() {
        assert_eq!(status_from_str("running"), Some(TaskStatus::Running));
        assert_eq!(status_from_str("in_progress"), Some(TaskStatus::Running));
        assert_eq!(status_from_str("done"), Some(TaskStatus::Done));
        assert_eq!(status_from_str("failed"), Some(TaskStatus::Failed));
        assert_eq!(status_from_str("blocked"), Some(TaskStatus::Blocked));
        assert_eq!(status_to_label(TaskStatus::Running), "running");
    }

    #[test]
    fn infer_task_status_from_payload_uses_ok_and_blocked() {
        assert_eq!(
            infer_task_status_from_payload(r#"{"task_status":"running","ok":true}"#),
            TaskStatus::Running
        );
        assert_eq!(
            infer_task_status_from_payload(r#"{"ok":true,"blocked":false}"#),
            TaskStatus::Done
        );
        assert_eq!(
            infer_task_status_from_payload(r#"{"ok":false,"blocked":true}"#),
            TaskStatus::Blocked
        );
        assert_eq!(
            infer_task_status_from_payload(r#"{"ok":false}"#),
            TaskStatus::Failed
        );
    }

    #[test]
    fn persist_and_summarize_tasks_by_session() {
        let base = std::env::current_dir()
            .expect("cwd")
            .join("target")
            .join("task-store-tests")
            .join(uuid::Uuid::new_v4().to_string());
        let session_file = base
            .join(".machineclaw")
            .join("sessions")
            .join("session-a.json");
        fs::create_dir_all(
            session_file
                .parent()
                .expect("session parent should exist after join"),
        )
        .expect("create session dir");
        fs::write(session_file.as_path(), "{}").expect("write session file");

        let _ = persist_task_record(PersistTaskRequest {
            session_file: session_file.as_path(),
            session_id: "session-a",
            session_name: "A",
            task_id: Some("task-alpha"),
            description: "task one",
            status: TaskStatus::Done,
            source: "builtin_task",
            tool_call_id: "call-1",
            raw_payload: r#"{"ok":true}"#,
        })
        .expect("persist task one");
        let _ = persist_task_record(PersistTaskRequest {
            session_file: session_file.as_path(),
            session_id: "session-a",
            session_name: "A",
            task_id: Some("task-beta"),
            description: "task two",
            status: TaskStatus::Blocked,
            source: "builtin_task",
            tool_call_id: "call-2",
            raw_payload: r#"{"ok":false,"blocked":true}"#,
        })
        .expect("persist task two");

        let listed = list_tasks_for_session(session_file.as_path(), "session-a", 10)
            .expect("list tasks for session");
        assert_eq!(listed.len(), 2);
        let summary = summarize_tasks_for_session(session_file.as_path(), "session-a", 3)
            .expect("summarize tasks");
        assert_eq!(summary.total, 2);
        assert_eq!(summary.done, 1);
        assert_eq!(summary.blocked, 1);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.running, 0);

        let _ = persist_task_record(PersistTaskRequest {
            session_file: session_file.as_path(),
            session_id: "session-a",
            session_name: "A",
            task_id: Some("task-alpha"),
            description: "task one updated",
            status: TaskStatus::Running,
            source: "builtin_task",
            tool_call_id: "call-3",
            raw_payload: r#"{"task_status":"running","ok":true}"#,
        })
        .expect("update existing task");
        let listed_after_update = list_tasks_for_session(session_file.as_path(), "session-a", 10)
            .expect("list updated tasks for session");
        assert_eq!(listed_after_update.len(), 2);
        assert!(
            listed_after_update
                .iter()
                .any(|item| item.task_id == "task-alpha" && item.status == TaskStatus::Running)
        );

        let tasks_dir = resolve_tasks_dir(session_file.as_path());
        assert!(tasks_dir.exists());
        fs::remove_dir_all(base).ok();
    }
}
