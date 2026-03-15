use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ai::ChatMessage, error::AppError, i18n, mask::mask_sensitive};

const SUMMARY_MAX_CHARS: usize = 4000;
const AI_COMPRESSION_MARKER: &str = "[ai_summary_compression]";
const CHAT_PROFILE_MARKER: &str = "[chat_profile_v1]";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    #[default]
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub kind: MessageKind,
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default)]
    pub created_at_epoch_ms: u128,
    #[serde(default)]
    pub tool_meta: Option<ToolExecutionMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolExecutionMeta {
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub function_name: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub arguments: String,
    #[serde(default)]
    pub result_payload: String,
    #[serde(default)]
    pub executed_at_epoch_ms: u128,
    #[serde(default)]
    pub account: String,
    #[serde(default)]
    pub environment: String,
    #[serde(default)]
    pub os_name: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub duration_ms: u128,
    #[serde(default)]
    pub timed_out: bool,
    #[serde(default)]
    pub interrupted: bool,
    #[serde(default)]
    pub blocked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionCompass {
    #[serde(default)]
    pub created_at_epoch_ms: u128,
    #[serde(default)]
    pub last_updated_epoch_ms: u128,
    #[serde(default)]
    pub truncated_messages: usize,
    #[serde(default)]
    pub compression_rounds: usize,
    #[serde(default)]
    pub dropped_groups: usize,
    #[serde(default)]
    pub total_user_messages: usize,
    #[serde(default)]
    pub total_assistant_messages: usize,
    #[serde(default)]
    pub total_tool_messages: usize,
    #[serde(default)]
    pub last_action: String,
    #[serde(default)]
    pub last_user_topic: String,
    #[serde(default)]
    pub last_assistant_focus: String,
    #[serde(default)]
    pub last_compaction_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    #[serde(default = "default_session_id")]
    pub session_id: String,
    #[serde(default)]
    pub session_name: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub messages: Vec<SessionMessage>,
    #[serde(default)]
    pub compass: SessionCompass,
}

pub struct SessionStore {
    path: PathBuf,
    state: SessionState,
    recent_limit: usize,
    max_limit: usize,
    compression_max_history_messages: usize,
    compression_max_chars_count: usize,
    compression_keep_recent_messages: usize,
}

#[derive(Debug, Clone)]
pub struct SessionOverview {
    pub session_id: String,
    pub session_name: String,
    pub file_path: PathBuf,
    pub message_count: usize,
    pub summary_len: usize,
    pub user_count: usize,
    pub assistant_count: usize,
    pub tool_count: usize,
    pub system_count: usize,
    pub created_at_epoch_ms: u128,
    pub last_updated_epoch_ms: u128,
    pub active: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionRoleCounts {
    pub total: usize,
    pub user: usize,
    pub assistant: usize,
    pub tool: usize,
    pub system: usize,
}

#[derive(Debug, Clone)]
pub struct AiCompressionPlan {
    pub candidate_messages: usize,
    pub previous_summaries: Vec<String>,
    pub transcript: String,
}

#[derive(Debug, Clone)]
pub struct AiCompressionApplyResult {
    pub removed_messages: usize,
    pub total_messages: usize,
}

impl SessionStore {
    pub fn load_or_new(
        path: PathBuf,
        recent_limit: usize,
        max_limit: usize,
        compression_max_history_messages: usize,
        compression_max_chars_count: usize,
    ) -> Result<Self, AppError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to create session directory {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let state = load_state_with_autosave_or_new(&path)?;

        let mut store = Self {
            path,
            state,
            recent_limit,
            max_limit,
            compression_max_history_messages,
            compression_max_chars_count,
            compression_keep_recent_messages: compute_keep_recent_messages(
                compression_max_history_messages,
            ),
        };
        store.ensure_session_path_in_sessions_dir()?;
        store.repair_compass();
        store.enforce_max_limit();
        store.persist_active_session_pointer()?;
        Ok(store)
    }

    pub fn add_user_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("user", MessageKind::User, content, group_id, None);
    }

    pub fn add_assistant_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("assistant", MessageKind::Assistant, content, group_id, None);
    }

    pub fn add_thinking_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("thinking", MessageKind::Assistant, content, group_id, None);
    }

    pub fn append_or_add_thinking_chunk(&mut self, chunk: &str, group_id: Option<&str>) {
        if chunk.is_empty() {
            return;
        }
        if let Some(last) = self.state.messages.last_mut()
            && last.role == "thinking"
            && last.group_id.as_deref() == group_id
        {
            last.content.push_str(chunk);
            self.state.compass.last_updated_epoch_ms = now_epoch_ms();
            return;
        }
        self.add_thinking_message(chunk.to_string(), group_id.map(|item| item.to_string()));
    }

    pub fn add_tool_message(&mut self, content: String, group_id: Option<String>) {
        self.add_tool_message_with_meta(content, group_id, None);
    }

    pub fn add_tool_message_with_meta(
        &mut self,
        content: String,
        group_id: Option<String>,
        tool_meta: Option<ToolExecutionMeta>,
    ) {
        self.add_message("tool", MessageKind::Tool, content, group_id, tool_meta);
    }

    pub fn add_system_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("system", MessageKind::System, content, group_id, None);
    }

    pub fn build_chat_history(&self) -> Vec<ChatMessage> {
        let mut history = Vec::new();
        if !self.state.summary.is_empty() {
            history.push(ChatMessage {
                role: "system".to_string(),
                content: format!("Conversation summary: {}", self.state.summary),
            });
        }
        let compass = self.compass_snapshot();
        if !compass.is_empty() {
            history.push(ChatMessage {
                role: "system".to_string(),
                content: format!("Conversation compass: {compass}"),
            });
        }

        let mut start = self.state.messages.len().saturating_sub(self.recent_limit);
        if start < self.state.messages.len()
            && let Some(group_id) = self.state.messages[start].group_id.as_deref()
        {
            while start > 0 && self.state.messages[start - 1].group_id.as_deref() == Some(group_id)
            {
                start -= 1;
            }
        }

        for message in &self.state.messages[start..] {
            if message.role == "thinking" {
                continue;
            }
            if message.role == "tool"
                && message.tool_meta.is_none()
                && !is_persisted_tool_trace_message(&message.content)
            {
                continue;
            }
            let Some(role) = (match message.role.as_str() {
                "assistant" => Some("assistant"),
                "system" => Some("system"),
                "user" => Some("user"),
                "tool" => Some("user"),
                _ => None,
            }) else {
                continue;
            };
            let content = if message.role == "tool" {
                format!("[tool] {}", message.content)
            } else {
                message.content.clone()
            };
            history.push(ChatMessage {
                role: role.to_string(),
                content,
            });
        }

        history
    }

    pub fn persist(&self) -> Result<(), AppError> {
        let raw = serde_json::to_string_pretty(&self.state).map_err(|err| {
            AppError::Runtime(format!("failed to serialize session state: {err}"))
        })?;
        write_string_atomically(&self.path, &raw)
    }

    pub fn serialized_state_pretty(&self) -> Result<String, AppError> {
        serde_json::to_string_pretty(&self.state)
            .map_err(|err| AppError::Runtime(format!("failed to serialize session state: {err}")))
    }

    pub fn autosave_file_path(&self) -> PathBuf {
        autosave_file_path_for(&self.path)
    }

    pub fn session_file(path: &Path) -> PathBuf {
        let base_dir = path.join(".machineclaw");
        let sessions_dir = base_dir.join("sessions");
        let pointer_files = [
            base_dir.join("active_session"),
            sessions_dir.join("active_session"),
        ];
        for pointer_file in pointer_files {
            if let Ok(raw) = fs::read_to_string(&pointer_file) {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let candidate = PathBuf::from(trimmed);
                if candidate.exists() {
                    return candidate;
                }
                let relative_base = base_dir.join(trimmed);
                if relative_base.exists() {
                    return relative_base;
                }
                let relative_sessions = sessions_dir.join(trimmed);
                if relative_sessions.exists() {
                    return relative_sessions;
                }
            }
        }
        let legacy_default = base_dir.join("session.json");
        if legacy_default.exists() {
            return legacy_default;
        }
        sessions_dir.join("session.json")
    }

    pub fn session_id(&self) -> &str {
        &self.state.session_id
    }

    pub fn session_name(&self) -> &str {
        &self.state.session_name
    }

    pub fn message_count(&self) -> usize {
        self.state.messages.len()
    }

    pub fn compass(&self) -> &SessionCompass {
        &self.state.compass
    }

    pub fn recent_messages_for_display(&self, limit: usize) -> Vec<SessionMessage> {
        let mut items = Vec::<SessionMessage>::new();
        for idx in self.recent_display_state_indices(limit) {
            if let Some(item) = self.state.messages.get(idx) {
                items.push(item.clone());
            }
        }
        items
    }

    pub fn remove_recent_display_message_by_signature(
        &mut self,
        limit: usize,
        role: &str,
        content: &str,
        occurrence_from_end: usize,
    ) -> Option<SessionMessage> {
        if occurrence_from_end == 0 {
            return None;
        }
        let display_indices = self.recent_display_state_indices(limit);
        let mut matched = 0usize;
        let mut target_index = None;
        for idx in display_indices.into_iter().rev() {
            let Some(message) = self.state.messages.get(idx) else {
                continue;
            };
            if message.role == role && message.content == content {
                matched = matched.saturating_add(1);
                if matched == occurrence_from_end {
                    target_index = Some(idx);
                    break;
                }
            }
        }
        let idx = target_index?;
        let removed = self.state.messages.remove(idx);
        self.update_compass_on_remove(&removed);
        Some(removed)
    }

    fn recent_display_state_indices(&self, limit: usize) -> Vec<usize> {
        let mut indices = Vec::<usize>::new();
        let wanted = limit.max(1);
        for (idx, item) in self.state.messages.iter().enumerate().rev() {
            if !is_visible_display_message(item) {
                continue;
            }
            indices.push(idx);
            if indices.len() >= wanted {
                break;
            }
        }
        indices.reverse();
        indices
    }

    pub fn summary_len(&self) -> usize {
        self.state.summary.chars().count()
    }

    pub fn total_message_chars(&self) -> usize {
        self.state
            .messages
            .iter()
            .map(|msg| msg.content.chars().count())
            .sum()
    }

    pub fn file_path(&self) -> &Path {
        &self.path
    }

    pub fn archived_role_counts(&self) -> SessionRoleCounts {
        let mut counts = SessionRoleCounts::default();
        for message in &self.state.messages {
            counts.total += 1;
            match message.role.as_str() {
                "user" => counts.user += 1,
                "assistant" => counts.assistant += 1,
                "tool" => counts.tool += 1,
                "system" => counts.system += 1,
                _ => {}
            }
        }
        counts
    }

    pub fn effective_context_role_counts(
        &self,
        include_base_system_prompt: bool,
    ) -> SessionRoleCounts {
        let mut counts = self.archived_role_counts();
        if !self.state.summary.trim().is_empty() {
            counts.total += 1;
            counts.system += 1;
        }
        if !self.compass_snapshot().trim().is_empty() {
            counts.total += 1;
            counts.system += 1;
        }
        if include_base_system_prompt {
            counts.total += 1;
            counts.system += 1;
        }
        counts
    }

    pub fn context_pressure_warning(
        &self,
        warn_percent: u8,
        critical_percent: u8,
    ) -> Option<String> {
        if self.max_limit == 0 {
            return None;
        }
        let usage_percent = ((self.state.messages.len() * 100) / self.max_limit) as u8;
        if usage_percent < warn_percent {
            return None;
        }
        Some(i18n::chat_context_pressure_warning(
            usage_percent,
            self.state.messages.len(),
            self.max_limit,
            self.recent_limit,
            self.state.summary.chars().count(),
            usage_percent >= critical_percent,
        ))
    }

    pub fn start_new_session_with_new_file(&mut self) -> Result<(), AppError> {
        let new_session_id = Uuid::new_v4().to_string();
        self.state = SessionState {
            session_id: new_session_id.clone(),
            session_name: default_session_name(&new_session_id),
            summary: String::new(),
            messages: Vec::new(),
            compass: new_compass(),
        };
        let session_dir = session_dir_from_session_path(&self.path);
        fs::create_dir_all(&session_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create session directory {}: {err}",
                session_dir.display()
            ))
        })?;
        let filename = format!("session-{new_session_id}.json");
        self.path = session_dir.join(filename);
        self.persist()?;
        self.persist_active_session_pointer()
    }

    pub fn rename_current_session(&mut self, new_name: &str) -> Result<(), AppError> {
        let normalized = normalize_session_name(&self.state.session_id, Some(new_name));
        self.state.session_name = normalized;
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
        self.persist()
    }

    pub fn rename_session_by_id(
        &mut self,
        session_id: &str,
        new_name: &str,
    ) -> Result<SessionOverview, AppError> {
        let normalized_id = session_id.trim();
        if normalized_id.is_empty() {
            return Err(AppError::Runtime("session id cannot be empty".to_string()));
        }
        if self.state.session_id == normalized_id {
            self.rename_current_session(new_name)?;
            return Ok(build_session_overview(
                &self.state,
                self.path.clone(),
                &self.path,
            ));
        }
        let sessions = self.list_sessions()?;
        let Some(target) = sessions
            .into_iter()
            .find(|item| item.session_id == normalized_id)
        else {
            return Err(AppError::Runtime(format!(
                "session not found: {}",
                normalized_id
            )));
        };
        let raw = fs::read_to_string(&target.file_path).map_err(|err| {
            AppError::Runtime(format!(
                "failed to read session file {}: {err}",
                target.file_path.display()
            ))
        })?;
        let mut state = serde_json::from_str::<SessionState>(&raw).map_err(|err| {
            AppError::Runtime(format!(
                "failed to parse session file {}: {err}",
                target.file_path.display()
            ))
        })?;
        state.session_name = normalize_session_name(&state.session_id, Some(new_name));
        state.compass.last_updated_epoch_ms = now_epoch_ms();
        let encoded = serde_json::to_string_pretty(&state).map_err(|err| {
            AppError::Runtime(format!("failed to serialize session state: {err}"))
        })?;
        write_string_atomically(&target.file_path, &encoded)?;
        Ok(build_session_overview(&state, target.file_path, &self.path))
    }

    pub fn delete_session_by_id(&mut self, session_id: &str) -> Result<SessionOverview, AppError> {
        let normalized_id = session_id.trim();
        if normalized_id.is_empty() {
            return Err(AppError::Runtime("session id cannot be empty".to_string()));
        }
        let sessions = self.list_sessions()?;
        let Some(target) = sessions
            .iter()
            .find(|item| item.session_id == normalized_id)
            .cloned()
        else {
            return Err(AppError::Runtime(format!(
                "session not found: {}",
                normalized_id
            )));
        };
        let deleting_active =
            target.file_path == self.path || target.session_id == self.state.session_id;
        if deleting_active {
            let fallback = sessions
                .iter()
                .find(|item| item.session_id != target.session_id)
                .cloned();
            if let Some(next) = fallback {
                let _ = self.switch_session_by_query(&next.session_id)?;
            } else {
                self.start_new_session_with_new_file()?;
            }
        }
        remove_session_files(&target.file_path)?;
        Ok(build_session_overview(
            &self.state,
            self.path.clone(),
            &self.path,
        ))
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionOverview>, AppError> {
        let parent = self.path.parent().ok_or_else(|| {
            AppError::Runtime("failed to resolve session directory for listing".to_string())
        })?;
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create session directory {}: {err}",
                parent.display()
            ))
        })?;

        let mut sessions = Vec::<SessionOverview>::new();
        let entries = fs::read_dir(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to read session directory {}: {err}",
                parent.display()
            ))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or_default()
                .to_string();
            if !is_session_state_file_name(&file_name) {
                continue;
            }
            let raw = match fs::read_to_string(&path) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let mut state = match serde_json::from_str::<SessionState>(&raw) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let expected_name =
                normalize_session_name(&state.session_id, Some(&state.session_name));
            if state.session_name != expected_name {
                state.session_name = expected_name;
            }
            sessions.push(build_session_overview(&state, path, &self.path));
        }

        if sessions.is_empty() {
            sessions.push(build_session_overview(
                &self.state,
                self.path.clone(),
                &self.path,
            ));
        }

        sessions.sort_by(|a, b| {
            b.last_updated_epoch_ms
                .cmp(&a.last_updated_epoch_ms)
                .then_with(|| b.created_at_epoch_ms.cmp(&a.created_at_epoch_ms))
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
        Ok(sessions)
    }

    pub fn switch_session_by_query(&mut self, query: &str) -> Result<SessionOverview, AppError> {
        let normalized_query = query.trim();
        if normalized_query.is_empty() {
            return Err(AppError::Runtime(
                "session query cannot be empty".to_string(),
            ));
        }
        let sessions = self.list_sessions()?;
        let q = normalized_query.to_ascii_lowercase();
        let mut exact = Vec::<SessionOverview>::new();
        let mut fuzzy = Vec::<SessionOverview>::new();
        for item in sessions {
            let id = item.session_id.as_str();
            let name = item.session_name.as_str();
            let name_l = name.to_ascii_lowercase();
            if id == normalized_query || name.eq_ignore_ascii_case(normalized_query) {
                exact.push(item);
                continue;
            }
            if id.starts_with(normalized_query) || name_l.starts_with(&q) {
                fuzzy.push(item);
            }
        }
        let selected = if exact.len() == 1 {
            exact.remove(0)
        } else if exact.len() > 1 {
            return Err(AppError::Runtime(format!(
                "session query is ambiguous: {}",
                normalized_query
            )));
        } else if fuzzy.len() == 1 {
            fuzzy.remove(0)
        } else if fuzzy.is_empty() {
            return Err(AppError::Runtime(format!(
                "session not found: {}",
                normalized_query
            )));
        } else {
            return Err(AppError::Runtime(format!(
                "session query is ambiguous: {}",
                normalized_query
            )));
        };

        if selected.file_path != self.path {
            let raw = fs::read_to_string(&selected.file_path).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to read session file {}: {err}",
                    selected.file_path.display()
                ))
            })?;
            let mut state = serde_json::from_str::<SessionState>(&raw).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to parse session file {}: {err}",
                    selected.file_path.display()
                ))
            })?;
            state.session_name =
                normalize_session_name(&state.session_id, Some(&state.session_name));
            self.state = state;
            self.path = selected.file_path.clone();
            self.repair_compass();
            self.enforce_max_limit();
            self.persist()?;
        }
        self.persist_active_session_pointer()?;
        Ok(build_session_overview(
            &self.state,
            self.path.clone(),
            &self.path,
        ))
    }

    pub fn has_chat_profile(&self) -> bool {
        self.state.messages.iter().any(|msg| {
            msg.role == "system" && msg.content.trim_start().starts_with(CHAT_PROFILE_MARKER)
        })
    }

    pub fn build_ai_compression_plan(&self) -> Option<AiCompressionPlan> {
        let total_messages = self.state.messages.len();
        let total_chars = self.total_message_chars();
        if total_messages <= self.compression_max_history_messages
            || total_chars <= self.compression_max_chars_count
        {
            return None;
        }
        let candidate_end = total_messages.saturating_sub(self.compression_keep_recent_messages);
        if candidate_end == 0 {
            return None;
        }
        let start = compression_start_index(&self.state.messages, candidate_end);
        if start >= candidate_end {
            return None;
        }
        let previous_summaries = self.state.messages[..candidate_end]
            .iter()
            .filter(|msg| is_ai_compression_message(msg))
            .map(|msg| strip_marker(msg.content.trim(), AI_COMPRESSION_MARKER))
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>();
        let candidate = self.state.messages[start..candidate_end]
            .iter()
            .filter(|msg| !is_ai_compression_message(msg))
            .cloned()
            .collect::<Vec<_>>();
        if candidate.is_empty() {
            return None;
        }
        let transcript = candidate
            .iter()
            .map(|msg| {
                format!(
                    "role={}\nkind={:?}\ncontent={}",
                    msg.role,
                    msg.kind,
                    mask_sensitive(&trim_chars(msg.content.trim(), 500))
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n");
        Some(AiCompressionPlan {
            candidate_messages: candidate.len(),
            previous_summaries,
            transcript,
        })
    }

    pub fn apply_ai_compression_summary(
        &mut self,
        summary: &str,
    ) -> Option<AiCompressionApplyResult> {
        let normalized = summary.trim();
        if normalized.is_empty() {
            return None;
        }
        let candidate_end = self
            .state
            .messages
            .len()
            .saturating_sub(self.compression_keep_recent_messages);
        if candidate_end == 0 {
            return None;
        }
        let start = compression_start_index(&self.state.messages, candidate_end);
        if start >= candidate_end {
            return None;
        }
        let removed = candidate_end.saturating_sub(start);
        if removed == 0 {
            return None;
        }
        self.state.messages.drain(start..candidate_end);
        self.state.messages.insert(
            start,
            SessionMessage {
                role: "system".to_string(),
                content: format!("{AI_COMPRESSION_MARKER}\n{normalized}"),
                kind: MessageKind::System,
                group_id: None,
                created_at_epoch_ms: now_epoch_ms(),
                tool_meta: None,
            },
        );
        self.state.compass.truncated_messages += removed;
        self.state.compass.compression_rounds += 1;
        self.state.compass.last_compaction_preview = trim_chars(normalized, 180);
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
        Some(AiCompressionApplyResult {
            removed_messages: removed,
            total_messages: self.state.messages.len(),
        })
    }

    pub fn wrap_chat_profile(content: &str) -> String {
        format!("{CHAT_PROFILE_MARKER}\n{}", content.trim())
    }

    fn add_message(
        &mut self,
        role: &str,
        kind: MessageKind,
        content: String,
        group_id: Option<String>,
        tool_meta: Option<ToolExecutionMeta>,
    ) {
        self.update_compass_on_append(role, &content);
        self.state.messages.push(SessionMessage {
            role: role.to_string(),
            content,
            kind,
            group_id,
            created_at_epoch_ms: now_epoch_ms(),
            tool_meta,
        });
        self.enforce_max_limit();
    }

    fn enforce_max_limit(&mut self) {
        if self.state.messages.len() <= self.max_limit {
            return;
        }

        let mut drop_count = self.state.messages.len() - self.max_limit;
        if drop_count < self.state.messages.len()
            && let Some(group_id) = self.state.messages[drop_count].group_id.as_deref()
        {
            while drop_count < self.state.messages.len()
                && self.state.messages[drop_count].group_id.as_deref() == Some(group_id)
            {
                drop_count += 1;
            }
        }

        if drop_count == 0 {
            return;
        }

        let removed: Vec<SessionMessage> = self.state.messages.drain(0..drop_count).collect();
        let compressed = compress_messages_semantic(&removed);
        if !compressed.is_empty() {
            self.state.summary = merge_summary(&self.state.summary, &compressed);
            self.state.compass.last_compaction_preview = trim_chars(&compressed, 180);
        }
        self.state.compass.truncated_messages += removed.len();
        self.state.compass.compression_rounds += 1;
        self.state.compass.dropped_groups += unique_group_count(&removed);
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
    }

    fn update_compass_on_append(&mut self, role: &str, content: &str) {
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
        match role {
            "user" => {
                self.state.compass.total_user_messages += 1;
                self.state.compass.last_user_topic = extract_topic(content);
                if content.starts_with("action=") {
                    self.state.compass.last_action = trim_chars(content, 80);
                }
            }
            "assistant" => {
                self.state.compass.total_assistant_messages += 1;
                self.state.compass.last_assistant_focus = first_sentence(content, 120);
            }
            "tool" => {
                self.state.compass.total_tool_messages += 1;
            }
            _ => {}
        }
    }

    fn update_compass_on_remove(&mut self, removed: &SessionMessage) {
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
        match removed.role.as_str() {
            "user" => {
                self.state.compass.total_user_messages =
                    self.state.compass.total_user_messages.saturating_sub(1);
                self.state.compass.last_user_topic = self
                    .state
                    .messages
                    .iter()
                    .rev()
                    .find(|msg| msg.role == "user")
                    .map(|msg| extract_topic(&msg.content))
                    .unwrap_or_default();
                self.state.compass.last_action = self
                    .state
                    .messages
                    .iter()
                    .rev()
                    .find(|msg| msg.role == "user" && msg.content.starts_with("action="))
                    .map(|msg| trim_chars(&msg.content, 80))
                    .unwrap_or_default();
            }
            "assistant" => {
                self.state.compass.total_assistant_messages = self
                    .state
                    .compass
                    .total_assistant_messages
                    .saturating_sub(1);
                self.state.compass.last_assistant_focus = self
                    .state
                    .messages
                    .iter()
                    .rev()
                    .find(|msg| msg.role == "assistant")
                    .map(|msg| first_sentence(&msg.content, 120))
                    .unwrap_or_default();
            }
            "tool" => {
                self.state.compass.total_tool_messages =
                    self.state.compass.total_tool_messages.saturating_sub(1);
            }
            _ => {}
        }
    }

    fn compass_snapshot(&self) -> String {
        let last_action = fallback_dash(&self.state.compass.last_action);
        let last_topic = fallback_dash(&self.state.compass.last_user_topic);
        format!(
            "session_id={}, messages={}, summary_chars={}, compressed={}, dropped={}, user={}, assistant={}, tool={}, last_action={}, last_topic={}",
            self.state.session_id,
            self.state.messages.len(),
            self.state.summary.chars().count(),
            self.state.compass.compression_rounds,
            self.state.compass.truncated_messages,
            self.state.compass.total_user_messages,
            self.state.compass.total_assistant_messages,
            self.state.compass.total_tool_messages,
            mask_sensitive(&last_action),
            mask_sensitive(&last_topic)
        )
    }

    fn repair_compass(&mut self) {
        let now = now_epoch_ms();
        if self.state.session_id.trim().is_empty() {
            self.state.session_id = default_session_id();
        }
        self.state.session_name =
            normalize_session_name(&self.state.session_id, Some(&self.state.session_name));
        let persisted_counts = self.archived_role_counts();
        self.state.compass.total_user_messages = self
            .state
            .compass
            .total_user_messages
            .max(persisted_counts.user);
        self.state.compass.total_assistant_messages = self
            .state
            .compass
            .total_assistant_messages
            .max(persisted_counts.assistant);
        self.state.compass.total_tool_messages = self
            .state
            .compass
            .total_tool_messages
            .max(persisted_counts.tool);
        if self.state.compass.last_user_topic.trim().is_empty()
            && let Some(message) = self
                .state
                .messages
                .iter()
                .rev()
                .find(|msg| msg.role == "user")
        {
            self.state.compass.last_user_topic = extract_topic(&message.content);
        }
        if self.state.compass.last_action.trim().is_empty()
            && let Some(message) = self
                .state
                .messages
                .iter()
                .rev()
                .find(|msg| msg.role == "user" && msg.content.starts_with("action="))
        {
            self.state.compass.last_action = trim_chars(&message.content, 80);
        }
        if self.state.compass.last_assistant_focus.trim().is_empty()
            && let Some(message) = self
                .state
                .messages
                .iter()
                .rev()
                .find(|msg| msg.role == "assistant")
        {
            self.state.compass.last_assistant_focus = first_sentence(&message.content, 120);
        }
        let first_message_ts = self
            .state
            .messages
            .iter()
            .map(|msg| msg.created_at_epoch_ms)
            .min()
            .unwrap_or(now);
        let last_message_ts = self
            .state
            .messages
            .iter()
            .map(|msg| msg.created_at_epoch_ms)
            .max()
            .unwrap_or(now);
        if self.state.compass.created_at_epoch_ms == 0 {
            self.state.compass.created_at_epoch_ms = first_message_ts;
        }
        if self.state.compass.last_updated_epoch_ms == 0 {
            self.state.compass.last_updated_epoch_ms = last_message_ts;
        } else {
            self.state.compass.last_updated_epoch_ms = self
                .state
                .compass
                .last_updated_epoch_ms
                .max(last_message_ts);
        }
    }

    fn persist_active_session_pointer(&self) -> Result<(), AppError> {
        let base_dir = machineclaw_base_dir_from_session_path(&self.path);
        fs::create_dir_all(&base_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create session directory {}: {err}",
                base_dir.display()
            ))
        })?;
        let pointer_file = base_dir.join("active_session");
        let pointer_value = self
            .path
            .strip_prefix(&base_dir)
            .ok()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| self.path.to_string_lossy().to_string());
        write_string_atomically(&pointer_file, &pointer_value)
    }

    fn ensure_session_path_in_sessions_dir(&mut self) -> Result<(), AppError> {
        let target_dir = session_dir_from_session_path(&self.path);
        let current_dir = self.path.parent().map(Path::to_path_buf);
        if current_dir.as_ref() == Some(&target_dir) {
            return Ok(());
        }
        fs::create_dir_all(&target_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create session directory {}: {err}",
                target_dir.display()
            ))
        })?;
        let current_file_name = self
            .path
            .file_name()
            .and_then(|item| item.to_str())
            .unwrap_or("session.json")
            .to_string();
        let mut target_path = target_dir.join(current_file_name);
        if target_path.exists() && target_path != self.path {
            target_path = target_dir.join(format!("session-{}.json", self.state.session_id));
        }
        self.path = target_path;
        self.persist()
    }
}

fn is_visible_display_message(item: &SessionMessage) -> bool {
    if item.role.trim().is_empty() && item.content.trim().is_empty() {
        return false;
    }
    if item.role == "system"
        && (item.content.trim_start().starts_with(AI_COMPRESSION_MARKER)
            || item.content.trim_start().starts_with(CHAT_PROFILE_MARKER))
    {
        return false;
    }
    true
}

fn is_persisted_tool_trace_message(content: &str) -> bool {
    let trimmed = content.trim();
    trimmed.starts_with("tool_call_id=")
        && trimmed.contains(" function=")
        && trimmed.contains(" args=")
        && trimmed.contains(" result=")
}

fn compute_keep_recent_messages(max_history_messages: usize) -> usize {
    let compress_recent = max_history_messages / 2;
    max_history_messages.saturating_sub(compress_recent).max(1)
}

fn load_state_with_autosave_or_new(path: &Path) -> Result<SessionState, AppError> {
    let autosave_path = autosave_file_path_for(path);
    let primary = if path.exists() {
        Some(read_state_from_file(path).or_else(|primary_err| {
            try_read_state_from_file(&autosave_path).map_err(|_| primary_err)
        })?)
    } else {
        None
    };
    let autosave = try_read_state_from_file(&autosave_path);
    match (primary, autosave) {
        (Some(primary_state), Ok(autosave_state)) => {
            if session_state_order_key(&autosave_state) >= session_state_order_key(&primary_state) {
                Ok(autosave_state)
            } else {
                Ok(primary_state)
            }
        }
        (Some(primary_state), Err(_)) => Ok(primary_state),
        (None, Ok(autosave_state)) => Ok(autosave_state),
        (None, Err(_)) => Ok(SessionState {
            session_id: Uuid::new_v4().to_string(),
            session_name: String::new(),
            summary: String::new(),
            messages: Vec::new(),
            compass: new_compass(),
        }),
    }
}

fn read_state_from_file(path: &Path) -> Result<SessionState, AppError> {
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read session file {}: {err}",
            path.display()
        ))
    })?;
    match serde_json::from_str::<SessionState>(&raw) {
        Ok(state) => Ok(state),
        Err(strict_err) => parse_session_state_lenient(&raw).ok_or_else(|| {
            AppError::Runtime(format!(
                "failed to parse session file {}: {strict_err}",
                path.display()
            ))
        }),
    }
}

fn try_read_state_from_file(path: &Path) -> Result<SessionState, AppError> {
    if !path.exists() {
        return Err(AppError::Runtime(format!(
            "session file not found: {}",
            path.display()
        )));
    }
    read_state_from_file(path)
}

fn session_state_order_key(state: &SessionState) -> (u128, usize) {
    (
        state.compass.last_updated_epoch_ms,
        state
            .messages
            .len()
            .saturating_add(state.summary.chars().count()),
    )
}

fn autosave_file_path_for(path: &Path) -> PathBuf {
    let mut value = OsString::from(path.as_os_str());
    value.push(".autosave");
    PathBuf::from(value)
}

fn remove_session_files(path: &Path) -> Result<(), AppError> {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => {
            return Err(AppError::Runtime(format!(
                "failed to remove session file {}: {err}",
                path.display()
            )));
        }
    }
    let autosave_path = autosave_file_path_for(path);
    match fs::remove_file(&autosave_path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => {
            return Err(AppError::Runtime(format!(
                "failed to remove session autosave file {}: {err}",
                autosave_path.display()
            )));
        }
    }
    Ok(())
}

fn parse_session_state_lenient(raw: &str) -> Option<SessionState> {
    let value = serde_json::from_str::<serde_json::Value>(raw).ok()?;
    let object = value.as_object()?;
    let mut state = SessionState {
        session_id: value_to_string(object.get("session_id")).unwrap_or_else(default_session_id),
        session_name: value_to_string(object.get("session_name")).unwrap_or_default(),
        summary: value_to_string(object.get("summary")).unwrap_or_default(),
        messages: object
            .get("messages")
            .and_then(|item| item.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(parse_session_message_lenient)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        compass: parse_session_compass_lenient(object.get("compass")),
    };
    if state.session_id.trim().is_empty() {
        state.session_id = default_session_id();
    }
    Some(state)
}

fn parse_session_message_lenient(value: &serde_json::Value) -> Option<SessionMessage> {
    let object = value.as_object()?;
    let raw_role = value_to_string(object.get("role")).unwrap_or_default();
    let kind = parse_message_kind_lenient(object.get("kind"), raw_role.as_str());
    let normalized_role = raw_role.trim().to_ascii_lowercase();
    let role = if matches!(
        normalized_role.as_str(),
        "user" | "assistant" | "tool" | "system" | "thinking"
    ) {
        normalized_role
    } else {
        default_role_for_kind(kind)
    };
    let content = value_to_string(object.get("content")).unwrap_or_default();
    if role.trim().is_empty() && content.trim().is_empty() {
        return None;
    }
    let group_id = value_to_string(object.get("group_id")).and_then(|item| {
        let trimmed = item.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    let created_at_epoch_ms = value_to_u128(object.get("created_at_epoch_ms")).unwrap_or_default();
    let tool_meta = parse_tool_execution_meta_lenient(object.get("tool_meta"));
    Some(SessionMessage {
        role,
        content,
        kind,
        group_id,
        created_at_epoch_ms,
        tool_meta,
    })
}

fn parse_tool_execution_meta_lenient(value: Option<&serde_json::Value>) -> Option<ToolExecutionMeta> {
    let object = value?.as_object()?;
    let meta = ToolExecutionMeta {
        tool_call_id: value_to_string(object.get("tool_call_id")).unwrap_or_default(),
        function_name: value_to_string(object.get("function_name")).unwrap_or_default(),
        command: value_to_string(object.get("command")).unwrap_or_default(),
        arguments: value_to_string(object.get("arguments")).unwrap_or_default(),
        result_payload: value_to_string(object.get("result_payload")).unwrap_or_default(),
        executed_at_epoch_ms: value_to_u128(object.get("executed_at_epoch_ms")).unwrap_or_default(),
        account: value_to_string(object.get("account")).unwrap_or_default(),
        environment: value_to_string(object.get("environment")).unwrap_or_default(),
        os_name: value_to_string(object.get("os_name")).unwrap_or_default(),
        cwd: value_to_string(object.get("cwd")).unwrap_or_default(),
        mode: value_to_string(object.get("mode")).unwrap_or_default(),
        label: value_to_string(object.get("label")).unwrap_or_default(),
        exit_code: value_to_i32(object.get("exit_code")),
        duration_ms: value_to_u128(object.get("duration_ms")).unwrap_or_default(),
        timed_out: value_to_bool(object.get("timed_out")).unwrap_or(false),
        interrupted: value_to_bool(object.get("interrupted")).unwrap_or(false),
        blocked: value_to_bool(object.get("blocked")).unwrap_or(false),
    };
    let has_payload = !meta.tool_call_id.trim().is_empty()
        || !meta.function_name.trim().is_empty()
        || !meta.command.trim().is_empty()
        || !meta.arguments.trim().is_empty()
        || !meta.result_payload.trim().is_empty()
        || meta.executed_at_epoch_ms > 0;
    if has_payload {
        Some(meta)
    } else {
        None
    }
}

fn parse_session_compass_lenient(value: Option<&serde_json::Value>) -> SessionCompass {
    let mut compass = SessionCompass::default();
    let Some(object) = value.and_then(|item| item.as_object()) else {
        return compass;
    };
    compass.created_at_epoch_ms = value_to_u128(object.get("created_at_epoch_ms")).unwrap_or(0);
    compass.last_updated_epoch_ms = value_to_u128(object.get("last_updated_epoch_ms")).unwrap_or(0);
    compass.truncated_messages = value_to_usize(object.get("truncated_messages")).unwrap_or(0);
    compass.compression_rounds = value_to_usize(object.get("compression_rounds")).unwrap_or(0);
    compass.dropped_groups = value_to_usize(object.get("dropped_groups")).unwrap_or(0);
    compass.total_user_messages = value_to_usize(object.get("total_user_messages")).unwrap_or(0);
    compass.total_assistant_messages =
        value_to_usize(object.get("total_assistant_messages")).unwrap_or(0);
    compass.total_tool_messages = value_to_usize(object.get("total_tool_messages")).unwrap_or(0);
    compass.last_action = value_to_string(object.get("last_action")).unwrap_or_default();
    compass.last_user_topic = value_to_string(object.get("last_user_topic")).unwrap_or_default();
    compass.last_assistant_focus =
        value_to_string(object.get("last_assistant_focus")).unwrap_or_default();
    compass.last_compaction_preview =
        value_to_string(object.get("last_compaction_preview")).unwrap_or_default();
    compass
}

fn parse_message_kind_lenient(value: Option<&serde_json::Value>, role: &str) -> MessageKind {
    let by_kind = value_to_string(value)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let by_role = role.trim().to_ascii_lowercase();
    match by_kind.as_str() {
        "user" => MessageKind::User,
        "assistant" => MessageKind::Assistant,
        "tool" => MessageKind::Tool,
        "system" => MessageKind::System,
        _ => match by_role.as_str() {
            "assistant" => MessageKind::Assistant,
            "tool" => MessageKind::Tool,
            "system" => MessageKind::System,
            "thinking" => MessageKind::Assistant,
            _ => MessageKind::User,
        },
    }
}

fn default_role_for_kind(kind: MessageKind) -> String {
    match kind {
        MessageKind::User => "user".to_string(),
        MessageKind::Assistant => "assistant".to_string(),
        MessageKind::Tool => "tool".to_string(),
        MessageKind::System => "system".to_string(),
    }
}

fn value_to_string(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(item) => Some(item.clone()),
        serde_json::Value::Number(item) => Some(item.to_string()),
        serde_json::Value::Bool(item) => Some(item.to_string()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn value_to_u128(value: Option<&serde_json::Value>) -> Option<u128> {
    match value? {
        serde_json::Value::Number(item) => item
            .as_u64()
            .map(|v| v as u128)
            .or_else(|| item.as_i64().filter(|v| *v >= 0).map(|v| v as u128)),
        serde_json::Value::String(item) => item.trim().parse::<u128>().ok(),
        _ => None,
    }
}

fn value_to_usize(value: Option<&serde_json::Value>) -> Option<usize> {
    value_to_u128(value).map(|item| item.min(usize::MAX as u128) as usize)
}

fn value_to_i32(value: Option<&serde_json::Value>) -> Option<i32> {
    match value? {
        serde_json::Value::Number(item) => item
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .or_else(|| item.as_u64().and_then(|v| i32::try_from(v).ok())),
        serde_json::Value::String(item) => item.trim().parse::<i32>().ok(),
        _ => None,
    }
}

fn value_to_bool(value: Option<&serde_json::Value>) -> Option<bool> {
    match value? {
        serde_json::Value::Bool(item) => Some(*item),
        serde_json::Value::Number(item) => item
            .as_i64()
            .map(|v| v != 0)
            .or_else(|| item.as_u64().map(|v| v != 0)),
        serde_json::Value::String(item) => {
            let normalized = item.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "true" | "1" | "yes" | "y" => Some(true),
                "false" | "0" | "no" | "n" => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_session_state_file_name(file_name: &str) -> bool {
    if !file_name.ends_with(".json") {
        return false;
    }
    if file_name == "session.json" {
        return true;
    }
    file_name.starts_with("session-")
}

fn default_session_name(session_id: &str) -> String {
    format!("session-{session_id}")
}

fn default_session_id() -> String {
    Uuid::new_v4().to_string()
}

fn normalize_session_name(session_id: &str, raw_name: Option<&str>) -> String {
    let trimmed = raw_name.unwrap_or_default().trim();
    if trimmed.is_empty() {
        return default_session_name(session_id);
    }
    let limited = trim_chars(trimmed, 80);
    if limited.trim().is_empty() {
        return default_session_name(session_id);
    }
    limited
}

fn build_session_overview(
    state: &SessionState,
    file_path: PathBuf,
    active_path: &Path,
) -> SessionOverview {
    let mut user_count = 0usize;
    let mut assistant_count = 0usize;
    let mut tool_count = 0usize;
    let mut system_count = 0usize;
    for item in &state.messages {
        match item.role.as_str() {
            "user" => user_count += 1,
            "assistant" => assistant_count += 1,
            "tool" => tool_count += 1,
            "system" => system_count += 1,
            _ => {}
        }
    }
    SessionOverview {
        session_id: state.session_id.clone(),
        session_name: normalize_session_name(&state.session_id, Some(&state.session_name)),
        message_count: state.messages.len(),
        summary_len: state.summary.chars().count(),
        created_at_epoch_ms: state.compass.created_at_epoch_ms,
        last_updated_epoch_ms: state.compass.last_updated_epoch_ms,
        file_path: file_path.clone(),
        user_count,
        assistant_count,
        tool_count,
        system_count,
        active: file_path == active_path,
    }
}

fn new_compass() -> SessionCompass {
    let now = now_epoch_ms();
    SessionCompass {
        created_at_epoch_ms: now,
        last_updated_epoch_ms: now,
        ..SessionCompass::default()
    }
}

fn compress_messages_semantic(messages: &[SessionMessage]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut user_intents = Vec::<String>::new();
    let mut assistant_conclusion = String::new();
    let mut freq = HashMap::<String, usize>::new();
    let mut tool_ok = 0usize;
    let mut tool_fail = 0usize;
    let mut tool_timeout = 0usize;
    let mut tool_blocked = 0usize;

    for item in messages {
        match item.role.as_str() {
            "user" => {
                if user_intents.len() < 3 {
                    user_intents.push(trim_chars(&item.content.replace('\n', " "), 56));
                }
            }
            "assistant" => {
                if assistant_conclusion.is_empty() {
                    assistant_conclusion = first_sentence(&item.content, 100);
                }
            }
            "tool" => {
                let lowered = item.content.to_ascii_lowercase();
                if lowered.contains("\"ok\":true") || lowered.contains("\"success\":true") {
                    tool_ok += 1;
                }
                if lowered.contains("\"ok\":false") || lowered.contains("\"success\":false") {
                    tool_fail += 1;
                }
                if lowered.contains("\"timed_out\":true") {
                    tool_timeout += 1;
                }
                if lowered.contains("\"blocked\":true") {
                    tool_blocked += 1;
                }
            }
            _ => {}
        }
        collect_keywords(&item.content, &mut freq);
    }

    let mut keywords = freq.into_iter().collect::<Vec<(String, usize)>>();
    keywords.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));
    let topics = keywords
        .into_iter()
        .map(|(k, _)| k)
        .filter(|k| !k.is_empty())
        .take(5)
        .collect::<Vec<String>>()
        .join("/");

    let intents = if user_intents.is_empty() {
        "-".to_string()
    } else {
        user_intents.join(" | ")
    };
    let conclusion = if assistant_conclusion.is_empty() {
        "-".to_string()
    } else {
        assistant_conclusion
    };

    format!(
        "topics={}; intents={}; tool(ok/fail/timeout/blocked)={}/{}/{}/{}; conclusion={}",
        fallback_dash(&topics),
        intents,
        tool_ok,
        tool_fail,
        tool_timeout,
        tool_blocked,
        conclusion
    )
}

fn compression_start_index(messages: &[SessionMessage], overflow_end: usize) -> usize {
    let mut last_marker = None;
    for (idx, msg) in messages.iter().enumerate().take(overflow_end) {
        if is_ai_compression_message(msg) {
            last_marker = Some(idx);
        }
    }
    last_marker.map(|v| v + 1).unwrap_or(0)
}

fn is_ai_compression_message(msg: &SessionMessage) -> bool {
    msg.role == "system" && msg.content.trim_start().starts_with(AI_COMPRESSION_MARKER)
}

fn strip_marker(content: &str, marker: &str) -> String {
    content
        .strip_prefix(marker)
        .unwrap_or(content)
        .trim()
        .to_string()
}

fn collect_keywords(content: &str, freq: &mut HashMap<String, usize>) {
    const ASCII_STOPWORDS: [&str; 16] = [
        "the", "and", "for", "that", "with", "from", "this", "have", "will", "into", "your", "you",
        "are", "was", "were", "been",
    ];
    let mut ascii = String::new();
    let mut cjk = String::new();

    let flush_ascii = |buf: &mut String, map: &mut HashMap<String, usize>| {
        if buf.len() < 3 {
            buf.clear();
            return;
        }
        let word = buf.to_ascii_lowercase();
        if !ASCII_STOPWORDS.contains(&word.as_str()) {
            *map.entry(word).or_insert(0) += 1;
        }
        buf.clear();
    };

    let flush_cjk = |buf: &mut String, map: &mut HashMap<String, usize>| {
        let len = buf.chars().count();
        if (2..=12).contains(&len) {
            *map.entry(buf.clone()).or_insert(0) += 1;
        }
        buf.clear();
    };

    for ch in content.chars() {
        if ch.is_ascii_alphanumeric() {
            ascii.push(ch);
            flush_cjk(&mut cjk, freq);
            continue;
        }
        if is_cjk(ch) {
            cjk.push(ch);
            flush_ascii(&mut ascii, freq);
            continue;
        }
        flush_ascii(&mut ascii, freq);
        flush_cjk(&mut cjk, freq);
    }
    flush_ascii(&mut ascii, freq);
    flush_cjk(&mut cjk, freq);
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF)
}

fn merge_summary(existing: &str, appended: &str) -> String {
    let merged = if existing.trim().is_empty() {
        appended.to_string()
    } else {
        format!("{}\n• {}", existing.trim(), appended.trim())
    };
    trim_tail_chars(&merged, SUMMARY_MAX_CHARS)
}

fn unique_group_count(messages: &[SessionMessage]) -> usize {
    let mut set = HashSet::<String>::new();
    for item in messages {
        if let Some(group_id) = item.group_id.as_deref()
            && !group_id.trim().is_empty()
        {
            set.insert(group_id.to_string());
        }
    }
    set.len()
}

fn extract_topic(text: &str) -> String {
    let mut freq = HashMap::<String, usize>::new();
    collect_keywords(text, &mut freq);
    if freq.is_empty() {
        return trim_chars(text, 40);
    }
    let mut sorted = freq.into_iter().collect::<Vec<(String, usize)>>();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));
    sorted
        .into_iter()
        .map(|(k, _)| k)
        .take(3)
        .collect::<Vec<String>>()
        .join("/")
}

fn first_sentence(text: &str, max_chars: usize) -> String {
    let collapsed = text.replace('\n', " ").trim().to_string();
    if collapsed.is_empty() {
        return String::new();
    }
    let mut end = collapsed.len();
    for (idx, ch) in collapsed.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
            end = idx + ch.len_utf8();
            break;
        }
    }
    trim_chars(&collapsed[..end], max_chars)
}

fn trim_tail_chars(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let skip = total.saturating_sub(max_chars);
    text.chars().skip(skip).collect::<String>()
}

fn trim_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let trimmed: String = text.chars().take(max_chars).collect();
    format!("{}...", trimmed)
}

fn fallback_dash(text: &str) -> String {
    if text.trim().is_empty() {
        return "-".to_string();
    }
    text.to_string()
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn machineclaw_base_dir_from_session_path(session_path: &Path) -> PathBuf {
    let Some(parent) = session_path.parent() else {
        return PathBuf::from(".machineclaw");
    };
    let parent_name = parent.file_name().and_then(|item| item.to_str());
    if parent_name == Some("sessions") {
        return parent
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| parent.to_path_buf());
    }
    parent.to_path_buf()
}

fn session_dir_from_session_path(session_path: &Path) -> PathBuf {
    machineclaw_base_dir_from_session_path(session_path).join("sessions")
}

fn write_string_atomically(path: &Path, content: &str) -> Result<(), AppError> {
    let parent = path.parent().ok_or_else(|| {
        AppError::Runtime(format!(
            "failed to resolve parent directory for {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create parent directory {}: {err}",
            parent.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("session");
    let temp_path = parent.join(format!(".{}.{}.tmp", file_name, Uuid::new_v4()));
    fs::write(&temp_path, content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write temporary file {}: {err}",
            temp_path.display()
        ))
    })?;
    if let Err(rename_err) = fs::rename(&temp_path, path) {
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return Err(AppError::Runtime(format!(
                "failed to replace file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            )));
        }
        let _ = fs::remove_file(&temp_path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AI_COMPRESSION_MARKER, CHAT_PROFILE_MARKER, MessageKind, SessionRoleCounts, SessionState,
        SessionStore, ToolExecutionMeta, new_compass,
    };
    use std::{fs, path::PathBuf};
    use uuid::Uuid;

    fn build_store(summary: &str) -> SessionStore {
        SessionStore {
            path: PathBuf::from("/tmp/session.json"),
            state: SessionState {
                session_id: "session-1".to_string(),
                session_name: "session-1".to_string(),
                summary: summary.to_string(),
                messages: vec![
                    super::SessionMessage {
                        role: "user".to_string(),
                        content: "hello".to_string(),
                        kind: MessageKind::User,
                        group_id: None,
                        created_at_epoch_ms: 1,
                        tool_meta: None,
                    },
                    super::SessionMessage {
                        role: "assistant".to_string(),
                        content: "world".to_string(),
                        kind: MessageKind::Assistant,
                        group_id: None,
                        created_at_epoch_ms: 2,
                        tool_meta: None,
                    },
                    super::SessionMessage {
                        role: "tool".to_string(),
                        content: "{}".to_string(),
                        kind: MessageKind::Tool,
                        group_id: None,
                        created_at_epoch_ms: 3,
                        tool_meta: None,
                    },
                ],
                compass: new_compass(),
            },
            recent_limit: 40,
            max_limit: 80,
            compression_max_history_messages: 40,
            compression_max_chars_count: 80_000,
            compression_keep_recent_messages: 20,
        }
    }

    #[test]
    fn archived_role_counts_only_counts_persisted_messages() {
        let store = build_store("");
        let counts = store.archived_role_counts();
        assert_eq!(counts.total, 3);
        assert_eq!(counts.user, 1);
        assert_eq!(counts.assistant, 1);
        assert_eq!(counts.tool, 1);
        assert_eq!(counts.system, 0);
    }

    #[test]
    fn effective_context_role_counts_include_ephemeral_system_messages() {
        let store = build_store("summary");
        let counts = store.effective_context_role_counts(true);
        assert_eq!(
            counts,
            SessionRoleCounts {
                total: 6,
                user: 1,
                assistant: 1,
                tool: 1,
                system: 3,
            }
        );
    }

    #[test]
    fn repair_compass_backfills_missing_counts_and_topics() {
        let mut store = build_store("");
        store.state.compass.total_user_messages = 0;
        store.state.compass.total_assistant_messages = 0;
        store.state.compass.total_tool_messages = 0;
        store.state.compass.last_user_topic.clear();
        store.state.compass.last_assistant_focus.clear();
        store.state.compass.last_updated_epoch_ms = 0;
        store.repair_compass();
        assert_eq!(store.state.compass.total_user_messages, 1);
        assert_eq!(store.state.compass.total_assistant_messages, 1);
        assert_eq!(store.state.compass.total_tool_messages, 1);
        assert!(!store.state.compass.last_user_topic.is_empty());
        assert!(!store.state.compass.last_assistant_focus.is_empty());
        assert_eq!(store.state.compass.last_updated_epoch_ms, 3);
    }

    #[test]
    fn recent_messages_for_display_skips_internal_system_markers() {
        let mut store = build_store("");
        store.add_system_message(format!("{AI_COMPRESSION_MARKER}\nsummary"), None);
        store.add_system_message(format!("{CHAT_PROFILE_MARKER}\nprofile"), None);
        store.add_assistant_message("assistant ok".to_string(), None);
        let items = store.recent_messages_for_display(10);
        assert!(items.iter().all(|item| {
            !item.content.starts_with(AI_COMPRESSION_MARKER)
                && !item.content.starts_with(CHAT_PROFILE_MARKER)
        }));
        assert!(items.iter().any(|item| item.content == "assistant ok"));
    }

    #[test]
    fn build_chat_history_skips_thinking_and_tool_progress_messages() {
        let mut store = build_store("");
        store.add_thinking_message("step-1".to_string(), Some("g1".to_string()));
        store.add_tool_message("Bash执行中: demo [读]\ndate".to_string(), Some("g1".to_string()));
        store.add_tool_message_with_meta(
            r#"tool_call_id=call_1 function=run_shell_command args={"command":"date"} result={"ok":true,"stdout":"Sun"}"#.to_string(),
            Some("g1".to_string()),
            Some(ToolExecutionMeta {
                tool_call_id: "call_1".to_string(),
                function_name: "run_shell_command".to_string(),
                command: "date".to_string(),
                arguments: "{\"command\":\"date\"}".to_string(),
                result_payload: "{\"ok\":true,\"stdout\":\"Sun\"}".to_string(),
                executed_at_epoch_ms: 1,
                account: "tester".to_string(),
                environment: "prod".to_string(),
                os_name: "macos".to_string(),
                cwd: "/tmp".to_string(),
                mode: "read".to_string(),
                label: "date".to_string(),
                exit_code: Some(0),
                duration_ms: 1,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        );

        let history = store.build_chat_history();
        let joined = history
            .iter()
            .map(|item| format!("{}|{}", item.role, item.content))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!joined.contains("step-1"));
        assert!(!joined.contains("Bash执行中"));
        assert!(joined.contains("[tool] tool_call_id=call_1"));
    }

    #[test]
    fn append_or_add_thinking_chunk_merges_last_thinking_message_in_same_group() {
        let mut store = build_store("");
        store.add_thinking_message("step-1".to_string(), Some("g1".to_string()));
        let before = store.message_count();
        store.append_or_add_thinking_chunk(" + step-2", Some("g1"));
        assert_eq!(store.message_count(), before);
        let last = store
            .recent_messages_for_display(50)
            .into_iter()
            .rev()
            .find(|item| item.role == "thinking")
            .expect("thinking message should exist");
        assert_eq!(last.content, "step-1 + step-2");
    }

    #[test]
    fn remove_recent_display_message_by_signature_removes_target_occurrence() {
        let mut store = build_store("");
        store.add_user_message("dup".to_string(), None);
        store.add_user_message("dup".to_string(), None);
        store.add_user_message("dup".to_string(), None);
        let removed = store.remove_recent_display_message_by_signature(20, "user", "dup", 2);
        assert!(removed.is_some());
        let user_dups = store
            .recent_messages_for_display(20)
            .into_iter()
            .filter(|item| item.role == "user" && item.content == "dup")
            .count();
        assert_eq!(user_dups, 2);
    }

    #[test]
    fn remove_recent_display_message_by_signature_ignores_internal_markers() {
        let mut store = build_store("");
        store.add_system_message(format!("{AI_COMPRESSION_MARKER}\nsummary"), None);
        let removed = store.remove_recent_display_message_by_signature(
            20,
            "system",
            format!("{AI_COMPRESSION_MARKER}\nsummary").as_str(),
            1,
        );
        assert!(removed.is_none());
    }

    #[test]
    fn autosave_file_path_appends_suffix() {
        let store = build_store("");
        let autosave = store.autosave_file_path();
        assert_eq!(
            autosave.file_name().and_then(|name| name.to_str()),
            Some("session.json.autosave")
        );
    }

    #[test]
    fn load_or_new_falls_back_to_valid_autosave_when_primary_is_corrupted() {
        let temp_dir = std::env::temp_dir().join(format!("machineclaw-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let primary_path = temp_dir.join("session.json");
        fs::write(&primary_path, "{invalid json").expect("primary file should be written");
        let autosave_path = PathBuf::from(format!("{}.autosave", primary_path.display()));
        let autosave_state = SessionState {
            session_id: "session-autosave".to_string(),
            session_name: "session-autosave".to_string(),
            summary: String::new(),
            messages: vec![super::SessionMessage {
                role: "user".to_string(),
                content: "recover me".to_string(),
                kind: MessageKind::User,
                group_id: None,
                created_at_epoch_ms: 123,
                tool_meta: None,
            }],
            compass: new_compass(),
        };
        let raw = serde_json::to_string_pretty(&autosave_state).expect("autosave state to json");
        fs::write(&autosave_path, raw).expect("autosave file should be written");

        let store = SessionStore::load_or_new(primary_path, 40, 80, 40, 80_000)
            .expect("load_or_new should recover from autosave");
        assert_eq!(store.session_id(), "session-autosave");
        assert_eq!(store.message_count(), 1);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn load_or_new_accepts_lenient_session_shape_with_missing_fields() {
        let temp_dir = std::env::temp_dir().join(format!("machineclaw-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let primary_path = temp_dir.join("session.json");
        let raw = r#"{
            "session_name": 123,
            "messages": [
                {"role": 1, "content": {"x": 1}, "kind": "assistant", "created_at_epoch_ms": "168"},
                {"content": "ok"}
            ],
            "compass": {"last_updated_epoch_ms": "456", "total_user_messages": "7"}
        }"#;
        fs::write(&primary_path, raw).expect("primary file should be written");

        let store = SessionStore::load_or_new(primary_path, 40, 80, 40, 80_000)
            .expect("lenient parser should recover session shape");
        assert!(!store.session_id().trim().is_empty());
        assert_eq!(store.message_count(), 2);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn tool_meta_roundtrip_preserves_full_result_payload() {
        let temp_dir = std::env::temp_dir().join(format!("machineclaw-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let session_path = temp_dir.join("session.json");
        let mut store = SessionStore::load_or_new(session_path.clone(), 40, 80, 40, 80_000)
            .expect("new session store should be created");
        let full_payload = "x".repeat(25_000);
        store.add_tool_message_with_meta(
            "tool_call_id=call_long function=run_shell_command args={\"command\":\"cat\"} result={\"ok\":true}".to_string(),
            Some("group-1".to_string()),
            Some(ToolExecutionMeta {
                tool_call_id: "call_long".to_string(),
                function_name: "run_shell_command".to_string(),
                command: "cat /tmp/huge.log".to_string(),
                arguments: "{\"command\":\"cat /tmp/huge.log\"}".to_string(),
                result_payload: full_payload.clone(),
                executed_at_epoch_ms: 123,
                account: "tester".to_string(),
                environment: "prod".to_string(),
                os_name: "macos".to_string(),
                cwd: "/tmp".to_string(),
                mode: "read".to_string(),
                label: "cat".to_string(),
                exit_code: Some(0),
                duration_ms: 12,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        );
        store.persist().expect("session should persist");
        let persisted_path = store.file_path().to_path_buf();
        let raw = fs::read_to_string(&persisted_path).expect("session file should be readable");
        let persisted: SessionState =
            serde_json::from_str(&raw).expect("session file should parse");
        let tool = persisted
            .messages
            .iter()
            .rev()
            .find(|item| item.role == "tool")
            .expect("tool message should exist");
        let meta = tool
            .tool_meta
            .as_ref()
            .expect("tool meta should be present");
        assert_eq!(meta.result_payload.len(), full_payload.len());
        assert_eq!(meta.result_payload, full_payload);

        let _ = fs::remove_dir_all(temp_dir);
    }
}
