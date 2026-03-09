use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ai::ChatMessage, error::AppError, i18n, mask::mask_sensitive};

const SUMMARY_MAX_CHARS: usize = 4000;
const AI_COMPRESSION_MARKER: &str = "[ai_summary_compression]";
const CHAT_PROFILE_MARKER: &str = "[chat_profile_v1]";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    pub kind: MessageKind,
    pub group_id: Option<String>,
    pub created_at_epoch_ms: u128,
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
    pub session_id: String,
    #[serde(default)]
    pub session_name: String,
    pub summary: String,
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

        let state = if path.exists() {
            let raw = fs::read_to_string(&path).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to read session file {}: {err}",
                    path.display()
                ))
            })?;
            serde_json::from_str::<SessionState>(&raw).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to parse session file {}: {err}",
                    path.display()
                ))
            })?
        } else {
            SessionState {
                session_id: Uuid::new_v4().to_string(),
                session_name: String::new(),
                summary: String::new(),
                messages: Vec::new(),
                compass: new_compass(),
            }
        };

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
        store.repair_compass();
        store.enforce_max_limit();
        store.persist_active_session_pointer()?;
        Ok(store)
    }

    pub fn add_user_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("user", MessageKind::User, content, group_id);
    }

    pub fn add_assistant_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("assistant", MessageKind::Assistant, content, group_id);
    }

    pub fn add_tool_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("tool", MessageKind::Tool, content, group_id);
    }

    pub fn add_system_message(&mut self, content: String, group_id: Option<String>) {
        self.add_message("system", MessageKind::System, content, group_id);
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
            let role = match message.role.as_str() {
                "assistant" => "assistant",
                "system" => "system",
                _ => "user",
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
        fs::write(&self.path, raw).map_err(|err| {
            AppError::Runtime(format!(
                "failed to write session file {}: {err}",
                self.path.display()
            ))
        })
    }

    pub fn session_file(path: &Path) -> PathBuf {
        let base_dir = path.join(".machineclaw");
        let active_file = base_dir.join("active_session");
        if let Ok(raw) = fs::read_to_string(&active_file) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                let candidate = PathBuf::from(trimmed);
                if candidate.exists() {
                    return candidate;
                }
                let relative_candidate = base_dir.join(trimmed);
                if relative_candidate.exists() {
                    return relative_candidate;
                }
            }
        }
        base_dir.join("session.json")
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

    pub fn count_by_role(&self, role: &str) -> usize {
        self.state
            .messages
            .iter()
            .filter(|msg| msg.role == role)
            .count()
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
        let parent = self.path.parent().ok_or_else(|| {
            AppError::Runtime("failed to resolve session parent directory".to_string())
        })?;
        let filename = format!("session-{new_session_id}.json");
        self.path = parent.join(filename);
        self.persist()?;
        self.persist_active_session_pointer()
    }

    pub fn rename_current_session(&mut self, new_name: &str) -> Result<(), AppError> {
        let normalized = normalize_session_name(&self.state.session_id, Some(new_name));
        self.state.session_name = normalized;
        self.state.compass.last_updated_epoch_ms = now_epoch_ms();
        self.persist()
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
    ) {
        self.update_compass_on_append(role, &content);
        self.state.messages.push(SessionMessage {
            role: role.to_string(),
            content,
            kind,
            group_id,
            created_at_epoch_ms: now_epoch_ms(),
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
        self.state.session_name =
            normalize_session_name(&self.state.session_id, Some(&self.state.session_name));
        if self.state.compass.created_at_epoch_ms == 0 {
            self.state.compass.created_at_epoch_ms = now;
        }
        if self.state.compass.last_updated_epoch_ms == 0 {
            self.state.compass.last_updated_epoch_ms = now;
        }
    }

    fn persist_active_session_pointer(&self) -> Result<(), AppError> {
        let Some(parent) = self.path.parent() else {
            return Ok(());
        };
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create session directory {}: {err}",
                parent.display()
            ))
        })?;
        let pointer_file = parent.join("active_session");
        fs::write(&pointer_file, self.path.to_string_lossy().to_string()).map_err(|err| {
            AppError::Runtime(format!(
                "failed to write active session pointer {}: {err}",
                pointer_file.display()
            ))
        })
    }
}

fn compute_keep_recent_messages(max_history_messages: usize) -> usize {
    let compress_recent = max_history_messages / 2;
    max_history_messages.saturating_sub(compress_recent).max(1)
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
