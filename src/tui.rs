use std::{
    borrow::Cow,
    collections::{BTreeMap, HashSet, VecDeque},
    env,
    fs,
    io::{self, IsTerminal, Stdout, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use chrono::{Local, TimeZone};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use dialoguer::Editor;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Sparkline,
        Table, TableState, Wrap,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use toml_edit::{DocumentMut, Item, Table as TomlTable, Value};
use unicode_width::UnicodeWidthChar;
use uuid::Uuid;
use wait_timeout::ChildExt;

use crate::{
    actions::{ActionOutcome, ActionServices, build_chat_system_prompt},
    ai::{
        ChatRoundEvent, ChatStreamEvent, ChatStreamEventKind, ChatToolResponse, ToolCallRequest,
        ToolUsePolicy,
    },
    cli::InspectTarget,
    config::{AppConfig, McpServerConfig, expand_tilde},
    context::{SessionMessage, SessionOverview, ToolExecutionMeta},
    error::{AppError, ExitCode},
    i18n, mask,
    mcp::{self, McpServerRecord},
    platform::OsType,
    render,
    shell::{CommandMode, CommandResult, CommandSpec, looks_like_write_command_hint},
};

const CHAT_RENDER_LIMIT: usize = 260;
const DEFAULT_HISTORY_LIMIT: usize = 20;
const UI_PREFS_FILE_NAME: &str = "ui-preferences.json";
const INSPECT_REFRESH_INTERVAL_MS: u128 = 900;
const INSPECT_DETAIL_REFRESH_INTERVAL_MS: u128 = 12_000;
const INSPECT_HISTORY_MAX: usize = 80;
const NAV_ITEMS_COUNT: usize = 5;
const CONVERSATION_TAIL_PADDING_LINES: usize = 1;
const THREAD_DOUBLE_CLICK_WINDOW_MS: u128 = 320;
const STARTUP_SPLASH_MIN_DURATION_MS: u64 = 280;
const STARTUP_SPLASH_FRAME_INTERVAL_MS: u64 = 56;
const STREAM_RENDER_BATCH_CHARS: usize = 24;
const THINKING_STREAM_PERSIST_INTERVAL_MS: u64 = 500;
const TOOL_RESULT_MODAL_MAX_RESULT_CHARS: usize = 120_000;
const MCP_TOOL_RESULT_CONTENT_MAX_CHARS: usize = 120_000;
const SESSION_AUTO_TITLE_MAX_UNITS: usize = 15;
const SESSION_AUTO_TITLE_SOURCE_MAX_CHARS: usize = 1200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiRole {
    User,
    Assistant,
    Thinking,
    System,
    Tool,
}

#[derive(Debug, Clone)]
struct UiMessage {
    role: UiRole,
    text: String,
    tool_meta: Option<ToolExecutionMeta>,
}

#[derive(Debug, Clone, Default)]
struct SessionConversationCache {
    messages: Vec<UiMessage>,
    message_persisted: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversationLineKind {
    Header,
    BubbleBorder,
    Body,
    Spacer,
}

#[derive(Debug, Clone)]
struct ConversationLine {
    kind: ConversationLineKind,
    role: UiRole,
    text: String,
    message_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct AiLiveState {
    started_at: Instant,
    tool_calls: usize,
    last_tool_label: String,
    cancel_requested: bool,
}

enum PendingAiEvent {
    Round(ChatRoundEvent),
    Stream(ChatStreamEvent),
    ToolCall {
        request: ToolCallRequest,
        reply_tx: mpsc::Sender<String>,
    },
    Finished(Result<ChatToolResponse, String>),
}

struct PendingAiWaitOutcome {
    response: Result<ChatToolResponse, String>,
    thinking_rendered_in_ui: bool,
    last_round_content: String,
    saw_stream_events: bool,
}

#[derive(Debug, Clone, Copy)]
struct ConversationCopyButton {
    message_index: usize,
    rect: Rect,
}

#[derive(Debug, Clone, Copy)]
struct ConversationDeleteButton {
    message_index: usize,
    rect: Rect,
}

#[derive(Debug, Clone, Copy)]
struct ConversationToolResultButton {
    message_index: usize,
    rect: Rect,
}

#[derive(Debug, Clone, Copy)]
struct ConversationHoverButtons {
    message_index: usize,
    copy_rect: Rect,
    result_rect: Option<Rect>,
    delete_rect: Rect,
}

#[derive(Debug, Clone, Copy)]
struct DeleteMessageConfirmState {
    message_index: usize,
    selected: usize,
}

#[derive(Debug, Clone, Copy)]
struct ThreadClickState {
    thread_index: usize,
    clicked_at_epoch_ms: u128,
}

#[derive(Debug, Clone, Copy)]
struct ThreadActionMenuState {
    thread_index: usize,
    selected: usize,
}

#[derive(Debug, Clone)]
struct ThreadRenameModalState {
    thread_index: usize,
    input: InputBuffer,
}

#[derive(Debug, Clone, Copy)]
struct ThreadDeleteConfirmState {
    thread_index: usize,
    selected: usize,
}

#[derive(Debug, Clone)]
struct ThreadMetadataModalState {
    session_id: String,
    session_name: String,
    rows: Vec<(String, String)>,
    scroll: usize,
}

#[derive(Debug, Clone)]
struct ToolResultModalState {
    message_index: usize,
    lines: Vec<String>,
    scroll: u16,
}

#[derive(Debug, Clone)]
struct ToolResultDetail {
    tool_call_id: String,
    function_name: String,
    command: String,
    arguments: String,
    result_payload: String,
    executed_at_epoch_ms: u128,
    account: String,
    environment: String,
    os_name: String,
    cwd: String,
    mode: String,
    label: String,
    exit_code: Option<i32>,
    duration_ms: u128,
    timed_out: bool,
    interrupted: bool,
    blocked: bool,
}

struct ToolExecutionMetaInput<'a> {
    tool_call_id: &'a str,
    function_name: &'a str,
    command: &'a str,
    arguments: &'a str,
    result_payload: &'a str,
    mode: &'a str,
    label: &'a str,
    exit_code: Option<i32>,
    duration_ms: u128,
    timed_out: bool,
    interrupted: bool,
    blocked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigFieldKind {
    Bool,
    String,
    OptionalString,
    U8,
    U32,
    U64,
    Usize,
    F64,
    StringList,
    StringMap,
    OptionalU64,
    Enum,
}

#[derive(Debug, Clone)]
struct ConfigField {
    key: String,
    label: String,
    category: String,
    kind: ConfigFieldKind,
    value: String,
    required: bool,
    options: Vec<String>,
    dirty: bool,
}

#[derive(Debug, Clone)]
struct ConfigCategory {
    id: String,
    label: String,
}

#[derive(Debug, Clone)]
struct ConfigUiState {
    categories: Vec<ConfigCategory>,
    selected_category: usize,
    selected_field_row: usize,
    editing: bool,
    edit_buffer: InputBuffer,
    fields: Vec<ConfigField>,
    dirty_count: usize,
    config_path: PathBuf,
}

struct ConfigFieldSeed {
    key: &'static str,
    label: &'static str,
    category: &'static str,
    kind: ConfigFieldKind,
    required: bool,
    options: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpFieldId {
    Name,
    Enabled,
    Transport,
    Url,
    Endpoint,
    Command,
    Args,
    Env,
    Headers,
    AuthType,
    AuthToken,
    TimeoutSeconds,
}

#[derive(Debug, Clone, Copy)]
struct McpFieldDef {
    id: McpFieldId,
    label: &'static str,
    kind: ConfigFieldKind,
    options: &'static [&'static str],
}

#[derive(Debug, Clone)]
struct McpUiServer {
    name: String,
    config: McpServerConfig,
    dirty: bool,
}

#[derive(Debug, Clone)]
struct McpUiState {
    servers: Vec<McpUiServer>,
    selected_server: usize,
    selected_field: usize,
    focus_servers: bool,
    editing: bool,
    edit_buffer: InputBuffer,
    dirty_count: usize,
    structural_dirty: bool,
    config_file_path: PathBuf,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusPanel {
    Nav,
    Threads,
    Conversation,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiMode {
    Chat,
    Skills,
    Mcp,
    Inspect,
    Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfirmMode {
    Deny,
    Edit,
    AllowOnce,
    AllowSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteDecision {
    Reject,
    Approve,
    ApproveSession,
    Edit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UiPreferences {
    theme: String,
}

impl Default for UiPreferences {
    fn default() -> Self {
        Self {
            theme: "graphite".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ThemePalette {
    name: &'static str,
    app_bg: Color,
    sidebar_bg: Color,
    panel_bg: Color,
    border: Color,
    border_focus: Color,
    text: Color,
    muted: Color,
    accent: Color,
    status: Color,
}

#[derive(Debug, Clone)]
struct InputBuffer {
    text: String,
    cursor_char: usize,
    view_char_offset: usize,
}

#[derive(Debug, Clone)]
struct PendingChoice {
    options: Vec<String>,
    selected: usize,
}

#[derive(Debug, Clone)]
struct InspectMetrics {
    usage_percent: f64,
    user_percent: f64,
    system_percent: f64,
    idle_percent: f64,
    model: String,
    logical_cores: String,
    physical_cores: String,
    freq_mhz: String,
    temperature: String,
    health: String,
    last_updated: String,
    raw_output: String,
}

impl Default for InspectMetrics {
    fn default() -> Self {
        Self {
            usage_percent: 0.0,
            user_percent: 0.0,
            system_percent: 0.0,
            idle_percent: 100.0,
            model: ui_text_na().to_string(),
            logical_cores: ui_text_na().to_string(),
            physical_cores: ui_text_na().to_string(),
            freq_mhz: ui_text_na().to_string(),
            temperature: ui_text_na().to_string(),
            health: ui_text_inspect_health_unknown().to_string(),
            last_updated: ui_text_na().to_string(),
            raw_output: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct InspectState {
    target: InspectTarget,
    metrics: InspectMetrics,
    usage_history: VecDeque<u64>,
}

#[derive(Debug, Clone)]
struct SkillPanelRow {
    name: String,
    summary: String,
    path: String,
}

#[derive(Debug, Clone)]
struct SkillDocModalState {
    skill_name: String,
    file_path: PathBuf,
    raw_content: String,
    rendered_content: String,
    scroll: u16,
}

#[derive(Debug, Clone, Copy)]
enum InspectWorkerCommand {
    SetTarget(InspectTarget),
    SetEnabled(bool),
    Stop,
}

#[derive(Debug, Clone)]
struct InspectWorkerUpdate {
    target: InspectTarget,
    metrics: InspectMetrics,
}

struct InspectWorkerHandle {
    control_tx: mpsc::Sender<InspectWorkerCommand>,
    update_rx: mpsc::Receiver<InspectWorkerUpdate>,
    join: Option<thread::JoinHandle<()>>,
}

struct AiConnectivityCheckHandle {
    rx: mpsc::Receiver<Result<(), AppError>>,
}

#[derive(Debug)]
struct SessionAutoTitleHandle {
    session_id: String,
    rx: mpsc::Receiver<Result<String, AppError>>,
}

impl InputBuffer {
    fn new() -> Self {
        Self {
            text: String::new(),
            cursor_char: 0,
            view_char_offset: 0,
        }
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor_char = 0;
        self.view_char_offset = 0;
    }

    fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    fn insert_char(&mut self, ch: char) {
        let idx = char_to_byte_idx(&self.text, self.cursor_char);
        self.text.insert(idx, ch);
        self.cursor_char = self.cursor_char.saturating_add(1);
    }

    fn backspace(&mut self) {
        if self.cursor_char == 0 {
            return;
        }
        let prev = self.cursor_char - 1;
        let start = char_to_byte_idx(&self.text, prev);
        let end = char_to_byte_idx(&self.text, self.cursor_char);
        self.text.replace_range(start..end, "");
        self.cursor_char = prev;
    }

    fn delete(&mut self) {
        if self.cursor_char >= self.char_count() {
            return;
        }
        let start = char_to_byte_idx(&self.text, self.cursor_char);
        let end = char_to_byte_idx(&self.text, self.cursor_char + 1);
        self.text.replace_range(start..end, "");
    }

    fn move_left(&mut self) {
        self.cursor_char = self.cursor_char.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor_char = (self.cursor_char + 1).min(self.char_count());
    }

    fn move_home(&mut self) {
        self.cursor_char = 0;
    }

    fn move_end(&mut self) {
        self.cursor_char = self.char_count();
    }
}

#[derive(Debug, Clone, Copy)]
struct UiLayout {
    sidebar: Rect,
    nav: Rect,
    threads: Rect,
    threads_body: Rect,
    footer: Rect,
    header: Rect,
    conversation: Rect,
    conversation_body: Rect,
    input: Rect,
    input_body: Rect,
    status: Rect,
}

#[derive(Debug)]
struct ChatUiState {
    mode: UiMode,
    focus: FocusPanel,
    nav_selected: usize,
    inspect_menu_open: bool,
    inspect_menu_selected: usize,
    inspect: InspectState,
    skills_selected_row: usize,
    skill_doc_modal: Option<SkillDocModalState>,
    status: String,
    messages: VecDeque<UiMessage>,
    message_persisted: VecDeque<bool>,
    current_session_id: String,
    session_conversation_cache: BTreeMap<String, SessionConversationCache>,
    conversation_lines: Vec<ConversationLine>,
    conversation_line_count: usize,
    conversation_wrap_width: u16,
    conversation_dirty: bool,
    conversation_scroll: u16,
    follow_tail: bool,
    hovered_message_idx: Option<usize>,
    pending_delete_confirm: Option<DeleteMessageConfirmState>,
    pending_thread_action_menu: Option<ThreadActionMenuState>,
    pending_thread_rename: Option<ThreadRenameModalState>,
    pending_thread_delete_confirm: Option<ThreadDeleteConfirmState>,
    pending_thread_metadata_modal: Option<ThreadMetadataModalState>,
    pending_tool_result_modal: Option<ToolResultModalState>,
    last_thread_click: Option<ThreadClickState>,
    ai_live: Option<AiLiveState>,
    config_ui: ConfigUiState,
    mcp_ui: McpUiState,
    threads: Vec<SessionOverview>,
    thread_selected: usize,
    input: InputBuffer,
    theme_idx: usize,
    prefs_path: PathBuf,
    write_session_approved: bool,
    pending_choice: Option<PendingChoice>,
    token_usage_committed: u64,
    token_live_estimate: u64,
    token_display_value: u64,
    ai_connectivity_checking: bool,
    session_auto_title_workers: Vec<SessionAutoTitleHandle>,
    session_auto_title_attempted: HashSet<String>,
}

impl ChatUiState {
    fn new(
        messages: Vec<UiMessage>,
        threads: Vec<SessionOverview>,
        active_session_id: String,
        theme_idx: usize,
        prefs_path: PathBuf,
        cfg: &AppConfig,
        config_path: &Path,
    ) -> Self {
        let initial_persisted = std::iter::repeat_n(true, messages.len()).collect::<Vec<_>>();
        let mut session_conversation_cache = BTreeMap::new();
        session_conversation_cache.insert(
            active_session_id.clone(),
            SessionConversationCache {
                messages: messages.clone(),
                message_persisted: initial_persisted.clone(),
            },
        );
        let mut state = Self {
            mode: UiMode::Chat,
            focus: FocusPanel::Input,
            nav_selected: 0,
            inspect_menu_open: false,
            inspect_menu_selected: 0,
            inspect: InspectState {
                target: InspectTarget::Cpu,
                metrics: InspectMetrics::default(),
                usage_history: VecDeque::new(),
            },
            skills_selected_row: 0,
            skill_doc_modal: None,
            status: ui_text_ready().to_string(),
            message_persisted: initial_persisted.into(),
            messages: messages.into(),
            current_session_id: active_session_id,
            session_conversation_cache,
            conversation_lines: Vec::new(),
            conversation_line_count: 0,
            conversation_wrap_width: 0,
            conversation_dirty: true,
            conversation_scroll: 0,
            follow_tail: true,
            hovered_message_idx: None,
            pending_delete_confirm: None,
            pending_thread_action_menu: None,
            pending_thread_rename: None,
            pending_thread_delete_confirm: None,
            pending_thread_metadata_modal: None,
            pending_tool_result_modal: None,
            last_thread_click: None,
            ai_live: None,
            config_ui: build_config_ui_state(cfg, config_path),
            mcp_ui: build_mcp_ui_state(cfg, config_path),
            threads,
            thread_selected: 0,
            input: InputBuffer::new(),
            theme_idx,
            prefs_path,
            write_session_approved: false,
            pending_choice: None,
            token_usage_committed: 0,
            token_live_estimate: 0,
            token_display_value: 0,
            ai_connectivity_checking: false,
            session_auto_title_workers: Vec::new(),
            session_auto_title_attempted: HashSet::new(),
        };
        state.sync_thread_selection_to_active();
        state
    }

    fn push(&mut self, role: UiRole, text: impl Into<String>) {
        self.push_with_persisted(role, text, false);
    }

    fn push_persisted(&mut self, role: UiRole, text: impl Into<String>) {
        self.push_with_persisted(role, text, true);
    }

    fn push_with_persisted(&mut self, role: UiRole, text: impl Into<String>, persisted: bool) {
        self.push_message_with_persisted(
            UiMessage {
                role,
                text: text.into(),
                tool_meta: None,
            },
            persisted,
        );
    }

    fn push_message_with_persisted(&mut self, message: UiMessage, persisted: bool) {
        self.messages.push_back(message);
        self.message_persisted.push_back(persisted);
        while self.messages.len() > CHAT_RENDER_LIMIT {
            self.messages.pop_front();
            self.message_persisted.pop_front();
        }
        self.conversation_dirty = true;
        self.follow_tail = true;
        self.hovered_message_idx = None;
        self.pending_delete_confirm = None;
        self.pending_thread_delete_confirm = None;
    }

    fn push_stream_chunk(&mut self, role: UiRole, chunk: &str, prefix: Option<&str>) {
        if chunk.is_empty() {
            return;
        }
        let can_append = self.messages.back().is_some_and(|last| {
            if last.role != role {
                return false;
            }
            if let Some(prefix_text) = prefix {
                last.text.starts_with(prefix_text)
            } else {
                true
            }
        });
        if can_append {
            if let Some(last) = self.messages.back_mut() {
                last.text.push_str(chunk);
            }
        } else {
            let mut text = String::new();
            if let Some(prefix_text) = prefix {
                text.push_str(prefix_text);
            }
            text.push_str(chunk);
            self.messages.push_back(UiMessage {
                role,
                text,
                tool_meta: None,
            });
            self.message_persisted.push_back(false);
            while self.messages.len() > CHAT_RENDER_LIMIT {
                self.messages.pop_front();
                self.message_persisted.pop_front();
            }
        }
        self.conversation_dirty = true;
        self.follow_tail = true;
        self.hovered_message_idx = None;
        self.pending_delete_confirm = None;
        self.pending_thread_delete_confirm = None;
    }

    fn token_usage_target(&self) -> u64 {
        self.token_usage_committed
            .saturating_add(self.token_live_estimate)
    }

    fn add_live_token_estimate(&mut self, delta: u64) {
        if delta == 0 {
            return;
        }
        self.token_live_estimate = self.token_live_estimate.saturating_add(delta);
    }

    fn reset_live_token_estimate(&mut self) {
        self.token_live_estimate = 0;
    }

    fn commit_token_usage(&mut self, total_tokens: u64) {
        self.token_usage_committed = self.token_usage_committed.saturating_add(total_tokens);
        self.token_live_estimate = 0;
    }

    fn tick_token_display_animation(&mut self) {
        let target = self.token_usage_target();
        if self.token_display_value == target {
            return;
        }
        if self.token_display_value < target {
            let gap = target - self.token_display_value;
            let step = gap.div_ceil(7).clamp(1, 256);
            self.token_display_value = self.token_display_value.saturating_add(step).min(target);
        } else {
            let gap = self.token_display_value - target;
            let step = gap.div_ceil(5).clamp(1, 384);
            self.token_display_value = self.token_display_value.saturating_sub(step).max(target);
        }
    }

    fn clear_conversation_viewport_only(&mut self) {
        self.messages.clear();
        self.message_persisted.clear();
        self.conversation_dirty = true;
        self.conversation_scroll = 0;
        self.follow_tail = true;
        self.hovered_message_idx = None;
        self.pending_delete_confirm = None;
        self.pending_thread_delete_confirm = None;
        self.pending_tool_result_modal = None;
    }

    fn is_message_persisted(&self, index: usize) -> bool {
        self.message_persisted.get(index).copied().unwrap_or(false)
    }

    fn mark_last_message_persisted_if_matches(&mut self, role: UiRole, text: &str) {
        let Some(last_message) = self.messages.back() else {
            return;
        };
        if last_message.role != role || last_message.text.trim() != text.trim() {
            return;
        }
        if let Some(last) = self.message_persisted.back_mut() {
            *last = true;
        }
    }

    fn mark_all_unpersisted_messages_by_role(&mut self, role: UiRole) -> usize {
        let mut updated = 0usize;
        for idx in 0..self.messages.len() {
            if self.is_message_persisted(idx) {
                continue;
            }
            let Some(item) = self.messages.get(idx) else {
                continue;
            };
            if item.role != role {
                continue;
            }
            if let Some(flag) = self.message_persisted.get_mut(idx) {
                *flag = true;
                updated = updated.saturating_add(1);
            }
        }
        updated
    }

    fn remove_message_at(&mut self, index: usize) -> bool {
        if index >= self.messages.len() {
            return false;
        }
        let _ = self.messages.remove(index);
        let _ = self.message_persisted.remove(index);
        self.conversation_dirty = true;
        self.follow_tail = true;
        if let Some(modal) = self.pending_tool_result_modal.as_ref()
            && modal.message_index == index
        {
            self.pending_tool_result_modal = None;
        }
        if let Some(hovered) = self.hovered_message_idx {
            self.hovered_message_idx = if hovered == index {
                None
            } else if hovered > index {
                Some(hovered - 1)
            } else {
                Some(hovered)
            };
        }
        if let Some(modal) = self.pending_tool_result_modal.as_mut()
            && modal.message_index > index
        {
            modal.message_index -= 1;
        }
        true
    }

    fn remember_current_session_messages(&mut self) {
        let cache = SessionConversationCache {
            messages: self.messages.iter().cloned().collect(),
            message_persisted: self.message_persisted.iter().copied().collect(),
        };
        self.session_conversation_cache
            .insert(self.current_session_id.clone(), cache);
    }

    fn set_active_session(&mut self, session_id: String) {
        self.current_session_id = session_id;
    }

    fn session_messages_or_fallback(
        &self,
        session_id: &str,
        fallback: Vec<UiMessage>,
    ) -> Vec<(UiMessage, bool)> {
        if let Some(cache) = self.session_conversation_cache.get(session_id) {
            let mut persisted = cache.message_persisted.clone();
            if persisted.len() < cache.messages.len() {
                persisted.resize(cache.messages.len(), true);
            } else if persisted.len() > cache.messages.len() {
                persisted.truncate(cache.messages.len());
            }
            return cache.messages.iter().cloned().zip(persisted).collect();
        }
        fallback.into_iter().map(|item| (item, true)).collect()
    }

    fn drop_session_cache(&mut self, session_id: &str) {
        self.session_conversation_cache.remove(session_id);
    }

    fn prune_session_cache_for_known_threads(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        let mut keep = HashSet::<String>::new();
        keep.insert(self.current_session_id.clone());
        for item in &self.threads {
            keep.insert(item.session_id.clone());
        }
        self.session_conversation_cache
            .retain(|session_id, _| keep.contains(session_id));
    }

    fn prune_session_auto_title_tracking_for_known_threads(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        let mut keep = HashSet::<String>::new();
        keep.insert(self.current_session_id.clone());
        for item in &self.threads {
            keep.insert(item.session_id.clone());
        }
        self.session_auto_title_attempted
            .retain(|session_id| keep.contains(session_id));
        self.session_auto_title_workers
            .retain(|worker| keep.contains(&worker.session_id));
    }

    fn set_threads(&mut self, threads: Vec<SessionOverview>) {
        if self.threads.is_empty() {
            self.threads = threads;
            self.prune_session_cache_for_known_threads();
            self.prune_session_auto_title_tracking_for_known_threads();
            self.sync_thread_selection_to_active();
            return;
        }
        let selected_id = self
            .threads
            .get(self.thread_selected)
            .map(|item| item.session_id.clone());
        let mut incoming = threads;
        let mut ordered = Vec::with_capacity(incoming.len());
        for item in &self.threads {
            if let Some(idx) = incoming
                .iter()
                .position(|entry| entry.session_id == item.session_id)
            {
                ordered.push(incoming.remove(idx));
            }
        }
        ordered.extend(incoming);
        self.threads = ordered;
        self.prune_session_cache_for_known_threads();
        self.prune_session_auto_title_tracking_for_known_threads();
        if self.threads.is_empty() {
            self.thread_selected = 0;
            return;
        }
        if let Some(id) = selected_id
            && let Some((idx, _)) = self
                .threads
                .iter()
                .enumerate()
                .find(|(_, item)| item.session_id == id)
        {
            self.thread_selected = idx;
            return;
        }
        self.thread_selected = self
            .thread_selected
            .min(self.threads.len().saturating_sub(1));
    }

    fn sync_thread_selection_to_active(&mut self) {
        if self.threads.is_empty() {
            self.thread_selected = 0;
            return;
        }
        if let Some((idx, _)) = self
            .threads
            .iter()
            .enumerate()
            .find(|(_, item)| item.active)
        {
            self.thread_selected = idx;
            return;
        }
        self.thread_selected = self.thread_selected.min(self.threads.len() - 1);
    }

    fn ensure_conversation_cache(&mut self) {
        if !self.conversation_dirty {
            return;
        }
        let bubble_max_inner_width =
            conversation_bubble_max_inner_width(self.conversation_wrap_width);
        let conversation_width = self.conversation_wrap_width.max(1) as usize;
        let live_streaming = self.ai_live.is_some();
        let last_message_idx = self.messages.len().saturating_sub(1);
        let mut lines = Vec::<ConversationLine>::new();
        for (idx, item) in self.messages.iter().enumerate() {
            let header_label = role_tag(item.role).to_string();
            let header_label = trim_ui_text(header_label.as_str(), conversation_width.max(1));
            let header_width = text_display_width(header_label.as_str());
            let rendered = render_conversation_item_text(
                item,
                live_streaming
                    && idx == last_message_idx
                    && matches!(item.role, UiRole::Assistant | UiRole::Thinking),
            );
            let source_lines = if rendered.trim().is_empty() {
                vec![String::new()]
            } else {
                rendered
                    .lines()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            };
            let mut body_lines = Vec::<String>::new();
            for line in source_lines {
                body_lines.extend(wrap_text_by_display_width(
                    normalize_conversation_line(line.as_str()).as_str(),
                    bubble_max_inner_width,
                ));
            }
            let inner_width = body_lines
                .iter()
                .map(|line| text_display_width(line))
                .max()
                .unwrap_or(1)
                .max(1);
            let bubble_outer_width = inner_width.saturating_add(4).max(header_width);
            let header_text = if item.role == UiRole::User {
                let left_pad = bubble_outer_width.saturating_sub(header_width);
                format!("{}{}", " ".repeat(left_pad), header_label)
            } else {
                format!(
                    "{}{}",
                    header_label,
                    " ".repeat(conversation_width.saturating_sub(header_width))
                )
            };
            let bubble_left_pad = if item.role == UiRole::User {
                conversation_width.saturating_sub(bubble_outer_width)
            } else {
                0
            };
            let apply_left_pad = |text: String| {
                if bubble_left_pad == 0 {
                    text
                } else {
                    format!("{}{}", " ".repeat(bubble_left_pad), text)
                }
            };
            lines.push(ConversationLine {
                kind: ConversationLineKind::Header,
                role: item.role,
                text: apply_left_pad(header_text),
                message_index: Some(idx),
            });
            lines.push(ConversationLine {
                kind: ConversationLineKind::BubbleBorder,
                role: item.role,
                text: apply_left_pad(format!("╭{}╮", "─".repeat(inner_width + 2))),
                message_index: Some(idx),
            });
            for body_line in body_lines {
                let pad = inner_width.saturating_sub(text_display_width(body_line.as_str()));
                lines.push(ConversationLine {
                    kind: ConversationLineKind::Body,
                    role: item.role,
                    text: apply_left_pad(format!("│ {}{} │", body_line, " ".repeat(pad))),
                    message_index: Some(idx),
                });
            }
            lines.push(ConversationLine {
                kind: ConversationLineKind::BubbleBorder,
                role: item.role,
                text: apply_left_pad(format!("╰{}╯", "─".repeat(inner_width + 2))),
                message_index: Some(idx),
            });
            if idx + 1 < self.messages.len() {
                lines.push(ConversationLine {
                    kind: ConversationLineKind::Spacer,
                    role: item.role,
                    text: String::new(),
                    message_index: None,
                });
            }
        }
        if lines.is_empty() {
            lines.push(ConversationLine {
                kind: ConversationLineKind::Body,
                role: UiRole::System,
                text: ui_text_no_messages().to_string(),
                message_index: None,
            });
        }
        self.conversation_lines = lines;
        self.recalculate_conversation_line_count();
        self.conversation_dirty = false;
    }

    fn recalculate_conversation_line_count(&mut self) {
        self.conversation_line_count = self.conversation_lines.len().max(1);
    }

    fn set_conversation_wrap_width(&mut self, wrap_width: u16) {
        if self.conversation_wrap_width == wrap_width {
            return;
        }
        self.conversation_wrap_width = wrap_width;
        self.conversation_dirty = true;
    }

    fn clamp_scroll(&mut self, viewport_height: u16) {
        let visible = viewport_height.max(1) as usize;
        let max = self
            .conversation_line_count
            .saturating_sub(visible)
            .min(u16::MAX as usize) as u16;
        if self.follow_tail {
            let tail_max = self
                .conversation_line_count
                .saturating_add(CONVERSATION_TAIL_PADDING_LINES)
                .saturating_sub(visible)
                .min(u16::MAX as usize) as u16;
            self.conversation_scroll = tail_max.max(max);
            return;
        }
        if self.conversation_scroll > max {
            self.conversation_scroll = max;
        }
    }

    fn scroll_by(&mut self, delta: i16, viewport_height: u16) {
        self.ensure_conversation_cache();
        self.clamp_scroll(viewport_height);
        let visible = viewport_height.max(1) as usize;
        let max = self
            .conversation_line_count
            .saturating_sub(visible)
            .min(u16::MAX as usize) as i32;
        let mut next = self.conversation_scroll as i32 + delta as i32;
        if next < 0 {
            next = 0;
        }
        if next > max {
            next = max;
        }
        self.conversation_scroll = next as u16;
        self.follow_tail = self.conversation_scroll as i32 >= max;
    }

    fn cycle_focus_forward(&mut self) {
        if self.inspect_menu_open {
            return;
        }
        self.focus = match self.focus {
            FocusPanel::Nav => FocusPanel::Threads,
            FocusPanel::Threads => FocusPanel::Conversation,
            FocusPanel::Conversation => {
                if self.mode == UiMode::Chat
                    || self.mode == UiMode::Config
                    || self.mode == UiMode::Mcp
                {
                    FocusPanel::Input
                } else {
                    FocusPanel::Nav
                }
            }
            FocusPanel::Input => FocusPanel::Nav,
        };
    }

    fn cycle_focus_backward(&mut self) {
        if self.inspect_menu_open {
            return;
        }
        self.focus = match self.focus {
            FocusPanel::Nav => {
                if self.mode == UiMode::Chat
                    || self.mode == UiMode::Config
                    || self.mode == UiMode::Mcp
                {
                    FocusPanel::Input
                } else {
                    FocusPanel::Conversation
                }
            }
            FocusPanel::Threads => FocusPanel::Nav,
            FocusPanel::Conversation => FocusPanel::Threads,
            FocusPanel::Input => FocusPanel::Conversation,
        };
    }
}

enum BuiltinCommand {
    Exit,
    Help,
    Stats,
    Meta,
    Skills,
    Mcps,
    New,
    Clear,
    List,
    Change(String),
    Name(String),
    History(usize),
}

#[derive(Debug, Deserialize)]
struct ShellToolArgs {
    #[serde(default)]
    label: Option<String>,
    command: String,
    #[serde(default)]
    mode: Option<String>,
}

pub fn run_chat_tui(services: &mut ActionServices<'_>) -> Result<ActionOutcome, AppError> {
    let started = Instant::now();
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(AppError::Command(i18n::chat_requires_interactive_terminal()));
    }
    let (theme_idx, prefs_path) = load_theme_preferences()?;
    let base_system_prompt = render::load_prompt_template(services.assets_dir, "chat_system.md")?;
    let system_prompt = build_chat_system_prompt(services, &base_system_prompt);
    let recent_messages = services
        .session
        .recent_messages_for_display(CHAT_RENDER_LIMIT);
    let initial_messages = recent_messages_to_ui_messages(&recent_messages);
    let initial_threads = services.session.list_sessions().unwrap_or_default();
    let mut state = ChatUiState::new(
        initial_messages,
        initial_threads,
        services.session.session_id().to_string(),
        theme_idx,
        prefs_path,
        services.cfg,
        services.config_path,
    );
    state.push(
        UiRole::System,
        format!(
            "{}: {} | {}: {}",
            ui_text_tui_ready(),
            services.session.session_name(),
            ui_text_model_label(),
            services.cfg.ai.model
        ),
    );

    let mut terminal = init_terminal()?;
    if let Err(err) = draw_startup_splash(&mut terminal, theme_idx) {
        let _ = restore_terminal(&mut terminal);
        return Err(err);
    }
    let mut chat_turns = 0usize;
    let mut last_assistant_reply = String::new();
    let mut inspect_worker = spawn_inspect_worker(services.os_type);
    let mut ai_connectivity_check =
        match run_startup_checks_in_tui(&mut terminal, services, &mut state) {
            Ok(handle) => handle,
            Err(err) => {
                inspect_worker.stop();
                let _ = restore_terminal(&mut terminal);
                return Err(err);
            }
        };
    let loop_result = run_loop(
        &mut terminal,
        services,
        &system_prompt,
        &mut state,
        &mut chat_turns,
        &mut last_assistant_reply,
        &mut inspect_worker,
        &mut ai_connectivity_check,
    );
    inspect_worker.stop();
    let restore_result = restore_terminal(&mut terminal);
    restore_result?;
    loop_result?;
    let ai_summary = if last_assistant_reply.trim().is_empty() {
        i18n::chat_goodbye().to_string()
    } else {
        last_assistant_reply
    };
    let rendered = render::render_action(
        services.assets_dir,
        "chat",
        &render::ActionRenderData {
            action: "chat".to_string(),
            status: i18n::status_success().to_string(),
            key_metrics: format!(
                "session_id={}\nchat_turns={}\nmessages={}\nsummary_chars={}\nui=ratatui\ntheme={}",
                services.session.session_id(),
                chat_turns,
                services.session.message_count(),
                services.session.summary_len(),
                palette_by_index(state.theme_idx).name
            ),
            risk_summary: i18n::risk_no_obvious().to_string(),
            ai_summary,
            command_summary: "ui_mode=tui\ninteraction=structured".to_string(),
            elapsed: i18n::human_duration_ms(started.elapsed().as_millis()),
        },
        services.cfg.console.colorful,
    )?;
    Ok(ActionOutcome {
        rendered,
        exit_code: ExitCode::Success,
    })
}

fn run_startup_checks_in_tui(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<Option<AiConnectivityCheckHandle>, AppError> {
    let started = Instant::now();
    state.push(UiRole::System, i18n::preflight_notice_start());
    state.push(UiRole::System, i18n::preflight_notice_config_check());
    state.push(UiRole::System, i18n::preflight_notice_permission_check());
    state.push(
        UiRole::System,
        i18n::preflight_notice_permission_check_skipped(),
    );
    let mut ai_connectivity_check = None;
    if services.cfg.ai.connectivity_check {
        state.ai_connectivity_checking = true;
        state.status = ui_text_status_ai_connectivity().to_string();
        state.push(
            UiRole::System,
            ui_text_status_ai_connectivity_background_started(),
        );
        ai_connectivity_check = Some(spawn_ai_connectivity_check_worker(services.ai.clone()));
    } else {
        state.push(UiRole::System, i18n::preflight_notice_ai_check_skipped());
    }
    refresh_mcp_runtime_metadata_if_needed(services, state, true);
    state.push(
        UiRole::System,
        i18n::preflight_notice_done(&i18n::human_duration_ms(started.elapsed().as_millis())),
    );
    if !state.ai_connectivity_checking {
        state.status = ui_text_ready().to_string();
    }
    draw_once(terminal, services, state)?;
    Ok(ai_connectivity_check)
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    system_prompt: &str,
    state: &mut ChatUiState,
    chat_turns: &mut usize,
    last_assistant_reply: &mut String,
    inspect_worker: &mut InspectWorkerHandle,
    ai_connectivity_check: &mut Option<AiConnectivityCheckHandle>,
) -> Result<(), AppError> {
    let mut worker_enabled = false;
    let mut worker_target = state.inspect.target;
    loop {
        sync_inspect_worker_state(
            inspect_worker,
            state,
            &mut worker_enabled,
            &mut worker_target,
        );
        drain_inspect_worker_updates(inspect_worker, state);
        poll_ai_connectivity_check(ai_connectivity_check, state);
        poll_session_auto_title_workers(services, state);
        draw_once(terminal, services, state)?;
        if !event::poll(Duration::from_millis(50))
            .map_err(|err| AppError::Command(format!("failed to poll input event: {err}")))?
        {
            drain_inspect_worker_updates(inspect_worker, state);
            poll_ai_connectivity_check(ai_connectivity_check, state);
            poll_session_auto_title_workers(services, state);
            continue;
        }
        match event::read()
            .map_err(|err| AppError::Command(format!("failed to read event: {err}")))?
        {
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if handle_key_event(
                    key,
                    terminal,
                    services,
                    system_prompt,
                    state,
                    chat_turns,
                    last_assistant_reply,
                )? {
                    break;
                }
            }
            Event::Mouse(mouse) => handle_mouse_event(mouse, terminal, services, state)?,
            Event::Resize(_, _) => {
                handle_tui_resize_redraw(terminal, services, state)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn spawn_inspect_worker(os_type: OsType) -> InspectWorkerHandle {
    let (control_tx, control_rx) = mpsc::channel::<InspectWorkerCommand>();
    let (update_tx, update_rx) = mpsc::channel::<InspectWorkerUpdate>();
    let join = thread::spawn(move || {
        let mut target = InspectTarget::Cpu;
        let mut enabled = false;
        let mut previous = InspectMetrics::default();
        let mut last_sample_epoch_ms = 0u128;
        let mut last_detail_epoch_ms = 0u128;
        loop {
            while let Ok(command) = control_rx.try_recv() {
                match command {
                    InspectWorkerCommand::SetTarget(next) => {
                        target = next;
                        previous = InspectMetrics::default();
                        last_sample_epoch_ms = 0;
                        last_detail_epoch_ms = 0;
                    }
                    InspectWorkerCommand::SetEnabled(flag) => {
                        enabled = flag;
                    }
                    InspectWorkerCommand::Stop => return,
                }
            }
            if !enabled {
                thread::sleep(Duration::from_millis(60));
                continue;
            }
            let now = now_epoch_ms();
            if now.saturating_sub(last_sample_epoch_ms) < INSPECT_REFRESH_INTERVAL_MS {
                thread::sleep(Duration::from_millis(45));
                continue;
            }
            last_sample_epoch_ms = now;
            let refresh_details = now.saturating_sub(last_detail_epoch_ms)
                >= INSPECT_DETAIL_REFRESH_INTERVAL_MS
                || previous.model == ui_text_na();
            let next_metrics = match target {
                InspectTarget::Cpu => {
                    collect_cpu_metrics_local(os_type, &previous, refresh_details)
                }
                item => collect_generic_inspect_metrics_local(os_type, item),
            };
            if refresh_details {
                last_detail_epoch_ms = now;
            }
            previous = next_metrics.clone();
            if update_tx
                .send(InspectWorkerUpdate {
                    target,
                    metrics: next_metrics,
                })
                .is_err()
            {
                return;
            }
        }
    });
    InspectWorkerHandle {
        control_tx,
        update_rx,
        join: Some(join),
    }
}

impl InspectWorkerHandle {
    fn notify(&self, command: InspectWorkerCommand) {
        let _ = self.control_tx.send(command);
    }

    fn stop(&mut self) {
        let _ = self.control_tx.send(InspectWorkerCommand::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn sync_inspect_worker_state(
    worker: &InspectWorkerHandle,
    state: &ChatUiState,
    worker_enabled: &mut bool,
    worker_target: &mut InspectTarget,
) {
    let enabled = state.mode == UiMode::Inspect;
    if enabled != *worker_enabled {
        worker.notify(InspectWorkerCommand::SetEnabled(enabled));
        *worker_enabled = enabled;
    }
    if enabled && state.inspect.target.as_str() != worker_target.as_str() {
        worker.notify(InspectWorkerCommand::SetTarget(state.inspect.target));
        *worker_target = state.inspect.target;
    }
}

fn drain_inspect_worker_updates(worker: &mut InspectWorkerHandle, state: &mut ChatUiState) {
    while let Ok(update) = worker.update_rx.try_recv() {
        if update.target.as_str() != state.inspect.target.as_str() {
            continue;
        }
        state.inspect.metrics = update.metrics;
        let usage = state.inspect.metrics.usage_percent.clamp(0.0, 100.0) as u64;
        state.inspect.usage_history.push_back(usage);
        while state.inspect.usage_history.len() > INSPECT_HISTORY_MAX {
            state.inspect.usage_history.pop_front();
        }
    }
}

fn draw_once(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    terminal
        .draw(|frame| draw_ui(frame, services, state))
        .map_err(|err| AppError::Command(format!("failed to draw tui frame: {err}")))?;
    Ok(())
}

fn draw_startup_splash(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    theme_idx: usize,
) -> Result<(), AppError> {
    let started = Instant::now();
    let palette = palette_by_index(theme_idx);
    let mut frame_tick = 0usize;
    while started.elapsed().as_millis() < STARTUP_SPLASH_MIN_DURATION_MS as u128 {
        terminal
            .draw(|frame| draw_startup_splash_frame(frame, palette, frame_tick))
            .map_err(|err| AppError::Command(format!("failed to draw startup splash: {err}")))?;
        frame_tick = frame_tick.saturating_add(1);
        thread::sleep(Duration::from_millis(STARTUP_SPLASH_FRAME_INTERVAL_MS));
    }
    Ok(())
}

fn draw_startup_splash_frame(frame: &mut Frame<'_>, palette: ThemePalette, tick: usize) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(palette.app_bg)),
        area,
    );
    let splash = centered_rect(area, 72, 13);
    let pulse = tick.is_multiple_of(2);
    let border_color = if pulse {
        palette.border_focus
    } else {
        palette.accent
    };
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(zh_or_en("启动中", "Launching"))
            .style(Style::default().bg(palette.panel_bg))
            .border_style(Style::default().fg(border_color)),
        splash,
    );
    let inner = splash.inner(ratatui::layout::Margin {
        horizontal: 2,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(inner);
    let spinner = spinner_frame(Instant::now() - Duration::from_millis((tick as u64) * 120));
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                ui_text_app_name(),
                Style::default()
                    .fg(palette.text)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!(
                    "{spinner} {}",
                    zh_or_en("正在加载 TUI 组件...", "Loading TUI components...")
                ),
                Style::default().fg(palette.muted),
            )),
        ]),
        rows[0],
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(if pulse {
                        palette.accent
                    } else {
                        palette.border_focus
                    })
                    .bg(palette.panel_bg),
            )
            .ratio(((tick % 10) as f64 + 1.0) / 10.0)
            .label(zh_or_en("准备中", "Preparing")),
        rows[2],
    );
    frame.render_widget(
        Paragraph::new(ui_text_startup_splash_hint()).style(Style::default().fg(palette.muted)),
        rows[3],
    );
}

fn autoresize_tui_terminal(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> Result<(), AppError> {
    terminal
        .autoresize()
        .map_err(|err| AppError::Command(format!("failed to autoresize tui terminal: {err}")))
}

fn clear_tui_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<(), AppError> {
    terminal
        .clear()
        .map_err(|err| AppError::Command(format!("failed to clear tui terminal: {err}")))
}

fn handle_tui_resize_redraw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    autoresize_tui_terminal(terminal)?;
    clear_tui_terminal(terminal)?;
    draw_once(terminal, services, state)
}

fn handle_tui_resize_only(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> Result<(), AppError> {
    autoresize_tui_terminal(terminal)?;
    clear_tui_terminal(terminal)
}

fn spawn_ai_connectivity_check_worker(ai: crate::ai::AiClient) -> AiConnectivityCheckHandle {
    let (tx, rx) = mpsc::channel::<Result<(), AppError>>();
    thread::spawn(move || {
        let _ = tx.send(ai.validate_connectivity());
    });
    AiConnectivityCheckHandle { rx }
}

fn poll_ai_connectivity_check(
    ai_connectivity_check: &mut Option<AiConnectivityCheckHandle>,
    state: &mut ChatUiState,
) {
    let Some(handle) = ai_connectivity_check.as_mut() else {
        return;
    };
    let check_result = match handle.rx.try_recv() {
        Ok(result) => Some(result),
        Err(mpsc::TryRecvError::Empty) => None,
        Err(mpsc::TryRecvError::Disconnected) => Some(Err(AppError::Runtime(
            "AI connectivity worker channel disconnected".to_string(),
        ))),
    };
    let Some(result) = check_result else {
        return;
    };
    state.ai_connectivity_checking = false;
    *ai_connectivity_check = None;
    match result {
        Ok(()) => {
            state.push(UiRole::System, ui_text_status_ai_connectivity_ok());
            if state.status == ui_text_status_ai_connectivity() {
                state.status = ui_text_ready().to_string();
            }
        }
        Err(err) => {
            state.push(
                UiRole::System,
                format!(
                    "{}: {}",
                    ui_text_status_ai_connectivity_failed(),
                    i18n::localize_error(&err)
                ),
            );
            state.status = ui_text_status_ai_connectivity_failed().to_string();
        }
    }
}

fn spawn_session_auto_title_worker(
    ai: crate::ai::AiClient,
    session_id: String,
    first_user_message: String,
    language: i18n::Language,
) -> SessionAutoTitleHandle {
    let (tx, rx) = mpsc::channel::<Result<String, AppError>>();
    thread::spawn(move || {
        let result = request_session_auto_title_from_ai(&ai, &first_user_message, language);
        let _ = tx.send(result);
    });
    SessionAutoTitleHandle { session_id, rx }
}

fn maybe_start_session_auto_title_worker(
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    first_user_message: &str,
) {
    let session_id = services.session.session_id().to_string();
    let session_name = services.session.session_name().to_string();
    let user_message_count = services.session.archived_role_counts().user;
    if !should_schedule_session_auto_title(
        &state.session_auto_title_attempted,
        &session_id,
        &session_name,
        user_message_count,
    ) {
        return;
    }
    state.session_auto_title_attempted.insert(session_id.clone());
    let language = i18n::resolve_language(services.cfg.app.language.as_deref());
    state
        .session_auto_title_workers
        .push(spawn_session_auto_title_worker(
            services.ai.clone(),
            session_id,
            first_user_message.to_string(),
            language,
        ));
}

fn poll_session_auto_title_workers(services: &mut ActionServices<'_>, state: &mut ChatUiState) {
    if state.session_auto_title_workers.is_empty() {
        return;
    }
    let mut pending = Vec::<SessionAutoTitleHandle>::new();
    for handle in std::mem::take(&mut state.session_auto_title_workers) {
        match handle.rx.try_recv() {
            Ok(Ok(title)) => {
                apply_session_auto_title(services, state, handle.session_id.as_str(), &title);
            }
            Ok(Err(_)) => {}
            Err(mpsc::TryRecvError::Empty) => pending.push(handle),
            Err(mpsc::TryRecvError::Disconnected) => {}
        }
    }
    state.session_auto_title_workers = pending;
}

fn apply_session_auto_title(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    session_id: &str,
    title: &str,
) {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return;
    }
    let default_name = default_session_name_for_id(session_id);
    let current_name = if services.session.session_id() == session_id {
        services.session.session_name().to_string()
    } else {
        match services.session.list_sessions() {
            Ok(items) => items
                .into_iter()
                .find(|item| item.session_id == session_id)
                .map(|item| item.session_name)
                .unwrap_or_default(),
            Err(_) => String::new(),
        }
    };
    if current_name != default_name {
        return;
    }
    if services
        .session
        .rename_session_by_id(session_id, trimmed)
        .is_err()
    {
        return;
    }
    if let Ok(threads) = services.session.list_sessions() {
        state.set_threads(threads);
    }
}

fn should_schedule_session_auto_title(
    attempted: &HashSet<String>,
    session_id: &str,
    session_name: &str,
    user_message_count: usize,
) -> bool {
    if session_id.trim().is_empty() {
        return false;
    }
    if attempted.contains(session_id) {
        return false;
    }
    if session_name != default_session_name_for_id(session_id) {
        return false;
    }
    user_message_count == 1
}

fn default_session_name_for_id(session_id: &str) -> String {
    format!("session-{session_id}")
}

fn request_session_auto_title_from_ai(
    ai: &crate::ai::AiClient,
    first_user_message: &str,
    language: i18n::Language,
) -> Result<String, AppError> {
    let clipped = trim_ui_text(first_user_message, SESSION_AUTO_TITLE_SOURCE_MAX_CHARS);
    let (system_prompt, user_prompt) = build_session_auto_title_prompts(language, clipped.as_str());
    let raw = ai.chat(&[], system_prompt, user_prompt.as_str())?;
    normalize_session_auto_title_candidate(raw.as_str(), SESSION_AUTO_TITLE_MAX_UNITS)
        .ok_or_else(|| AppError::Ai("AI returned invalid session title".to_string()))
}

fn build_session_auto_title_prompts(
    language: i18n::Language,
    first_user_message: &str,
) -> (&'static str, String) {
    match language {
        i18n::Language::ZhCn => (
            "你负责生成简洁的会话标题。",
            format!(
                "用户首条消息：\n{first_user_message}\n\n请生成一个能准确概括意图的会话标题。\n要求：\n1. 仅输出标题，不要解释。\n2. 必须使用简体中文。\n3. 不超过15个汉字。\n4. 优先使用具体主题词和动作词。"
            ),
        ),
        i18n::Language::ZhTw => (
            "你負責產生精簡的會話標題。",
            format!(
                "使用者首條訊息：\n{first_user_message}\n\n請產生一個能準確概括意圖的會話標題。\n要求：\n1. 僅輸出標題，不要解釋。\n2. 必須使用繁體中文。\n3. 不超過15個漢字。\n4. 優先使用具體主題詞和動作詞。"
            ),
        ),
        i18n::Language::Fr => (
            "Vous générez des titres de session concis.",
            format!(
                "Premier message utilisateur :\n{first_user_message}\n\nGénérez un titre de session concis qui résume l'intention.\nRègles :\n1. Retournez uniquement le titre.\n2. La langue de sortie doit être le français.\n3. Maximum 15 mots.\n4. Privilégiez des termes concrets et orientés action."
            ),
        ),
        i18n::Language::De => (
            "Sie erstellen prägnante Sitzungstitel.",
            format!(
                "Erste Nutzernachricht:\n{first_user_message}\n\nErstellen Sie einen kurzen Sitzungstitel, der die Absicht zusammenfasst.\nRegeln:\n1. Geben Sie nur den Titel zurück.\n2. Die Ausgabesprache muss Deutsch sein.\n3. Maximal 15 Wörter.\n4. Nutzen Sie konkrete Themen- und Aktionsbegriffe."
            ),
        ),
        i18n::Language::Ja => (
            "あなたは簡潔なセッションタイトルを生成します。",
            format!(
                "ユーザーの最初のメッセージ:\n{first_user_message}\n\n意図を要約した短いセッションタイトルを作成してください。\nルール:\n1. タイトルのみを出力する。\n2. 出力言語は日本語にする。\n3. 15語以内（または15文字以内）にする。\n4. 具体的なトピック語と行動語を優先する。"
            ),
        ),
        i18n::Language::En => (
            "You generate concise chat session titles.",
            format!(
                "First user message:\n{first_user_message}\n\nGenerate a concise session title that captures the user's intent.\nRules:\n1. Return title text only, no explanation.\n2. Output language must be English.\n3. Keep it within 15 words.\n4. Prefer concrete topic nouns and action phrases."
            ),
        ),
    }
}

fn normalize_session_auto_title_candidate(raw: &str, max_units: usize) -> Option<String> {
    let first_line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .trim_matches(|ch| matches!(ch, '"' | '\'' | '`' | '“' | '”' | '‘' | '’'))
        .trim();
    if first_line.is_empty() {
        return None;
    }
    let mut compact = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    for prefix in ["标题：", "标题:", "Title:", "title:"] {
        if let Some(rest) = compact.strip_prefix(prefix) {
            compact = rest.trim().to_string();
            break;
        }
    }
    let limited = limit_title_by_units(compact.as_str(), max_units);
    if limited.is_empty() {
        None
    } else {
        Some(limited)
    }
}

fn limit_title_by_units(text: &str, max_units: usize) -> String {
    if max_units == 0 || text.trim().is_empty() {
        return String::new();
    }
    let mut units = 0usize;
    let mut in_word = false;
    let mut pending_space = false;
    let mut out = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            in_word = false;
            continue;
        }
        if is_cjk_title_char(ch) {
            if units >= max_units {
                break;
            }
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            units = units.saturating_add(1);
            pending_space = false;
            in_word = false;
            continue;
        }
        if is_title_word_char(ch) {
            if !in_word {
                if units >= max_units {
                    break;
                }
                if pending_space && !out.is_empty() {
                    out.push(' ');
                }
                units = units.saturating_add(1);
                pending_space = false;
                in_word = true;
            }
            out.push(ch);
            continue;
        }
        in_word = false;
        if !out.is_empty() {
            pending_space = true;
        }
    }
    out.trim().to_string()
}

fn is_cjk_title_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{3040}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}'
    )
}

fn is_title_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '-' | '_' | '/' | '&' | '+')
}

#[allow(clippy::too_many_arguments)]
fn handle_key_event(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    system_prompt: &str,
    state: &mut ChatUiState,
    chat_turns: &mut usize,
    last_assistant_reply: &mut String,
) -> Result<bool, AppError> {
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c' | 'd' | 'z'))
    {
        return Ok(true);
    }
    if key.code == KeyCode::F(2) {
        state.theme_idx = (state.theme_idx + 1) % palettes().len();
        let _ = save_theme_preferences(state.theme_idx, &state.prefs_path);
        state.status = format!(
            "{}: {}",
            ui_text_theme_switched(),
            palette_by_index(state.theme_idx).name
        );
        return Ok(false);
    }
    if key.code == KeyCode::Tab {
        state.cycle_focus_forward();
        state.status = ui_text_focus_hint(state.focus).to_string();
        return Ok(false);
    }
    if key.code == KeyCode::BackTab {
        state.cycle_focus_backward();
        state.status = ui_text_focus_hint(state.focus).to_string();
        return Ok(false);
    }
    if state.pending_delete_confirm.is_some() {
        return handle_delete_message_confirm_key(key, services, state);
    }
    if state.pending_thread_delete_confirm.is_some() {
        return handle_thread_delete_confirm_key(key, services, state);
    }
    if state.pending_thread_metadata_modal.is_some() {
        return handle_thread_metadata_modal_key(key, state);
    }
    if state.pending_tool_result_modal.is_some() {
        return handle_tool_result_modal_key(key, state);
    }
    if state.pending_thread_rename.is_some() {
        return handle_thread_rename_modal_key(key, services, state);
    }
    if state.pending_thread_action_menu.is_some() {
        return handle_thread_action_menu_key(key, services, state);
    }
    if state.inspect_menu_open {
        return handle_inspect_menu_key(key, state);
    }
    match state.focus {
        FocusPanel::Nav => handle_nav_key(key, services, state),
        FocusPanel::Threads => handle_threads_key(key, services, state),
        FocusPanel::Conversation => handle_conversation_key(key, terminal, services, state),
        FocusPanel::Input => handle_input_key(
            key,
            terminal,
            services,
            system_prompt,
            state,
            chat_turns,
            last_assistant_reply,
        ),
    }
}

fn handle_nav_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => state.nav_selected = state.nav_selected.saturating_sub(1),
        KeyCode::Down => state.nav_selected = (state.nav_selected + 1).min(NAV_ITEMS_COUNT - 1),
        KeyCode::Enter => activate_nav_item(services, state)?,
        KeyCode::Char('/') if state.mode == UiMode::Chat => {
            state.focus = FocusPanel::Input;
            state.input.clear();
            state.input.insert_char('/');
        }
        _ => {}
    }
    Ok(false)
}

fn handle_threads_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => {
            state.thread_selected = state.thread_selected.saturating_sub(1);
        }
        KeyCode::Down => {
            if !state.threads.is_empty() {
                state.thread_selected = (state.thread_selected + 1).min(state.threads.len() - 1);
            }
        }
        KeyCode::Enter => {
            state.mode = UiMode::Chat;
            switch_selected_thread(services, state)?;
        }
        KeyCode::Char('/') if state.mode == UiMode::Chat => {
            state.focus = FocusPanel::Input;
            state.input.clear();
            state.input.insert_char('/');
        }
        _ => {}
    }
    Ok(false)
}

fn handle_conversation_key(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    if state.mode == UiMode::Inspect {
        match key.code {
            KeyCode::Enter => {
                state.inspect_menu_open = true;
                state.inspect_menu_selected = inspect_target_index(state.inspect.target);
                state.status = ui_text_inspect_target_menu_hint().to_string();
            }
            KeyCode::Char('i' | 'I') => {
                state.focus = FocusPanel::Nav;
                state.nav_selected = 3;
                state.status = ui_text_focus_hint(state.focus).to_string();
            }
            _ => {}
        }
        return Ok(false);
    }
    if state.mode == UiMode::Skills {
        return handle_skills_conversation_key(key, terminal, services, state);
    }
    if state.mode == UiMode::Mcp {
        return handle_mcp_conversation_key(key, state);
    }
    if state.mode == UiMode::Config {
        return handle_config_conversation_key(key, state);
    }
    let viewport = conversation_viewport_height(terminal)?;
    match key.code {
        KeyCode::Up => state.scroll_by(-1, viewport),
        KeyCode::Down => state.scroll_by(1, viewport),
        KeyCode::PageUp => state.scroll_by(-(viewport as i16), viewport),
        KeyCode::PageDown => state.scroll_by(viewport as i16, viewport),
        KeyCode::Home => {
            state.follow_tail = false;
            state.conversation_scroll = 0;
        }
        KeyCode::End => {
            state.follow_tail = true;
            state.ensure_conversation_cache();
            state.clamp_scroll(viewport);
        }
        KeyCode::Char('/') => {
            state.focus = FocusPanel::Input;
            state.input.clear();
            state.input.insert_char('/');
        }
        _ => {}
    }
    Ok(false)
}

fn handle_skills_conversation_key(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    if state.skill_doc_modal.is_some() {
        let area = terminal
            .size()
            .map_err(|err| AppError::Command(format!("failed to get terminal size: {err}")))?;
        let layout = compute_layout(Rect::new(0, 0, area.width, area.height));
        match key.code {
            KeyCode::Esc => {
                state.skill_doc_modal = None;
                state.status = ui_text_skills_doc_closed().to_string();
            }
            KeyCode::Up => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_add(1);
                }
            }
            KeyCode::PageUp => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_sub(8);
                }
            }
            KeyCode::PageDown => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_add(8);
                }
            }
            KeyCode::Home => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = 0;
                }
            }
            KeyCode::Char('e' | 'E') | KeyCode::Enter => {
                let (file_path, previous) = match state.skill_doc_modal.as_ref() {
                    Some(modal) => (modal.file_path.clone(), modal.raw_content.clone()),
                    None => return Ok(false),
                };
                let edited = match edit_text_with_external_editor(terminal, &previous) {
                    Ok(value) => value,
                    Err(err) => {
                        state.status = i18n::localize_error(&err);
                        return Ok(false);
                    }
                };
                if let Some(next) = edited {
                    if next == previous {
                        state.status = ui_text_skills_doc_unchanged().to_string();
                    } else {
                        if let Err(err) = fs::write(&file_path, &next) {
                            state.status = format!(
                                "{}: {}",
                                ui_text_skills_doc_save_failed(),
                                trim_ui_text(&format!("{}: {err}", file_path.display()), 84)
                            );
                            return Ok(false);
                        }
                        if let Some(modal_state) = state.skill_doc_modal.as_mut() {
                            modal_state.raw_content = next;
                            modal_state.rendered_content =
                                render_skill_markdown_for_modal(&modal_state.raw_content);
                            modal_state.scroll = 0;
                        }
                        state.status = format!(
                            "{}: {}",
                            ui_text_skills_doc_saved(),
                            trim_ui_text(&file_path.display().to_string(), 72)
                        );
                    }
                } else {
                    state.status = ui_text_skills_doc_edit_cancelled().to_string();
                }
            }
            _ => {}
        }
        clamp_skill_doc_modal_scroll(state, layout);
        return Ok(false);
    }

    let rows = build_skill_panel_rows(
        expand_tilde(&services.cfg.ai.tools.skills.dir).as_path(),
        services.skills,
    );
    if rows.is_empty() {
        if matches!(key.code, KeyCode::Esc | KeyCode::Char('s' | 'S')) {
            state.focus = FocusPanel::Nav;
            state.nav_selected = 1;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        return Ok(false);
    }
    state.skills_selected_row = state.skills_selected_row.min(rows.len().saturating_sub(1));
    match key.code {
        KeyCode::Up => {
            state.skills_selected_row = state.skills_selected_row.saturating_sub(1);
        }
        KeyCode::Down => {
            state.skills_selected_row = (state.skills_selected_row + 1).min(rows.len() - 1);
        }
        KeyCode::PageUp => {
            state.skills_selected_row = state.skills_selected_row.saturating_sub(8);
        }
        KeyCode::PageDown => {
            state.skills_selected_row = (state.skills_selected_row + 8).min(rows.len() - 1);
        }
        KeyCode::Home => {
            state.skills_selected_row = 0;
        }
        KeyCode::End => {
            state.skills_selected_row = rows.len().saturating_sub(1);
        }
        KeyCode::Enter | KeyCode::Char('e' | 'E') => {
            open_selected_skill_doc_modal(services, state)?;
        }
        KeyCode::Esc | KeyCode::Char('s' | 'S') => {
            state.focus = FocusPanel::Nav;
            state.nav_selected = 1;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        _ => {}
    }
    Ok(false)
}

fn open_selected_skill_doc_modal(
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    let skills_dir = expand_tilde(&services.cfg.ai.tools.skills.dir);
    let rows = build_skill_panel_rows(skills_dir.as_path(), services.skills);
    if rows.is_empty() {
        state.status = ui_text_skills_empty().to_string();
        return Ok(());
    }
    state.skills_selected_row = state.skills_selected_row.min(rows.len().saturating_sub(1));
    let selected = &rows[state.skills_selected_row];
    let file_path = skills_dir.join(&selected.name).join("SKILL.md");
    let raw_content = match fs::read_to_string(&file_path) {
        Ok(content) => content,
        Err(err) => {
            state.status = format!(
                "{}: {}",
                ui_text_skills_doc_open_failed(),
                trim_ui_text(&format!("{}: {err}", file_path.display()), 84)
            );
            return Ok(());
        }
    };
    state.skill_doc_modal = Some(SkillDocModalState {
        skill_name: selected.name.clone(),
        file_path: file_path.clone(),
        raw_content: raw_content.clone(),
        rendered_content: render_skill_markdown_for_modal(&raw_content),
        scroll: 0,
    });
    state.status = format!(
        "{}: {}",
        ui_text_skills_doc_opened(),
        trim_ui_text(&selected.name, 36)
    );
    Ok(())
}

fn render_skill_markdown_for_modal(raw: &str) -> String {
    normalize_conversation_markdown_for_bubble(&render::render_markdown_for_terminal(raw, false))
}

fn handle_config_conversation_key(
    key: KeyEvent,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => config_scroll_fields(state, -1),
        KeyCode::Down => config_scroll_fields(state, 1),
        KeyCode::PageUp => config_scroll_fields(state, -8),
        KeyCode::PageDown => config_scroll_fields(state, 8),
        KeyCode::Left => config_switch_category(state, -1),
        KeyCode::Right => config_switch_category(state, 1),
        KeyCode::Home => {
            state.config_ui.selected_field_row = 0;
        }
        KeyCode::End => {
            let count = config_visible_field_indices(state).len();
            if count > 0 {
                state.config_ui.selected_field_row = count - 1;
            }
        }
        KeyCode::Enter => {
            config_activate_selected_field(state, true);
        }
        KeyCode::Char(' ') => {
            if config_step_selected_option(state, 1) {
                state.status = ui_text_config_edit_applied().to_string();
            }
        }
        KeyCode::Char('s' | 'S') | KeyCode::F(5) => {
            state.focus = FocusPanel::Input;
        }
        _ => {}
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn handle_input_key(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    system_prompt: &str,
    state: &mut ChatUiState,
    chat_turns: &mut usize,
    last_assistant_reply: &mut String,
) -> Result<bool, AppError> {
    if state.mode == UiMode::Config {
        return handle_config_input_key(key, services, state);
    }
    if state.mode == UiMode::Mcp {
        return handle_mcp_input_key(key, services, state);
    }
    if state.mode != UiMode::Chat {
        if key.code == KeyCode::Esc {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        return Ok(false);
    }
    if state.pending_choice.is_some() {
        return handle_choice_input_key(
            key,
            terminal,
            services,
            system_prompt,
            state,
            chat_turns,
            last_assistant_reply,
        );
    }
    match key.code {
        KeyCode::Esc => {
            state.input.clear();
            state.status = ui_text_ready().to_string();
        }
        KeyCode::Left => state.input.move_left(),
        KeyCode::Right => state.input.move_right(),
        KeyCode::Home => state.input.move_home(),
        KeyCode::End => state.input.move_end(),
        KeyCode::Backspace => state.input.backspace(),
        KeyCode::Delete => state.input.delete(),
        KeyCode::Up => {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        KeyCode::Enter => {
            let message = state.input.text.trim().to_string();
            state.input.clear();
            if message.is_empty() {
                return Ok(false);
            }
            if submit_chat_message(
                terminal,
                services,
                system_prompt,
                state,
                chat_turns,
                last_assistant_reply,
                message,
            )? {
                return Ok(true);
            }
        }
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                state.input.insert_char(ch);
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_config_input_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('s')) {
        return save_config_ui(services, state).map(|_| false);
    }
    match key.code {
        KeyCode::Esc => {
            if state.config_ui.editing {
                state.config_ui.editing = false;
                state.status = ui_text_config_edit_cancelled().to_string();
            } else {
                state.focus = FocusPanel::Conversation;
                state.status = ui_text_focus_hint(state.focus).to_string();
            }
        }
        KeyCode::Up => {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        KeyCode::Enter => {
            if state.config_ui.editing {
                config_apply_edit_buffer(state)?;
            } else if state.config_ui.dirty_count > 0 {
                save_config_ui(services, state)?;
            } else {
                config_activate_selected_field(state, false);
            }
        }
        KeyCode::Left => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.move_left();
            } else if config_step_selected_option(state, -1) {
                state.status = ui_text_config_edit_applied().to_string();
            }
        }
        KeyCode::Right => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.move_right();
            } else if config_step_selected_option(state, 1) {
                state.status = ui_text_config_edit_applied().to_string();
            }
        }
        KeyCode::Home => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.move_home();
            }
        }
        KeyCode::End => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.move_end();
            }
        }
        KeyCode::Backspace => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.backspace();
            }
        }
        KeyCode::Delete => {
            if state.config_ui.editing {
                state.config_ui.edit_buffer.delete();
            }
        }
        KeyCode::Char(ch) => {
            if state.config_ui.editing {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    state.config_ui.edit_buffer.insert_char(ch);
                }
            } else if ch == ' ' && config_step_selected_option(state, 1) {
                state.status = ui_text_config_edit_applied().to_string();
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_mcp_conversation_key(key: KeyEvent, state: &mut ChatUiState) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => {
            if state.mcp_ui.focus_servers {
                mcp_scroll_servers(state, -1);
            } else {
                mcp_scroll_fields(state, -1);
            }
        }
        KeyCode::Down => {
            if state.mcp_ui.focus_servers {
                mcp_scroll_servers(state, 1);
            } else {
                mcp_scroll_fields(state, 1);
            }
        }
        KeyCode::PageUp => {
            if state.mcp_ui.focus_servers {
                mcp_scroll_servers(state, -6);
            } else {
                mcp_scroll_fields(state, -6);
            }
        }
        KeyCode::PageDown => {
            if state.mcp_ui.focus_servers {
                mcp_scroll_servers(state, 6);
            } else {
                mcp_scroll_fields(state, 6);
            }
        }
        KeyCode::Left => {
            state.mcp_ui.focus_servers = true;
            state.status = ui_text_mcp_focus_servers().to_string();
        }
        KeyCode::Right => {
            state.mcp_ui.focus_servers = false;
            state.status = ui_text_mcp_focus_fields().to_string();
        }
        KeyCode::Enter => {
            if state.mcp_ui.focus_servers {
                state.mcp_ui.focus_servers = false;
                state.status = ui_text_mcp_focus_fields().to_string();
            } else {
                let field = mcp_selected_field_def(state);
                if let Some(field) = field {
                    if mcp_field_is_selectable(field) {
                        mcp_cycle_field_option(state, field.id, 1)?;
                    } else {
                        state.focus = FocusPanel::Input;
                        mcp_start_edit(state, field.id)?;
                    }
                }
            }
        }
        KeyCode::Char(' ') => {
            if !state.mcp_ui.focus_servers {
                let field = mcp_selected_field_def(state);
                if let Some(field) = field
                    && mcp_field_is_selectable(field)
                {
                    mcp_cycle_field_option(state, field.id, 1)?;
                }
            }
        }
        KeyCode::Char('a' | 'A') => {
            mcp_add_server(state);
        }
        KeyCode::Char('d' | 'D') => {
            mcp_delete_selected_server(state);
        }
        KeyCode::Char('s' | 'S') | KeyCode::F(5) => {
            state.focus = FocusPanel::Input;
            state.status = ui_text_mcp_input_ready().to_string();
        }
        _ => {}
    }
    Ok(false)
}

fn handle_mcp_input_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('s')) {
        return save_mcp_ui(services, state).map(|_| false);
    }
    match key.code {
        KeyCode::Esc => {
            if state.mcp_ui.editing {
                state.mcp_ui.editing = false;
                state.status = ui_text_mcp_edit_cancelled().to_string();
            } else {
                state.focus = FocusPanel::Conversation;
                state.status = ui_text_focus_hint(state.focus).to_string();
            }
        }
        KeyCode::Up => {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        KeyCode::Enter => {
            if state.mcp_ui.editing {
                mcp_apply_edit_buffer(state)?;
            } else if state.mcp_ui.dirty_count > 0 {
                save_mcp_ui(services, state)?;
            } else if let Some(field) = mcp_selected_field_def(state) {
                if mcp_field_is_selectable(field) {
                    mcp_cycle_field_option(state, field.id, 1)?;
                } else {
                    mcp_start_edit(state, field.id)?;
                }
            }
        }
        KeyCode::Left => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.move_left();
            } else if let Some(field) = mcp_selected_field_def(state)
                && mcp_field_is_selectable(field)
            {
                mcp_cycle_field_option(state, field.id, -1)?;
            }
        }
        KeyCode::Right => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.move_right();
            } else if let Some(field) = mcp_selected_field_def(state)
                && mcp_field_is_selectable(field)
            {
                mcp_cycle_field_option(state, field.id, 1)?;
            }
        }
        KeyCode::Home => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.move_home();
            }
        }
        KeyCode::End => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.move_end();
            }
        }
        KeyCode::Backspace => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.backspace();
            }
        }
        KeyCode::Delete => {
            if state.mcp_ui.editing {
                state.mcp_ui.edit_buffer.delete();
            }
        }
        KeyCode::Char(ch) => {
            if state.mcp_ui.editing {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    state.mcp_ui.edit_buffer.insert_char(ch);
                }
            } else if ch == ' '
                && let Some(field) = mcp_selected_field_def(state)
                && mcp_field_is_selectable(field)
            {
                mcp_cycle_field_option(state, field.id, 1)?;
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_mouse_event(
    mouse: MouseEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    let area = terminal
        .size()
        .map_err(|err| AppError::Command(format!("failed to get terminal size: {err}")))?;
    let layout = compute_layout(Rect::new(0, 0, area.width, area.height));
    if state.mode == UiMode::Chat {
        refresh_conversation_cache_for_layout(state, layout);
    } else {
        state.hovered_message_idx = None;
    }
    if state.pending_delete_confirm.is_some() {
        handle_delete_message_confirm_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_delete_confirm.is_some() {
        handle_thread_delete_confirm_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_metadata_modal.is_some() {
        handle_thread_metadata_modal_mouse(mouse, state, layout)?;
        return Ok(());
    }
    if state.pending_tool_result_modal.is_some() {
        handle_tool_result_modal_mouse(mouse, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_rename.is_some() {
        handle_thread_rename_modal_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_action_menu.is_some() {
        handle_thread_action_menu_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.mode == UiMode::Skills && state.skill_doc_modal.is_some() {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_sub(2);
                }
            }
            MouseEventKind::ScrollDown => {
                if let Some(modal) = state.skill_doc_modal.as_mut() {
                    modal.scroll = modal.scroll.saturating_add(2);
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if !rect_contains(skill_doc_modal_rect(layout), mouse.column, mouse.row) {
                    state.skill_doc_modal = None;
                    state.status = ui_text_skills_doc_closed().to_string();
                }
            }
            _ => {}
        }
        clamp_skill_doc_modal_scroll(state, layout);
        return Ok(());
    }
    if state.inspect_menu_open && matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        let menu = inspect_menu_rect(layout);
        if rect_contains(menu, mouse.column, mouse.row) {
            let inner = menu.inner(ratatui::layout::Margin {
                horizontal: 1,
                vertical: 1,
            });
            let idx = mouse.row.saturating_sub(inner.y) as usize;
            if idx < inspect_targets().len() {
                state.inspect_menu_selected = idx;
                apply_selected_inspect_target(state);
            }
            return Ok(());
        }
        state.inspect_menu_open = false;
    }
    if state.mode == UiMode::Chat && matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        if let Some(button) = conversation_copy_button_data(state, layout)
            && rect_contains(button.rect, mouse.column, mouse.row)
        {
            if let Some(message) = state.messages.get(button.message_index) {
                match copy_text_to_clipboard(&message.text) {
                    Ok(()) => {
                        state.status = ui_text_message_copy_success().to_string();
                    }
                    Err(err) => {
                        state.status = format!(
                            "{}: {}",
                            ui_text_message_copy_failed(),
                            trim_ui_text(&i18n::localize_error(&err), 84)
                        );
                    }
                }
            } else {
                state.status = ui_text_message_copy_failed().to_string();
            }
            return Ok(());
        }
        if let Some(button) = conversation_delete_button_data(state, layout)
            && rect_contains(button.rect, mouse.column, mouse.row)
        {
            state.pending_delete_confirm = Some(DeleteMessageConfirmState {
                message_index: button.message_index,
                selected: 1,
            });
            state.status = ui_text_message_delete_confirm_prompt().to_string();
            return Ok(());
        }
        if let Some(button) = conversation_tool_result_button_data(state, layout)
            && rect_contains(button.rect, mouse.column, mouse.row)
        {
            open_tool_result_modal(state, button.message_index);
            return Ok(());
        }
    }
    match mouse.kind {
        MouseEventKind::Moved => {
            if state.mode == UiMode::Chat {
                refresh_hovered_message_from_mouse(state, layout, mouse);
            }
        }
        MouseEventKind::ScrollUp => {
            if (state.mode == UiMode::Chat
                || state.mode == UiMode::Config
                || state.mode == UiMode::Mcp)
                && rect_contains(layout.conversation, mouse.column, mouse.row)
            {
                if state.mode == UiMode::Config {
                    config_scroll_fields(state, -1);
                } else if state.mode == UiMode::Mcp {
                    mcp_scroll_by_mouse(layout, mouse, state, -1);
                } else {
                    let viewport = layout.conversation_body.height.max(1);
                    state.scroll_by(-2, viewport);
                }
                state.focus = FocusPanel::Conversation;
                if state.mode == UiMode::Chat {
                    refresh_hovered_message_from_mouse(state, layout, mouse);
                }
            }
        }
        MouseEventKind::ScrollDown => {
            if (state.mode == UiMode::Chat
                || state.mode == UiMode::Config
                || state.mode == UiMode::Mcp)
                && rect_contains(layout.conversation, mouse.column, mouse.row)
            {
                if state.mode == UiMode::Config {
                    config_scroll_fields(state, 1);
                } else if state.mode == UiMode::Mcp {
                    mcp_scroll_by_mouse(layout, mouse, state, 1);
                } else {
                    let viewport = layout.conversation_body.height.max(1);
                    state.scroll_by(2, viewport);
                }
                state.focus = FocusPanel::Conversation;
                if state.mode == UiMode::Chat {
                    refresh_hovered_message_from_mouse(state, layout, mouse);
                }
            }
        }
        MouseEventKind::Down(MouseButton::Left) => {
            if state.mode == UiMode::Skills
                && rect_contains(inspect_panel_rect(layout), mouse.column, mouse.row)
            {
                state.focus = FocusPanel::Conversation;
                if let Some(row_idx) = skill_row_index_from_mouse(layout, mouse, services) {
                    state.skills_selected_row = row_idx;
                    open_selected_skill_doc_modal(services, state)?;
                }
                return Ok(());
            }
            if rect_contains(layout.nav, mouse.column, mouse.row) {
                state.focus = FocusPanel::Nav;
                let nav_idx = mouse.row.saturating_sub(layout.nav.y.saturating_add(1)) as usize;
                if nav_idx < NAV_ITEMS_COUNT {
                    state.nav_selected = nav_idx;
                    activate_nav_item(services, state)?;
                }
            } else if (state.mode == UiMode::Chat
                || state.mode == UiMode::Config
                || state.mode == UiMode::Mcp)
                && rect_contains(layout.input, mouse.column, mouse.row)
            {
                state.focus = FocusPanel::Input;
                if state.mode == UiMode::Config {
                    if mouse.row >= layout.input_body.y.saturating_add(2)
                        && state.config_ui.dirty_count > 0
                    {
                        let _ = save_config_ui(services, state);
                    } else if let Some(field_idx) = config_selected_field_index(state) {
                        config_start_edit(state, field_idx);
                    }
                } else if state.mode == UiMode::Mcp {
                    if mouse.row >= layout.input_body.y.saturating_add(2)
                        && state.mcp_ui.dirty_count > 0
                    {
                        let _ = save_mcp_ui(services, state);
                    } else if let Some(field) = mcp_selected_field_def(state) {
                        let _ = mcp_start_edit(state, field.id);
                    }
                } else if let Some(choice) = state.pending_choice.as_ref() {
                    if choice.options.is_empty() {
                        state.input.move_end();
                    } else {
                        let row = mouse.row.saturating_sub(layout.input_body.y) as usize;
                        if row == 0 {
                            if let Some(item) = state.pending_choice.as_mut() {
                                item.selected = item.selected.min(item.options.len() - 1);
                            }
                        } else {
                            let idx = row.saturating_sub(1);
                            if idx < choice.options.len()
                                && let Some(item) = state.pending_choice.as_mut()
                            {
                                item.selected = idx;
                            }
                        }
                    }
                } else {
                    state.input.move_end();
                }
            } else if rect_contains(layout.conversation, mouse.column, mouse.row) {
                state.focus = FocusPanel::Conversation;
                if state.mode == UiMode::Config {
                    config_select_by_mouse(layout, mouse, state);
                } else if state.mode == UiMode::Mcp {
                    mcp_select_by_mouse(layout, mouse, state);
                } else if state.mode == UiMode::Chat {
                    refresh_hovered_message_from_mouse(state, layout, mouse);
                }
            } else if rect_contains(layout.threads_body, mouse.column, mouse.row) {
                state.focus = FocusPanel::Threads;
                let row = mouse.row.saturating_sub(layout.threads_body.y);
                let idx = row as usize;
                if idx < state.threads.len() {
                    state.thread_selected = idx;
                    let now = now_epoch_ms();
                    let is_double_click = state.last_thread_click.is_some_and(|item| {
                        item.thread_index == idx
                            && now.saturating_sub(item.clicked_at_epoch_ms)
                                <= THREAD_DOUBLE_CLICK_WINDOW_MS
                    });
                    state.last_thread_click = Some(ThreadClickState {
                        thread_index: idx,
                        clicked_at_epoch_ms: now,
                    });
                    if is_double_click {
                        state.pending_thread_action_menu = Some(ThreadActionMenuState {
                            thread_index: idx,
                            selected: 0,
                        });
                        state.status = ui_text_thread_action_menu_hint().to_string();
                        return Ok(());
                    }
                    state.mode = UiMode::Chat;
                    switch_selected_thread(services, state)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn handle_delete_message_confirm_mouse(
    mouse: MouseEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    if !matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        return Ok(());
    }
    let modal_rect = delete_message_modal_rect(layout);
    if !rect_contains(modal_rect, mouse.column, mouse.row) {
        state.pending_delete_confirm = None;
        state.status = ui_text_message_delete_cancelled().to_string();
        return Ok(());
    }
    let (confirm_rect, cancel_rect) = delete_message_modal_button_rects(modal_rect);
    if rect_contains(confirm_rect, mouse.column, mouse.row) {
        let idx = state
            .pending_delete_confirm
            .as_ref()
            .map(|item| item.message_index)
            .unwrap_or(usize::MAX);
        state.pending_delete_confirm = None;
        let _ = delete_message_from_session(services, state, idx)?;
        return Ok(());
    }
    if rect_contains(cancel_rect, mouse.column, mouse.row) {
        state.pending_delete_confirm = None;
        state.status = ui_text_message_delete_cancelled().to_string();
        return Ok(());
    }
    Ok(())
}

fn handle_thread_delete_confirm_mouse(
    mouse: MouseEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    if !matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        return Ok(());
    }
    let modal_rect = thread_delete_modal_rect(layout);
    if !rect_contains(modal_rect, mouse.column, mouse.row) {
        state.pending_thread_delete_confirm = None;
        state.status = ui_text_thread_delete_cancelled().to_string();
        return Ok(());
    }
    let (confirm_rect, cancel_rect) = thread_delete_modal_button_rects(modal_rect);
    if rect_contains(confirm_rect, mouse.column, mouse.row) {
        let idx = state
            .pending_thread_delete_confirm
            .as_ref()
            .map(|item| item.thread_index)
            .unwrap_or(usize::MAX);
        state.pending_thread_delete_confirm = None;
        delete_thread_session_by_index(services, state, idx)?;
        return Ok(());
    }
    if rect_contains(cancel_rect, mouse.column, mouse.row) {
        state.pending_thread_delete_confirm = None;
        state.status = ui_text_thread_delete_cancelled().to_string();
        return Ok(());
    }
    Ok(())
}

fn handle_thread_metadata_modal_mouse(
    mouse: MouseEvent,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            if let Some(modal) = state.pending_thread_metadata_modal.as_mut() {
                modal.scroll = modal.scroll.saturating_sub(1);
            }
        }
        MouseEventKind::ScrollDown => {
            if let Some(modal) = state.pending_thread_metadata_modal.as_mut() {
                let max_scroll = modal.rows.len().saturating_sub(1);
                modal.scroll = (modal.scroll + 1).min(max_scroll);
            }
        }
        MouseEventKind::Down(MouseButton::Left) => {
            if !rect_contains(thread_metadata_modal_rect(layout), mouse.column, mouse.row) {
                state.pending_thread_metadata_modal = None;
                state.status = ui_text_thread_metadata_closed().to_string();
            }
        }
        _ => {}
    }
    Ok(())
}

fn handle_tool_result_modal_mouse(
    mouse: MouseEvent,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            if let Some(modal) = state.pending_tool_result_modal.as_mut() {
                modal.scroll = modal.scroll.saturating_sub(1);
            }
        }
        MouseEventKind::ScrollDown => {
            if let Some(modal) = state.pending_tool_result_modal.as_mut() {
                let max_scroll = tool_result_modal_scroll_max(modal, layout);
                modal.scroll = modal.scroll.saturating_add(1).min(max_scroll);
            }
        }
        MouseEventKind::Down(MouseButton::Left) => {
            if !rect_contains(tool_result_modal_rect(layout), mouse.column, mouse.row) {
                state.pending_tool_result_modal = None;
                state.status = ui_text_message_tool_result_modal_closed().to_string();
            }
        }
        _ => {}
    }
    Ok(())
}

fn handle_tool_result_modal_key(key: KeyEvent, state: &mut ChatUiState) -> Result<bool, AppError> {
    let Some(modal) = state.pending_tool_result_modal.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_tool_result_modal = None;
            state.status = ui_text_message_tool_result_modal_closed().to_string();
        }
        KeyCode::Up => modal.scroll = modal.scroll.saturating_sub(1),
        KeyCode::Down => modal.scroll = modal.scroll.saturating_add(1),
        KeyCode::PageUp => modal.scroll = modal.scroll.saturating_sub(8),
        KeyCode::PageDown => modal.scroll = modal.scroll.saturating_add(8),
        KeyCode::Home => modal.scroll = 0,
        KeyCode::End => modal.scroll = u16::MAX,
        _ => {}
    }
    Ok(false)
}

fn open_tool_result_modal(state: &mut ChatUiState, message_index: usize) {
    let Some(message) = state.messages.get(message_index) else {
        state.status = ui_text_message_tool_result_unavailable().to_string();
        return;
    };
    let Some(detail) = tool_result_detail_from_message(message) else {
        state.status = ui_text_message_tool_result_unavailable().to_string();
        return;
    };
    let mut lines = Vec::<String>::new();
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_modal_subtitle(),
        detail.function_name
    ));
    lines.push(String::new());
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_call_id(),
        detail.tool_call_id
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_time(),
        format_epoch_ms(detail.executed_at_epoch_ms)
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_account(),
        if detail.account.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.account.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_env(),
        if detail.environment.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.environment.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_os(),
        if detail.os_name.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.os_name.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_cwd(),
        if detail.cwd.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.cwd.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_label(),
        if detail.label.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.label.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_mode(),
        if detail.mode.trim().is_empty() {
            ui_text_na().to_string()
        } else {
            detail.mode.clone()
        }
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_exit_code(),
        detail
            .exit_code
            .map(|code| code.to_string())
            .unwrap_or_else(|| ui_text_na().to_string())
    ));
    lines.push(format!(
        "{} {}",
        ui_text_message_tool_result_meta_duration(),
        if detail.duration_ms == 0 {
            ui_text_na().to_string()
        } else {
            format!("{}ms", detail.duration_ms)
        }
    ));
    lines.push(format!(
        "{} timeout={} interrupted={} blocked={}",
        ui_text_message_tool_result_meta_status(),
        detail.timed_out,
        detail.interrupted,
        detail.blocked
    ));
    lines.push(String::new());
    lines.push(format!("{}:", ui_text_message_tool_result_meta_command()));
    lines.push(if detail.command.trim().is_empty() {
        ui_text_na().to_string()
    } else {
        detail.command
    });
    lines.push(String::new());
    lines.push(format!("{}:", ui_text_message_tool_result_meta_arguments()));
    lines.push(if detail.arguments.trim().is_empty() {
        ui_text_na().to_string()
    } else {
        detail.arguments
    });
    lines.push(String::new());
    lines.push(format!("{}:", ui_text_message_tool_result_meta_output()));
    lines.push(if detail.result_payload.trim().is_empty() {
        ui_text_na().to_string()
    } else {
        detail.result_payload
    });
    state.pending_tool_result_modal = Some(ToolResultModalState {
        message_index,
        lines,
        scroll: 0,
    });
    state.status = ui_text_message_tool_result_modal_opened().to_string();
}

fn handle_thread_action_menu_mouse(
    mouse: MouseEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    if !matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        return Ok(());
    }
    let rect = thread_action_menu_rect(layout);
    if !rect_contains(rect, mouse.column, mouse.row) {
        state.pending_thread_action_menu = None;
        state.status = ui_text_ready().to_string();
        return Ok(());
    }
    let inner = rect.inner(ratatui::layout::Margin {
        horizontal: 2,
        vertical: 2,
    });
    let idx = mouse.row.saturating_sub(inner.y) as usize;
    if idx < thread_action_menu_options().len() {
        if let Some(menu) = state.pending_thread_action_menu.as_mut() {
            menu.selected = idx;
        }
        apply_thread_action_menu_selection(services, state)?;
    }
    Ok(())
}

fn handle_thread_rename_modal_mouse(
    mouse: MouseEvent,
    _services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
) -> Result<(), AppError> {
    if !matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        return Ok(());
    }
    if !rect_contains(thread_rename_modal_rect(layout), mouse.column, mouse.row) {
        state.pending_thread_rename = None;
        state.status = ui_text_thread_rename_cancelled().to_string();
    }
    Ok(())
}

fn delete_message_from_session(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    message_index: usize,
) -> Result<bool, AppError> {
    let Some(message) = state.messages.get(message_index).cloned() else {
        state.status = ui_text_message_delete_not_found().to_string();
        return Ok(false);
    };
    let Some(session_role) = ui_role_to_session_role(message.role) else {
        state.status = ui_text_message_delete_not_supported().to_string();
        return Ok(false);
    };
    if !state.is_message_persisted(message_index) {
        state.status = ui_text_message_delete_not_supported().to_string();
        return Ok(false);
    }
    let Some(occurrence_from_end) =
        same_persisted_message_occurrence_from_end(state, message_index)
    else {
        state.status = ui_text_message_delete_not_supported().to_string();
        return Ok(false);
    };
    let removed = services.session.remove_recent_display_message_by_signature(
        CHAT_RENDER_LIMIT,
        session_role,
        &message.text,
        occurrence_from_end,
    );
    if removed.is_none() {
        state.status = ui_text_message_delete_not_found().to_string();
        return Ok(false);
    }
    services.session.persist()?;
    let _ = state.remove_message_at(message_index);
    state.set_threads(services.session.list_sessions().unwrap_or_default());
    state.status = ui_text_message_delete_success().to_string();
    Ok(true)
}

fn same_persisted_message_occurrence_from_end(
    state: &ChatUiState,
    target_index: usize,
) -> Option<usize> {
    let target = state.messages.get(target_index)?;
    if !state.is_message_persisted(target_index) {
        return None;
    }
    let mut count = 0usize;
    for idx in (0..state.messages.len()).rev() {
        if !state.is_message_persisted(idx) {
            continue;
        }
        let Some(current) = state.messages.get(idx) else {
            continue;
        };
        if current.role == target.role && current.text == target.text {
            count = count.saturating_add(1);
            if idx == target_index {
                return Some(count.max(1));
            }
        }
    }
    None
}

fn ui_role_to_session_role(role: UiRole) -> Option<&'static str> {
    match role {
        UiRole::User => Some("user"),
        UiRole::Assistant => Some("assistant"),
        UiRole::System => Some("system"),
        UiRole::Tool => Some("tool"),
        UiRole::Thinking => Some("thinking"),
    }
}

fn switch_selected_thread(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    if state.threads.is_empty() || state.thread_selected >= state.threads.len() {
        return Ok(());
    }
    state.remember_current_session_messages();
    let target = state.threads[state.thread_selected].session_id.clone();
    let switched = services.session.switch_session_by_query(&target)?;
    state.set_active_session(switched.session_id.clone());
    state.pending_thread_action_menu = None;
    state.pending_thread_rename = None;
    state.pending_thread_delete_confirm = None;
    state.pending_thread_metadata_modal = None;
    state.clear_conversation_viewport_only();
    state.push(
        UiRole::System,
        i18n::chat_session_changed(
            switched.session_name.as_str(),
            switched.session_id.as_str(),
            switched.file_path.as_path(),
        ),
    );
    let current = services
        .session
        .recent_messages_for_display(CHAT_RENDER_LIMIT);
    let restored = state.session_messages_or_fallback(
        switched.session_id.as_str(),
        recent_messages_to_ui_messages(&current),
    );
    for (item, persisted) in restored {
        state.push_message_with_persisted(item, persisted);
    }
    state.remember_current_session_messages();
    state.set_threads(services.session.list_sessions().unwrap_or_default());
    services.session.persist()?;
    Ok(())
}

fn selected_thread_is_active_session(state: &ChatUiState) -> bool {
    state
        .threads
        .get(state.thread_selected)
        .is_some_and(|item| item.session_id == state.current_session_id)
}

fn handle_inspect_menu_key(key: KeyEvent, state: &mut ChatUiState) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Esc => {
            state.inspect_menu_open = false;
            state.status = ui_text_ready().to_string();
        }
        KeyCode::Up => {
            state.inspect_menu_selected = state.inspect_menu_selected.saturating_sub(1);
        }
        KeyCode::Down => {
            state.inspect_menu_selected =
                (state.inspect_menu_selected + 1).min(inspect_targets().len().saturating_sub(1));
        }
        KeyCode::Enter => apply_selected_inspect_target(state),
        KeyCode::Char(ch) if ch.is_ascii_digit() => {
            if let Some(idx) = ch
                .to_digit(10)
                .map(|v| v as usize)
                .and_then(|v| v.checked_sub(1))
                && idx < inspect_targets().len()
            {
                state.inspect_menu_selected = idx;
                apply_selected_inspect_target(state);
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_delete_message_confirm_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    let Some(modal) = state.pending_delete_confirm.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_delete_confirm = None;
            state.status = ui_text_message_delete_cancelled().to_string();
        }
        KeyCode::Left | KeyCode::Up => {
            modal.selected = modal.selected.saturating_sub(1);
        }
        KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
            modal.selected = (modal.selected + 1).min(1);
        }
        KeyCode::Enter => {
            let index = modal.message_index;
            let confirm_delete = modal.selected == 0;
            state.pending_delete_confirm = None;
            if confirm_delete {
                let _ = delete_message_from_session(services, state, index)?;
            } else {
                state.status = ui_text_message_delete_cancelled().to_string();
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_thread_delete_confirm_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    let Some(modal) = state.pending_thread_delete_confirm.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_thread_delete_confirm = None;
            state.status = ui_text_thread_delete_cancelled().to_string();
        }
        KeyCode::Left | KeyCode::Up => {
            modal.selected = modal.selected.saturating_sub(1);
        }
        KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
            modal.selected = (modal.selected + 1).min(1);
        }
        KeyCode::Enter => {
            let index = modal.thread_index;
            let confirm_delete = modal.selected == 0;
            state.pending_thread_delete_confirm = None;
            if confirm_delete {
                delete_thread_session_by_index(services, state, index)?;
            } else {
                state.status = ui_text_thread_delete_cancelled().to_string();
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_thread_metadata_modal_key(
    key: KeyEvent,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    let Some(modal) = state.pending_thread_metadata_modal.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc | KeyCode::Enter => {
            state.pending_thread_metadata_modal = None;
            state.status = ui_text_thread_metadata_closed().to_string();
        }
        KeyCode::Up => {
            modal.scroll = modal.scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Tab => {
            let max_scroll = modal.rows.len().saturating_sub(1);
            modal.scroll = (modal.scroll + 1).min(max_scroll);
        }
        KeyCode::PageUp => {
            modal.scroll = modal.scroll.saturating_sub(6);
        }
        KeyCode::PageDown => {
            let max_scroll = modal.rows.len().saturating_sub(1);
            modal.scroll = (modal.scroll + 6).min(max_scroll);
        }
        KeyCode::Home => {
            modal.scroll = 0;
        }
        KeyCode::End => {
            modal.scroll = modal.rows.len().saturating_sub(1);
        }
        _ => {}
    }
    Ok(false)
}

fn handle_thread_action_menu_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    let Some(menu) = state.pending_thread_action_menu.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_thread_action_menu = None;
            state.status = ui_text_ready().to_string();
        }
        KeyCode::Up => {
            menu.selected = menu.selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Tab => {
            menu.selected = (menu.selected + 1).min(thread_action_menu_options().len() - 1);
        }
        KeyCode::Enter => {
            apply_thread_action_menu_selection(services, state)?;
        }
        _ => {}
    }
    Ok(false)
}

fn handle_thread_rename_modal_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    let Some(modal) = state.pending_thread_rename.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_thread_rename = None;
            state.status = ui_text_thread_rename_cancelled().to_string();
        }
        KeyCode::Left => modal.input.move_left(),
        KeyCode::Right => modal.input.move_right(),
        KeyCode::Home => modal.input.move_home(),
        KeyCode::End => modal.input.move_end(),
        KeyCode::Backspace => modal.input.backspace(),
        KeyCode::Delete => modal.input.delete(),
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                modal.input.insert_char(ch);
            }
        }
        KeyCode::Enter => {
            apply_thread_rename_modal(services, state)?;
        }
        _ => {}
    }
    Ok(false)
}

fn apply_thread_action_menu_selection(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    let Some(menu) = state.pending_thread_action_menu.take() else {
        return Ok(());
    };
    let idx = menu.thread_index;
    match menu.selected {
        0 => open_thread_delete_confirm(state, idx)?,
        1 => open_thread_rename_modal(state, idx)?,
        _ => open_thread_metadata_modal(services, state, idx)?,
    }
    Ok(())
}

fn open_thread_delete_confirm(
    state: &mut ChatUiState,
    thread_index: usize,
) -> Result<(), AppError> {
    if state.threads.get(thread_index).is_none() {
        state.status = ui_text_thread_not_found().to_string();
        return Ok(());
    }
    state.pending_thread_metadata_modal = None;
    state.pending_thread_delete_confirm = Some(ThreadDeleteConfirmState {
        thread_index,
        selected: 1,
    });
    state.status = ui_text_thread_delete_confirm_prompt().to_string();
    Ok(())
}

fn open_thread_rename_modal(state: &mut ChatUiState, thread_index: usize) -> Result<(), AppError> {
    let Some(target) = state.threads.get(thread_index) else {
        state.status = ui_text_thread_not_found().to_string();
        return Ok(());
    };
    state.pending_thread_metadata_modal = None;
    let mut input = InputBuffer::new();
    input.text = target.session_name.clone();
    input.cursor_char = input.char_count();
    input.view_char_offset = 0;
    state.pending_thread_rename = Some(ThreadRenameModalState {
        thread_index,
        input,
    });
    state.status = ui_text_thread_rename_prompt().to_string();
    Ok(())
}

fn apply_thread_rename_modal(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    let Some(modal) = state.pending_thread_rename.as_ref() else {
        return Ok(());
    };
    let Some(target) = state.threads.get(modal.thread_index).cloned() else {
        state.pending_thread_rename = None;
        state.status = ui_text_thread_not_found().to_string();
        return Ok(());
    };
    let new_name = modal.input.text.trim().to_string();
    if new_name.is_empty() {
        state.status = ui_text_thread_rename_empty().to_string();
        return Ok(());
    }
    let was_active = target.session_id == services.session.session_id();
    let updated = services
        .session
        .rename_session_by_id(&target.session_id, &new_name)?;
    state.pending_thread_rename = None;
    state.set_threads(services.session.list_sessions().unwrap_or_default());
    if was_active {
        state.push(
            UiRole::System,
            i18n::chat_session_renamed(
                services.session.session_name(),
                services.session.session_id(),
            ),
        );
    }
    state.status = format!("{}: {}", ui_text_thread_renamed(), updated.session_name);
    Ok(())
}

fn delete_thread_session_by_index(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    thread_index: usize,
) -> Result<(), AppError> {
    let Some(target) = state.threads.get(thread_index).cloned() else {
        state.status = ui_text_thread_not_found().to_string();
        return Ok(());
    };
    state.remember_current_session_messages();
    let previous_active_id = services.session.session_id().to_string();
    let active_after = services.session.delete_session_by_id(&target.session_id)?;
    state.drop_session_cache(&target.session_id);
    state.session_auto_title_attempted.remove(&target.session_id);
    state
        .session_auto_title_workers
        .retain(|item| item.session_id != target.session_id);
    state.pending_thread_rename = None;
    state.pending_thread_action_menu = None;
    state.pending_thread_delete_confirm = None;
    state.pending_thread_metadata_modal = None;
    state.set_threads(services.session.list_sessions().unwrap_or_default());
    if previous_active_id != services.session.session_id() {
        state.set_active_session(active_after.session_id.clone());
        state.clear_conversation_viewport_only();
        state.push(
            UiRole::System,
            i18n::chat_session_changed(
                active_after.session_name.as_str(),
                active_after.session_id.as_str(),
                active_after.file_path.as_path(),
            ),
        );
        let current = services
            .session
            .recent_messages_for_display(CHAT_RENDER_LIMIT);
        let restored = state.session_messages_or_fallback(
            active_after.session_id.as_str(),
            recent_messages_to_ui_messages(&current),
        );
        for (item, persisted) in restored {
            state.push_message_with_persisted(item, persisted);
        }
        state.remember_current_session_messages();
    }
    state.status = format!("{}: {}", ui_text_thread_deleted(), target.session_name);
    Ok(())
}

fn open_thread_metadata_modal(
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    thread_index: usize,
) -> Result<(), AppError> {
    let Some(target) = state.threads.get(thread_index).cloned() else {
        state.status = ui_text_thread_not_found().to_string();
        return Ok(());
    };
    let detail = if target.session_id == services.session.session_id() {
        build_session_metadata_report(services, state)
    } else {
        build_thread_metadata_report(services, &target)
    };
    state.pending_thread_delete_confirm = None;
    state.pending_thread_rename = None;
    state.pending_thread_metadata_modal = Some(ThreadMetadataModalState {
        session_id: target.session_id.clone(),
        session_name: target.session_name.clone(),
        rows: metadata_report_to_rows(detail.as_str()),
        scroll: 0,
    });
    state.status = ui_text_thread_metadata_opened().to_string();
    Ok(())
}

fn activate_nav_item(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    state.hovered_message_idx = None;
    state.pending_delete_confirm = None;
    state.pending_thread_action_menu = None;
    state.pending_thread_rename = None;
    state.pending_thread_delete_confirm = None;
    state.pending_thread_metadata_modal = None;
    state.pending_tool_result_modal = None;
    match state.nav_selected {
        0 => {
            state.mode = UiMode::Chat;
            state.inspect_menu_open = false;
            state.skill_doc_modal = None;
            state.pending_choice = None;
            let _ = execute_builtin_command(BuiltinCommand::New, services, state)?;
            state.focus = FocusPanel::Input;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        1 => {
            state.mode = UiMode::Skills;
            state.inspect_menu_open = false;
            state.skill_doc_modal = None;
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_skills_panel_ready().to_string();
        }
        2 => {
            state.mode = UiMode::Mcp;
            state.inspect_menu_open = false;
            state.skill_doc_modal = None;
            state.mcp_ui = build_mcp_ui_state(services.cfg, services.config_path);
            refresh_mcp_runtime_metadata_if_needed(services, state, false);
            state.focus = FocusPanel::Conversation;
            state.mcp_ui.focus_servers = true;
            state.mcp_ui.editing = false;
            state.status = ui_text_mcp_panel_ready().to_string();
        }
        3 => {
            state.mode = UiMode::Inspect;
            state.skill_doc_modal = None;
            state.focus = FocusPanel::Conversation;
            state.inspect_menu_open = true;
            state.inspect_menu_selected = inspect_target_index(state.inspect.target);
            state.status = ui_text_inspect_target_menu_hint().to_string();
        }
        4 => {
            state.mode = UiMode::Config;
            state.inspect_menu_open = false;
            state.skill_doc_modal = None;
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_config_mode_ready().to_string();
        }
        _ => {}
    }
    Ok(())
}

fn apply_selected_inspect_target(state: &mut ChatUiState) {
    let targets = inspect_targets();
    if state.inspect_menu_selected >= targets.len() {
        state.inspect_menu_selected = 0;
    }
    state.inspect.target = targets[state.inspect_menu_selected];
    state.inspect_menu_open = false;
    state.mode = UiMode::Inspect;
    state.focus = FocusPanel::Conversation;
    state.status = format!(
        "{}: {}",
        ui_text_inspect_target_changed(),
        inspect_target_label(state.inspect.target)
    );
}

fn inspect_targets() -> &'static [InspectTarget] {
    static TARGETS: [InspectTarget; 10] = [
        InspectTarget::Cpu,
        InspectTarget::Memory,
        InspectTarget::Disk,
        InspectTarget::Os,
        InspectTarget::Process,
        InspectTarget::Filesystem,
        InspectTarget::Hardware,
        InspectTarget::Logs,
        InspectTarget::Network,
        InspectTarget::All,
    ];
    &TARGETS
}

fn inspect_target_index(target: InspectTarget) -> usize {
    inspect_targets()
        .iter()
        .position(|item| item.as_str() == target.as_str())
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn handle_choice_input_key(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    system_prompt: &str,
    state: &mut ChatUiState,
    chat_turns: &mut usize,
    last_assistant_reply: &mut String,
) -> Result<bool, AppError> {
    let Some(choice) = state.pending_choice.as_mut() else {
        return Ok(false);
    };
    match key.code {
        KeyCode::Esc => {
            state.pending_choice = None;
            state.status = ui_text_choice_cancelled().to_string();
        }
        KeyCode::Left | KeyCode::Up => {
            choice.selected = choice.selected.saturating_sub(1);
        }
        KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
            if !choice.options.is_empty() {
                choice.selected = (choice.selected + 1) % choice.options.len();
            }
        }
        KeyCode::Char(ch) if ch.is_ascii_digit() => {
            if let Some(idx) = ch
                .to_digit(10)
                .map(|v| v as usize)
                .and_then(|v| v.checked_sub(1))
                && idx < choice.options.len()
            {
                choice.selected = idx;
            }
        }
        KeyCode::Char(ch) => {
            let lower = ch.to_ascii_lowercase().to_string();
            if let Some((idx, _)) = choice
                .options
                .iter()
                .enumerate()
                .find(|(_, item)| item.to_ascii_lowercase().starts_with(&lower))
            {
                choice.selected = idx;
            }
        }
        KeyCode::Enter => {
            if choice.options.is_empty() {
                state.pending_choice = None;
                return Ok(false);
            }
            let picked = choice.options[choice.selected].clone();
            state.pending_choice = None;
            return submit_chat_message(
                terminal,
                services,
                system_prompt,
                state,
                chat_turns,
                last_assistant_reply,
                picked,
            );
        }
        _ => {}
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn submit_chat_message(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    system_prompt: &str,
    state: &mut ChatUiState,
    chat_turns: &mut usize,
    last_assistant_reply: &mut String,
    message: String,
) -> Result<bool, AppError> {
    state.mode = UiMode::Chat;
    state.pending_choice = None;
    if state.ai_live.is_some() {
        stage_blocked_pending_message(state, &message);
        mark_pending_ai_send_blocked(state);
        return Ok(false);
    }
    if let Some(command) = parse_builtin_command(&message) {
        if execute_builtin_command(command, services, state)? {
            return Ok(true);
        }
        return Ok(false);
    }
    if message.starts_with('/') {
        let unknown = message
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or_default();
        state.push(UiRole::System, i18n::chat_unknown_builtin_command(unknown));
        return Ok(false);
    }
    if state.ai_connectivity_checking {
        stage_blocked_pending_message(state, &message);
        state.status = ui_text_status_ai_connectivity_pending_send_blocked().to_string();
        let already_reported = state.messages.back().is_some_and(|item| {
            item.role == UiRole::System
                && item.text == ui_text_status_ai_connectivity_pending_send_blocked()
        });
        if !already_reported {
            state.push(
                UiRole::System,
                ui_text_status_ai_connectivity_pending_send_blocked(),
            );
        }
        return Ok(false);
    }
    *chat_turns = chat_turns.saturating_add(1);
    let group_id = Uuid::new_v4().to_string();
    services
        .session
        .add_user_message(message.clone(), Some(group_id.clone()));
    services.session.persist()?;
    maybe_start_session_auto_title_worker(services, state, &message);
    state.push_persisted(UiRole::User, message.clone());
    state.reset_live_token_estimate();
    state.add_live_token_estimate(estimate_tokens_from_text_delta(&message));
    state.status = ui_text_thinking().to_string();
    state.ai_live = Some(AiLiveState {
        started_at: Instant::now(),
        tool_calls: 0,
        last_tool_label: String::new(),
        cancel_requested: false,
    });
    draw_once(terminal, services, state)?;
    let history = services.session.build_chat_history();
    let external_mcp_tools = if services.cfg.ai.tools.mcp.enabled {
        services.mcp.external_tool_definitions()
    } else {
        Vec::new()
    };
    let policy = if should_require_tool_call(&message) {
        ToolUsePolicy::RequireAtLeastOne
    } else {
        ToolUsePolicy::Auto
    };
    let (event_tx, event_rx) = mpsc::channel::<PendingAiEvent>();
    let cancel_requested = Arc::new(AtomicBool::new(false));
    spawn_chat_worker(
        services.ai.clone(),
        history,
        system_prompt.to_string(),
        message.clone(),
        services.session.session_id().to_string(),
        policy,
        services.cfg.ai.chat.max_tool_rounds,
        services.cfg.ai.chat.max_total_tool_calls,
        services.cfg.ai.chat.stream_output,
        external_mcp_tools,
        cancel_requested.clone(),
        event_tx,
    );
    let wait_outcome = wait_chat_worker_result(
        terminal,
        services,
        state,
        &group_id,
        event_rx,
        cancel_requested.as_ref(),
    )?;
    if services.cfg.ai.chat.stream_output && !wait_outcome.saw_stream_events {
        state.push(UiRole::System, ui_text_stream_fallback_notice());
    }
    let response = wait_outcome.response;
    state.ai_live = None;
    state.conversation_dirty = true;
    match response {
        Ok(chat_result) => {
            if let Some(thinking) = chat_result.thinking.as_deref()
                && should_append_final_thinking(thinking, wait_outcome.thinking_rendered_in_ui)
            {
                services
                    .session
                    .add_thinking_message(thinking.trim().to_string(), Some(group_id.clone()));
                services.session.persist()?;
                state.push_persisted(UiRole::Thinking, thinking.trim());
            }
            if let Some(stop_reason) = chat_result.stop_reason {
                state.push(
                    UiRole::System,
                    format!("tool guard: {}", stop_reason.code()),
                );
            }
            let reply = chat_result.archived_content.trim().to_string();
            services
                .session
                .add_assistant_message(reply.clone(), Some(group_id.clone()));
            services.session.persist()?;
            let final_content_already_printed = !wait_outcome.last_round_content.trim().is_empty()
                && wait_outcome.last_round_content.trim() == reply.trim();
            if !final_content_already_printed {
                state.push_persisted(UiRole::Assistant, reply.clone());
            } else {
                state.mark_last_message_persisted_if_matches(UiRole::Assistant, &reply);
            }
            let measured_tokens = if chat_result.metrics.total_tokens > 0 {
                chat_result.metrics.total_tokens
            } else {
                state.token_live_estimate
            };
            state.commit_token_usage(measured_tokens);
            state.pending_choice = detect_pending_choice(&reply);
            if state.pending_choice.is_some() {
                state.status = ui_text_choice_ready().to_string();
            } else {
                state.status = ui_text_ready().to_string();
            }
            *last_assistant_reply = reply;
        }
        Err(err) => {
            state.push(UiRole::System, err);
            state.reset_live_token_estimate();
            state.status = ui_text_ai_failed().to_string();
        }
    }
    state.set_threads(services.session.list_sessions().unwrap_or_default());
    Ok(false)
}

fn stage_blocked_pending_message(state: &mut ChatUiState, message: &str) {
    state.input.text = message.to_string();
    state.input.cursor_char = state.input.char_count();
    state.input.view_char_offset = 0;
}

fn mark_pending_ai_send_blocked(state: &mut ChatUiState) {
    state.status = ui_text_status_ai_pending_send_blocked().to_string();
    let already_reported = state.messages.back().is_some_and(|item| {
        item.role == UiRole::System && item.text == ui_text_status_ai_pending_send_blocked()
    });
    if !already_reported {
        state.push(UiRole::System, ui_text_status_ai_pending_send_blocked());
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_chat_worker(
    ai: crate::ai::AiClient,
    history: Vec<crate::ai::ChatMessage>,
    system_prompt: String,
    message: String,
    session_id: String,
    policy: ToolUsePolicy,
    max_tool_rounds: usize,
    max_total_tool_calls: usize,
    stream_output: bool,
    external_mcp_tools: Vec<crate::ai::ExternalToolDefinition>,
    cancel_requested: Arc<AtomicBool>,
    event_tx: mpsc::Sender<PendingAiEvent>,
) {
    thread::spawn(move || {
        let tool_event_tx = event_tx.clone();
        let round_event_tx = event_tx.clone();
        let stream_event_tx = event_tx.clone();
        let result = ai.chat_with_shell_tool_with_debug_session(
            &history,
            &system_prompt,
            &message,
            policy,
            max_tool_rounds,
            max_total_tool_calls,
            stream_output,
            &external_mcp_tools,
            Some(cancel_requested.as_ref()),
            Some(session_id.as_str()),
            |tool_call| {
                let (reply_tx, reply_rx) = mpsc::channel::<String>();
                if tool_event_tx
                    .send(PendingAiEvent::ToolCall {
                        request: tool_call.clone(),
                        reply_tx,
                    })
                    .is_err()
                {
                    return json!({
                        "ok": false,
                        "error": "tool call channel closed",
                    })
                    .to_string();
                }
                match reply_rx.recv() {
                    Ok(payload) => payload,
                    Err(_) => json!({
                        "ok": false,
                        "error": "tool call response channel closed",
                    })
                    .to_string(),
                }
            },
            |round_event| {
                let _ = round_event_tx.send(PendingAiEvent::Round(round_event));
            },
            |stream_event: ChatStreamEvent| {
                let _ = stream_event_tx.send(PendingAiEvent::Stream(stream_event));
            },
        );
        let normalized = result.map_err(|err| i18n::localize_error(&err));
        let _ = event_tx.send(PendingAiEvent::Finished(normalized));
    });
}

fn wait_chat_worker_result(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    group_id: &str,
    event_rx: mpsc::Receiver<PendingAiEvent>,
    cancel_requested: &AtomicBool,
) -> Result<PendingAiWaitOutcome, AppError> {
    let mut finished: Option<Result<ChatToolResponse, String>> = None;
    let mut thinking_rendered_in_ui = false;
    let mut streamed_thinking_dirty = false;
    let mut streamed_thinking_last_flush = Instant::now();
    let mut last_round_content = String::new();
    let mut saw_stream_events = false;
    let mut last_idle_draw = Instant::now();
    while finished.is_none() {
        poll_session_auto_title_workers(services, state);
        let mut ui_dirty = false;
        let mut stream_draw_happened = false;
        loop {
            match event_rx.try_recv() {
                Ok(PendingAiEvent::Round(event)) => {
                    ui_dirty = true;
                    if let Some(thinking) = event.thinking.as_deref()
                        && !thinking.trim().is_empty()
                        && !event.streamed_thinking
                    {
                        let thinking_text = thinking.trim();
                        state.push_persisted(UiRole::Thinking, thinking_text);
                        services
                            .session
                            .add_thinking_message(thinking_text.to_string(), Some(group_id.to_string()));
                        services.session.persist()?;
                        thinking_rendered_in_ui = true;
                        state.add_live_token_estimate(estimate_tokens_from_text_delta(
                            thinking_text,
                        ));
                    }
                    if !event.content.trim().is_empty() {
                        last_round_content = event.content.trim().to_string();
                        if !event.streamed_content {
                            state.push(UiRole::Assistant, event.content.trim());
                            state.add_live_token_estimate(estimate_tokens_from_text_delta(
                                event.content.trim(),
                            ));
                        }
                    }
                }
                Ok(PendingAiEvent::Stream(event)) => match event.kind {
                    ChatStreamEventKind::Content => {
                        ui_dirty = true;
                        stream_draw_happened = true;
                        saw_stream_events = true;
                        state.add_live_token_estimate(estimate_tokens_from_text_delta(&event.text));
                        render_stream_event_typewriter(
                            terminal,
                            services,
                            state,
                            UiRole::Assistant,
                            &event.text,
                            cancel_requested,
                        )?;
                    }
                    ChatStreamEventKind::Thinking => {
                        ui_dirty = true;
                        stream_draw_happened = true;
                        saw_stream_events = true;
                        if !event.text.is_empty() {
                            thinking_rendered_in_ui = true;
                            services
                                .session
                                .append_or_add_thinking_chunk(&event.text, Some(group_id));
                            streamed_thinking_dirty = true;
                        }
                        state.add_live_token_estimate(estimate_tokens_from_text_delta(&event.text));
                        render_stream_event_typewriter(
                            terminal,
                            services,
                            state,
                            UiRole::Thinking,
                            &event.text,
                            cancel_requested,
                        )?;
                    }
                },
                Ok(PendingAiEvent::ToolCall { request, reply_tx }) => {
                    ui_dirty = true;
                    if let Some(live) = state.ai_live.as_mut() {
                        live.tool_calls = live.tool_calls.saturating_add(1);
                        live.last_tool_label = format_live_tool_label(&request.name);
                    }
                    let payload =
                        execute_tool_call_tui(terminal, services, group_id, &request, state);
                    let _ = reply_tx.send(payload);
                }
                Ok(PendingAiEvent::Finished(result)) => {
                    ui_dirty = true;
                    finished = Some(result);
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    ui_dirty = true;
                    finished = Some(Err(ui_text_ai_channel_closed().to_string()));
                    break;
                }
            }
        }
        if streamed_thinking_dirty
            && streamed_thinking_last_flush.elapsed()
                >= Duration::from_millis(THINKING_STREAM_PERSIST_INTERVAL_MS)
        {
            services.session.persist()?;
            let _ = state.mark_all_unpersisted_messages_by_role(UiRole::Thinking);
            streamed_thinking_dirty = false;
            streamed_thinking_last_flush = Instant::now();
        }
        if (!stream_draw_happened && ui_dirty) || last_idle_draw.elapsed() >= Duration::from_millis(120) {
            draw_once(terminal, services, state)?;
            last_idle_draw = Instant::now();
        }
        if finished.is_some() {
            break;
        }
        if !event::poll(Duration::from_millis(60))
            .map_err(|err| AppError::Command(format!("failed to poll input event: {err}")))?
        {
            continue;
        }
        let mut redraw_after_input = false;
        match event::read()
            .map_err(|err| AppError::Command(format!("failed to read event: {err}")))?
        {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                redraw_after_input = true;
                if handle_pending_ai_key_event(key, terminal, services, state, cancel_requested)? {
                    break;
                }
            }
            Event::Mouse(mouse) => {
                redraw_after_input = true;
                handle_pending_ai_mouse_event(mouse, terminal, services, state)?
            }
            Event::Resize(_, _) => {
                handle_tui_resize_redraw(terminal, services, state)?;
                last_idle_draw = Instant::now();
            }
            _ => {}
        }
        if redraw_after_input {
            draw_once(terminal, services, state)?;
            last_idle_draw = Instant::now();
        }
    }
    if streamed_thinking_dirty {
        services.session.persist()?;
        let _ = state.mark_all_unpersisted_messages_by_role(UiRole::Thinking);
    }
    Ok(PendingAiWaitOutcome {
        response: finished.unwrap_or_else(|| Err(ui_text_ai_channel_closed().to_string())),
        thinking_rendered_in_ui,
        last_round_content,
        saw_stream_events,
    })
}

fn should_append_final_thinking(final_thinking: &str, thinking_rendered_in_ui: bool) -> bool {
    !final_thinking.trim().is_empty() && !thinking_rendered_in_ui
}

fn render_stream_event_typewriter(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    role: UiRole,
    text: &str,
    cancel_requested: &AtomicBool,
) -> Result<(), AppError> {
    if text.is_empty() {
        return Ok(());
    }
    let mut batch = String::new();
    let mut batch_chars = 0usize;
    for ch in text.chars() {
        if cancel_requested.load(Ordering::SeqCst) {
            break;
        }
        batch.push(ch);
        batch_chars += 1;
        let should_flush = batch_chars >= STREAM_RENDER_BATCH_CHARS || ch == '\n';
        if should_flush {
            flush_stream_batch(
                terminal,
                services,
                state,
                role,
                batch.as_str(),
                cancel_requested,
            )?;
            batch.clear();
            batch_chars = 0;
        }
    }
    if !batch.is_empty() {
        flush_stream_batch(
            terminal,
            services,
            state,
            role,
            batch.as_str(),
            cancel_requested,
        )?;
    }
    Ok(())
}

fn flush_stream_batch(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    role: UiRole,
    chunk: &str,
    cancel_requested: &AtomicBool,
) -> Result<(), AppError> {
    if chunk.is_empty() {
        return Ok(());
    }
    state.push_stream_chunk(role, chunk, None);
    draw_once(terminal, services, state)?;
    pump_pending_ai_input_events(terminal, services, state, cancel_requested)
}

fn pump_pending_ai_input_events(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    cancel_requested: &AtomicBool,
) -> Result<(), AppError> {
    while event::poll(Duration::from_millis(0))
        .map_err(|err| AppError::Command(format!("failed to poll input event: {err}")))?
    {
        match event::read().map_err(|err| {
            AppError::Command(format!("failed to read input event while streaming: {err}"))
        })? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                let _ =
                    handle_pending_ai_key_event(key, terminal, services, state, cancel_requested)?;
            }
            Event::Mouse(mouse) => handle_pending_ai_mouse_event(mouse, terminal, services, state)?,
            Event::Resize(_, _) => handle_tui_resize_redraw(terminal, services, state)?,
            _ => {}
        }
        if cancel_requested.load(Ordering::SeqCst) {
            break;
        }
    }
    Ok(())
}

fn handle_pending_ai_key_event(
    key: KeyEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    cancel_requested: &AtomicBool,
) -> Result<bool, AppError> {
    if state.pending_delete_confirm.is_some() {
        return handle_delete_message_confirm_key(key, services, state);
    }
    if state.pending_thread_delete_confirm.is_some() {
        return handle_thread_delete_confirm_key(key, services, state);
    }
    if state.pending_thread_metadata_modal.is_some() {
        return handle_thread_metadata_modal_key(key, state);
    }
    if state.pending_tool_result_modal.is_some() {
        return handle_tool_result_modal_key(key, state);
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        cancel_requested.store(true, Ordering::SeqCst);
        if let Some(live) = state.ai_live.as_mut() {
            live.cancel_requested = true;
        }
        state.status = ui_text_status_ai_cancelling().to_string();
        return Ok(false);
    }
    if key.code == KeyCode::Esc {
        cancel_requested.store(true, Ordering::SeqCst);
        if let Some(live) = state.ai_live.as_mut() {
            live.cancel_requested = true;
        }
        state.status = ui_text_status_ai_cancelling().to_string();
        return Ok(false);
    }
    if key.code == KeyCode::F(2) {
        state.theme_idx = (state.theme_idx + 1) % palettes().len();
        let _ = save_theme_preferences(state.theme_idx, &state.prefs_path);
        state.status = format!(
            "{}: {}",
            ui_text_theme_switched(),
            palette_by_index(state.theme_idx).name
        );
        return Ok(false);
    }
    if key.code == KeyCode::Tab {
        state.cycle_focus_forward();
        return Ok(false);
    }
    if key.code == KeyCode::BackTab {
        state.cycle_focus_backward();
        return Ok(false);
    }
    match state.focus {
        FocusPanel::Nav => handle_pending_ai_nav_key(key, services, state),
        FocusPanel::Threads => handle_pending_ai_threads_key(key, state),
        FocusPanel::Conversation => handle_conversation_key(key, terminal, services, state),
        FocusPanel::Input => handle_pending_ai_input_key(key, state),
    }
}

fn handle_pending_ai_mouse_event(
    mouse: MouseEvent,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    let area = terminal
        .size()
        .map_err(|err| AppError::Command(format!("failed to get terminal size: {err}")))?;
    let layout = compute_layout(Rect::new(0, 0, area.width, area.height));
    refresh_conversation_cache_for_layout(state, layout);
    if state.pending_delete_confirm.is_some() {
        handle_delete_message_confirm_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_delete_confirm.is_some() {
        handle_thread_delete_confirm_mouse(mouse, services, state, layout)?;
        return Ok(());
    }
    if state.pending_thread_metadata_modal.is_some() {
        handle_thread_metadata_modal_mouse(mouse, state, layout)?;
        return Ok(());
    }
    if state.pending_tool_result_modal.is_some() {
        handle_tool_result_modal_mouse(mouse, state, layout)?;
        return Ok(());
    }
    if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
        if rect_contains(layout.nav, mouse.column, mouse.row) {
            let nav_idx = mouse.row.saturating_sub(layout.nav.y.saturating_add(1)) as usize;
            if nav_idx == 0 {
                state.focus = FocusPanel::Nav;
                state.nav_selected = 0;
                state.status = ui_text_status_ai_pending_session_switch_blocked().to_string();
                return Ok(());
            }
        }
        if rect_contains(layout.threads_body, mouse.column, mouse.row) {
            state.focus = FocusPanel::Threads;
            let row = mouse.row.saturating_sub(layout.threads_body.y) as usize;
            let mut clicked_valid_thread = false;
            if row < state.threads.len() {
                state.thread_selected = row;
                clicked_valid_thread = true;
            }
            if clicked_valid_thread && selected_thread_is_active_session(state) {
                state.mode = UiMode::Chat;
                state.inspect_menu_open = false;
                state.skill_doc_modal = None;
                state.status = ui_text_focus_hint(state.focus).to_string();
            } else {
                state.status = ui_text_status_ai_pending_session_switch_blocked().to_string();
            }
            return Ok(());
        }
    }
    handle_mouse_event(mouse, terminal, services, state)
}

fn handle_pending_ai_nav_key(
    key: KeyEvent,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => state.nav_selected = state.nav_selected.saturating_sub(1),
        KeyCode::Down => state.nav_selected = (state.nav_selected + 1).min(NAV_ITEMS_COUNT - 1),
        KeyCode::Enter => {
            if state.nav_selected == 0 {
                state.status = ui_text_status_ai_pending_session_switch_blocked().to_string();
            } else {
                activate_nav_item(services, state)?;
            }
        }
        KeyCode::Char('/') if state.mode == UiMode::Chat => {
            state.focus = FocusPanel::Input;
            state.input.clear();
            state.input.insert_char('/');
        }
        _ => {}
    }
    Ok(false)
}

fn handle_pending_ai_threads_key(
    key: KeyEvent,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match key.code {
        KeyCode::Up => {
            state.thread_selected = state.thread_selected.saturating_sub(1);
        }
        KeyCode::Down => {
            if !state.threads.is_empty() {
                state.thread_selected = (state.thread_selected + 1).min(state.threads.len() - 1);
            }
        }
        KeyCode::Enter => {
            if selected_thread_is_active_session(state) {
                state.mode = UiMode::Chat;
                state.inspect_menu_open = false;
                state.skill_doc_modal = None;
                state.status = ui_text_focus_hint(state.focus).to_string();
            } else {
                state.status = ui_text_status_ai_pending_session_switch_blocked().to_string();
            }
        }
        KeyCode::Char('/') if state.mode == UiMode::Chat => {
            state.focus = FocusPanel::Input;
            state.input.clear();
            state.input.insert_char('/');
        }
        _ => {}
    }
    Ok(false)
}

fn handle_pending_ai_input_key(key: KeyEvent, state: &mut ChatUiState) -> Result<bool, AppError> {
    if state.mode != UiMode::Chat {
        if key.code == KeyCode::Esc {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        return Ok(false);
    }
    match key.code {
        KeyCode::Esc => {
            state.input.clear();
        }
        KeyCode::Left => state.input.move_left(),
        KeyCode::Right => state.input.move_right(),
        KeyCode::Home => state.input.move_home(),
        KeyCode::End => state.input.move_end(),
        KeyCode::Backspace => state.input.backspace(),
        KeyCode::Delete => state.input.delete(),
        KeyCode::Up => {
            state.focus = FocusPanel::Conversation;
            state.status = ui_text_focus_hint(state.focus).to_string();
        }
        KeyCode::Enter => {
            if !state.input.text.trim().is_empty() {
                mark_pending_ai_send_blocked(state);
            }
        }
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                state.input.insert_char(ch);
            }
        }
        _ => {}
    }
    Ok(false)
}

fn detect_pending_choice(reply: &str) -> Option<PendingChoice> {
    let lowered = reply.to_ascii_lowercase();
    if lowered.contains("yes or no")
        || lowered.contains("y/n")
        || lowered.contains("[y/n")
        || reply.contains("是否继续")
    {
        return Some(PendingChoice {
            options: if lang_is_zh() {
                vec!["是".to_string(), "否".to_string()]
            } else {
                vec!["Yes".to_string(), "No".to_string()]
            },
            selected: 0,
        });
    }
    for marker in [
        zh_or_en("选项：", "Options:"),
        zh_or_en("选项:", "Options:"),
    ] {
        if let Some(idx) = reply.find(marker) {
            let raw = reply[idx + marker.len()..]
                .lines()
                .next()
                .unwrap_or_default();
            let options = parse_inline_options(raw);
            if options.len() >= 2 {
                return Some(PendingChoice {
                    options,
                    selected: 0,
                });
            }
        }
    }
    if let Some((start, end)) = reply.find('[').zip(reply.find(']'))
        && end > start + 2
    {
        let options = parse_inline_options(&reply[start + 1..end]);
        if options.len() >= 2 {
            return Some(PendingChoice {
                options,
                selected: 0,
            });
        }
    }
    None
}

fn parse_inline_options(raw: &str) -> Vec<String> {
    let mut out = Vec::<String>::new();
    for item in raw.split(['/', '|', ',', '，', '、']) {
        let token = item
            .trim()
            .trim_start_matches(|ch: char| {
                ch.is_ascii_digit() || ch == '.' || ch == ')' || ch == '('
            })
            .trim()
            .to_string();
        if token.is_empty() || token.len() > 20 {
            continue;
        }
        if !out.iter().any(|v| v.eq_ignore_ascii_case(&token)) {
            out.push(token);
        }
        if out.len() >= 6 {
            break;
        }
    }
    out
}

fn execute_builtin_command(
    command: BuiltinCommand,
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<bool, AppError> {
    match command {
        BuiltinCommand::Exit => return Ok(true),
        BuiltinCommand::Help => {
            state.push(UiRole::System, ui_text_help_commands());
        }
        BuiltinCommand::Stats => {
            let archived = services.session.archived_role_counts();
            let effective = services.session.effective_context_role_counts(true);
            let mcp_summary = services.mcp.summary();
            state.push(
                UiRole::System,
                i18n::chat_stats(
                    services.session.session_id(),
                    services.session.file_path(),
                    archived.total,
                    effective.total,
                    services.session.summary_len(),
                    services.cfg.session.recent_messages,
                    services.cfg.session.max_messages,
                    0,
                    services.os_name,
                    &services.cfg.ai.model,
                    services.skills.len(),
                    mcp_summary.as_str(),
                    archived.user,
                    archived.assistant,
                    archived.tool,
                    archived.system,
                    effective.user,
                    effective.assistant,
                    effective.tool,
                    effective.system,
                ),
            );
        }
        BuiltinCommand::Meta => {
            state.push(
                UiRole::System,
                build_session_metadata_report(services, state),
            );
        }
        BuiltinCommand::Skills => {
            let skills_text = if services.skills.is_empty() {
                zh_or_en("未检测到技能", "No detected skills").to_string()
            } else {
                format!(
                    "{}:\n{}",
                    zh_or_en("已检测技能", "Detected skills"),
                    services.skills.join("\n")
                )
            };
            state.push(UiRole::System, skills_text);
        }
        BuiltinCommand::Mcps => {
            state.push(UiRole::System, services.mcp.summary());
        }
        BuiltinCommand::New => {
            state.remember_current_session_messages();
            services.session.start_new_session_with_new_file()?;
            services.session.persist()?;
            state.set_active_session(services.session.session_id().to_string());
            state.clear_conversation_viewport_only();
            state.push(
                UiRole::System,
                i18n::chat_session_switched(
                    services.session.session_id(),
                    services.session.file_path(),
                ),
            );
            state.remember_current_session_messages();
            state.set_threads(services.session.list_sessions().unwrap_or_default());
        }
        BuiltinCommand::Clear => {
            state.clear_conversation_viewport_only();
            state.push(UiRole::System, i18n::chat_cleared());
        }
        BuiltinCommand::List => {
            let sessions = services.session.list_sessions()?;
            state.push(
                UiRole::System,
                i18n::chat_session_list_title(sessions.len()),
            );
            for item in sessions.iter().take(12) {
                let active = if item.active {
                    i18n::chat_session_list_active_yes()
                } else {
                    i18n::chat_session_list_active_no()
                };
                state.push(
                    UiRole::System,
                    format!("[{active}] {} ({})", item.session_name, item.session_id),
                );
            }
            state.set_threads(sessions);
        }
        BuiltinCommand::Change(query) => match services.session.switch_session_by_query(&query) {
            Ok(switched) => {
                state.remember_current_session_messages();
                state.set_active_session(switched.session_id.clone());
                state.clear_conversation_viewport_only();
                state.push(
                    UiRole::System,
                    i18n::chat_session_changed(
                        switched.session_name.as_str(),
                        switched.session_id.as_str(),
                        switched.file_path.as_path(),
                    ),
                );
                let current = services
                    .session
                    .recent_messages_for_display(CHAT_RENDER_LIMIT);
                let restored = state.session_messages_or_fallback(
                    switched.session_id.as_str(),
                    recent_messages_to_ui_messages(&current),
                );
                for (item, persisted) in restored {
                    state.push_message_with_persisted(item, persisted);
                }
                state.remember_current_session_messages();
                state.set_threads(services.session.list_sessions().unwrap_or_default());
                services.session.persist()?;
            }
            Err(err) => state.push(UiRole::System, i18n::localize_error(&err)),
        },
        BuiltinCommand::Name(new_name) => {
            services.session.rename_current_session(&new_name)?;
            services.session.persist()?;
            state.push(
                UiRole::System,
                i18n::chat_session_renamed(
                    services.session.session_name(),
                    services.session.session_id(),
                ),
            );
            state.set_threads(services.session.list_sessions().unwrap_or_default());
        }
        BuiltinCommand::History(limit) => {
            let items = services.session.recent_messages_for_display(limit);
            state.push(
                UiRole::System,
                i18n::chat_history_title(items.len(), limit, services.session.message_count()),
            );
            for message in items {
                let role = match message.role.as_str() {
                    "assistant" => UiRole::Assistant,
                    "thinking" => UiRole::Thinking,
                    "tool" => UiRole::Tool,
                    "system" => UiRole::System,
                    _ => UiRole::User,
                };
                state.push(role, message.content);
            }
        }
    }
    Ok(false)
}

fn execute_tool_call_tui(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &mut ActionServices<'_>,
    group_id: &str,
    tool_call: &ToolCallRequest,
    state: &mut ChatUiState,
) -> String {
    if tool_call.name != "run_shell_command" {
        if services.mcp.has_ai_tool(&tool_call.name) {
            return execute_mcp_tool_call_tui(services, group_id, tool_call, state);
        }
        let payload = json!({
            "ok": false,
            "error": format!("unsupported tool function: {}", tool_call.name),
        })
        .to_string();
        persist_tool_call_record(
            services,
            state,
            group_id,
            tool_call,
            &tool_call.arguments,
            payload.as_str(),
            build_tool_execution_meta(services, ToolExecutionMetaInput {
                tool_call_id: tool_call.id.as_str(),
                function_name: tool_call.name.as_str(),
                command: "",
                arguments: tool_call.arguments.as_str(),
                result_payload: payload.as_str(),
                mode: "",
                label: "",
                exit_code: None,
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        );
        return payload;
    }
    let parsed: ShellToolArgs = match mcp::parse_json_object_arguments(&tool_call.arguments)
        .and_then(|value| {
            serde_json::from_value::<ShellToolArgs>(value).map_err(|err| err.to_string())
        }) {
        Ok(value) => value,
        Err(err) => {
            let payload = json!({
                "ok": false,
                "error": format!("invalid tool arguments: {err}"),
                "raw_arguments": trim_tool_text(&tool_call.arguments, 300),
            })
            .to_string();
            persist_tool_call_record(
                services,
                state,
                group_id,
                tool_call,
                &tool_call.arguments,
                payload.as_str(),
                build_tool_execution_meta(services, ToolExecutionMetaInput {
                    tool_call_id: tool_call.id.as_str(),
                    function_name: tool_call.name.as_str(),
                    command: "",
                    arguments: tool_call.arguments.as_str(),
                    result_payload: payload.as_str(),
                    mode: "",
                    label: "",
                    exit_code: None,
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: false,
                }),
            );
            return payload;
        }
    };
    let command = parsed.command.trim().to_string();
    if command.is_empty() {
        let payload = json!({
            "ok": false,
            "error": "command is empty",
        })
        .to_string();
        persist_tool_call_record(
            services,
            state,
            group_id,
            tool_call,
            tool_call.arguments.as_str(),
            payload.as_str(),
            build_tool_execution_meta(services, ToolExecutionMetaInput {
                tool_call_id: tool_call.id.as_str(),
                function_name: tool_call.name.as_str(),
                command: "",
                arguments: tool_call.arguments.as_str(),
                result_payload: payload.as_str(),
                mode: "",
                label: "",
                exit_code: None,
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        );
        return payload;
    }
    let mode = if parsed
        .mode
        .as_deref()
        .unwrap_or("read")
        .eq_ignore_ascii_case("write")
    {
        CommandMode::Write
    } else {
        CommandMode::Read
    };
    let label = parsed
        .label
        .as_deref()
        .unwrap_or("chat_tool")
        .trim()
        .to_string();
    let mut effective_command = command;
    let needs_write_confirm = services.cfg.ai.tools.bash.write_cmd_run_confirm
        && (matches!(mode, CommandMode::Write)
            || looks_like_write_command_hint(&effective_command));
    if needs_write_confirm {
        let confirm_mode =
            parse_confirm_mode(services.cfg.ai.tools.bash.write_cmd_confirm_mode.as_str());
        match prompt_write_confirmation_in_tui(
            terminal,
            services,
            state,
            confirm_mode,
            &effective_command,
        ) {
            Ok(WriteDecision::Reject) => {
                let payload = json!({
                    "ok": false,
                    "error": i18n::command_write_denied_by_user(),
                    "blocked": true
                })
                .to_string();
                persist_tool_call_record(
                    services,
                    state,
                    group_id,
                    tool_call,
                    tool_call.arguments.as_str(),
                    payload.as_str(),
                    build_tool_execution_meta(services, ToolExecutionMetaInput {
                        tool_call_id: tool_call.id.as_str(),
                        function_name: tool_call.name.as_str(),
                        command: effective_command.as_str(),
                        arguments: tool_call.arguments.as_str(),
                        result_payload: payload.as_str(),
                        mode: if matches!(mode, CommandMode::Write) {
                            ui_text_mode_write()
                        } else {
                            ui_text_mode_read()
                        },
                        label: "",
                        exit_code: None,
                        duration_ms: 0,
                        timed_out: false,
                        interrupted: false,
                        blocked: true,
                    }),
                );
                return payload;
            }
            Ok(WriteDecision::ApproveSession) => {
                state.write_session_approved = true;
            }
            Ok(WriteDecision::Edit) => {
                match prompt_edit_command_in_tui(terminal, services, state, &effective_command) {
                    Ok(Some(edited)) => effective_command = edited,
                    Ok(None) => {
                        let payload = json!({
                            "ok": false,
                            "error": i18n::command_write_denied_by_user(),
                            "blocked": true
                        })
                        .to_string();
                        persist_tool_call_record(
                            services,
                            state,
                            group_id,
                            tool_call,
                            tool_call.arguments.as_str(),
                            payload.as_str(),
                            build_tool_execution_meta(services, ToolExecutionMetaInput {
                                tool_call_id: tool_call.id.as_str(),
                                function_name: tool_call.name.as_str(),
                                command: effective_command.as_str(),
                                arguments: tool_call.arguments.as_str(),
                                result_payload: payload.as_str(),
                                mode: if matches!(mode, CommandMode::Write) {
                                    ui_text_mode_write()
                                } else {
                                    ui_text_mode_read()
                                },
                                label: "",
                                exit_code: None,
                                duration_ms: 0,
                                timed_out: false,
                                interrupted: false,
                                blocked: true,
                            }),
                        );
                        return payload;
                    }
                    Err(err) => {
                        let payload = json!({
                            "ok": false,
                            "error": err.to_string(),
                        })
                        .to_string();
                        persist_tool_call_record(
                            services,
                            state,
                            group_id,
                            tool_call,
                            tool_call.arguments.as_str(),
                            payload.as_str(),
                            build_tool_execution_meta(services, ToolExecutionMetaInput {
                                tool_call_id: tool_call.id.as_str(),
                                function_name: tool_call.name.as_str(),
                                command: effective_command.as_str(),
                                arguments: tool_call.arguments.as_str(),
                                result_payload: payload.as_str(),
                                mode: if matches!(mode, CommandMode::Write) {
                                    ui_text_mode_write()
                                } else {
                                    ui_text_mode_read()
                                },
                                label: "",
                                exit_code: None,
                                duration_ms: 0,
                                timed_out: false,
                                interrupted: false,
                                blocked: false,
                            }),
                        );
                        return payload;
                    }
                }
            }
            Ok(WriteDecision::Approve) => {}
            Err(err) => {
                let payload = json!({
                    "ok": false,
                    "error": err.to_string(),
                })
                .to_string();
                persist_tool_call_record(
                    services,
                    state,
                    group_id,
                    tool_call,
                    tool_call.arguments.as_str(),
                    payload.as_str(),
                    build_tool_execution_meta(services, ToolExecutionMetaInput {
                        tool_call_id: tool_call.id.as_str(),
                        function_name: tool_call.name.as_str(),
                        command: effective_command.as_str(),
                        arguments: tool_call.arguments.as_str(),
                        result_payload: payload.as_str(),
                        mode: if matches!(mode, CommandMode::Write) {
                            ui_text_mode_write()
                        } else {
                            ui_text_mode_read()
                        },
                        label: "",
                        exit_code: None,
                        duration_ms: 0,
                        timed_out: false,
                        interrupted: false,
                        blocked: false,
                    }),
                );
                return payload;
            }
        }
    }
    let spec = CommandSpec {
        label: if label.is_empty() {
            "chat_tool".to_string()
        } else {
            label
        },
        command: effective_command.clone(),
        mode,
    };
    let running_message = format!(
        "{}: {} [{}]\n{}",
        ui_text_tool_running(),
        spec.label,
        if matches!(spec.mode, CommandMode::Write) {
            ui_text_mode_write()
        } else {
            ui_text_mode_read()
        },
        trim_tool_text(&spec.command, 280),
    );
    state.push_persisted(UiRole::Tool, running_message.clone());
    services
        .session
        .add_tool_message(running_message, Some(group_id.to_string()));
    if let Err(err) = services.session.persist() {
        report_tool_session_persist_failure(state, &err, ui_text_tool_running());
    }
    let timeout = Duration::from_secs(services.cfg.ai.chat.cmd_run_timout);
    let run_result = services
        .shell
        .run_with_timeout_skip_write_confirm(&spec, timeout);
    match run_result {
        Ok(result) => {
            let payload = format_tool_result_payload(&result);
            state.push(
                UiRole::Tool,
                format!(
                    "{}: ok={} exit={:?} timeout={} blocked={}",
                    ui_text_tool_finished(),
                    result.success,
                    result.exit_code,
                    result.timed_out,
                    result.blocked
                ),
            );
            let tool_meta = build_tool_execution_meta(services, ToolExecutionMetaInput {
                tool_call_id: tool_call.id.as_str(),
                function_name: tool_call.name.as_str(),
                command: effective_command.as_str(),
                arguments: tool_call.arguments.as_str(),
                result_payload: payload.as_str(),
                mode: result.mode.as_str(),
                label: result.label.as_str(),
                exit_code: result.exit_code,
                duration_ms: result.duration_ms,
                timed_out: result.timed_out,
                interrupted: result.interrupted,
                blocked: result.blocked,
            });
            persist_tool_call_record(
                services,
                state,
                group_id,
                tool_call,
                &tool_call.arguments,
                payload.as_str(),
                tool_meta.clone(),
            );
            attach_tool_meta_to_last_tool_message(state, &tool_meta);
            attach_tool_meta_to_recent_running_tool_message(state, &tool_meta);
            payload
        }
        Err(err) => {
            let payload = json!({
                "ok": false,
                "error": err.to_string(),
            })
            .to_string();
            state.push(UiRole::Tool, format!("{}: {}", ui_text_tool_error(), err));
            let tool_meta = build_tool_execution_meta(services, ToolExecutionMetaInput {
                tool_call_id: tool_call.id.as_str(),
                function_name: tool_call.name.as_str(),
                command: effective_command.as_str(),
                arguments: tool_call.arguments.as_str(),
                result_payload: payload.as_str(),
                mode: if matches!(mode, CommandMode::Write) {
                    ui_text_mode_write()
                } else {
                    ui_text_mode_read()
                },
                label: spec.label.as_str(),
                exit_code: None,
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: false,
            });
            persist_tool_call_record(
                services,
                state,
                group_id,
                tool_call,
                &tool_call.arguments,
                payload.as_str(),
                tool_meta.clone(),
            );
            attach_tool_meta_to_last_tool_message(state, &tool_meta);
            attach_tool_meta_to_recent_running_tool_message(state, &tool_meta);
            payload
        }
    }
}

fn prompt_write_confirmation_in_tui(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    mode: ConfirmMode,
    command: &str,
) -> Result<WriteDecision, AppError> {
    if matches!(mode, ConfirmMode::Deny) {
        return Ok(WriteDecision::Reject);
    }
    if matches!(mode, ConfirmMode::AllowSession) && state.write_session_approved {
        return Ok(WriteDecision::Approve);
    }
    if matches!(mode, ConfirmMode::Edit) {
        return Ok(WriteDecision::Edit);
    }
    let mut options = vec![WriteDecision::Approve, WriteDecision::Reject];
    if matches!(mode, ConfirmMode::AllowSession) {
        options.insert(1, WriteDecision::ApproveSession);
    }
    let mut selected = 0usize;
    loop {
        terminal
            .draw(|frame| {
                draw_ui(frame, services, state);
                draw_confirm_modal(frame, services, state, command, &options, selected);
            })
            .map_err(|err| AppError::Command(format!("failed to draw confirm modal: {err}")))?;
        if !event::poll(Duration::from_millis(60))
            .map_err(|err| AppError::Command(format!("failed to poll confirm event: {err}")))?
        {
            continue;
        }
        match event::read()
            .map_err(|err| AppError::Command(format!("failed to read confirm event: {err}")))?
        {
            Event::Key(key) if is_key_event_actionable(&key) => match key.code {
                KeyCode::Esc => return Ok(WriteDecision::Reject),
                KeyCode::Left => selected = selected.saturating_sub(1),
                KeyCode::Right | KeyCode::Tab => {
                    if !options.is_empty() {
                        selected = (selected + 1) % options.len();
                    }
                }
                KeyCode::Enter => return Ok(options[selected]),
                KeyCode::Char('y' | 'Y') => return Ok(WriteDecision::Approve),
                KeyCode::Char('n' | 'N') => return Ok(WriteDecision::Reject),
                KeyCode::Char('a' | 'A') if matches!(mode, ConfirmMode::AllowSession) => {
                    return Ok(WriteDecision::ApproveSession);
                }
                _ => {}
            },
            Event::Resize(_, _) => {
                handle_tui_resize_only(terminal)?;
            }
            _ => {}
        }
    }
}

fn prompt_edit_command_in_tui(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    original: &str,
) -> Result<Option<String>, AppError> {
    let mut edit = InputBuffer {
        text: original.to_string(),
        cursor_char: original.chars().count(),
        view_char_offset: 0,
    };
    loop {
        terminal
            .draw(|frame| {
                draw_ui(frame, services, state);
                draw_edit_modal(frame, services, state, &mut edit);
            })
            .map_err(|err| AppError::Command(format!("failed to draw edit modal: {err}")))?;
        if !event::poll(Duration::from_millis(60))
            .map_err(|err| AppError::Command(format!("failed to poll edit event: {err}")))?
        {
            continue;
        }
        match event::read()
            .map_err(|err| AppError::Command(format!("failed to read edit event: {err}")))?
        {
            Event::Key(key) if is_key_event_actionable(&key) => match key.code {
                KeyCode::Esc => return Ok(None),
                KeyCode::Enter => return Ok(Some(edit.text.trim().to_string())),
                KeyCode::Left => edit.move_left(),
                KeyCode::Right => edit.move_right(),
                KeyCode::Home => edit.move_home(),
                KeyCode::End => edit.move_end(),
                KeyCode::Backspace => edit.backspace(),
                KeyCode::Delete => edit.delete(),
                KeyCode::Char(ch) => {
                    if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                        edit.insert_char(ch);
                    }
                }
                _ => {}
            },
            Event::Resize(_, _) => {
                handle_tui_resize_only(terminal)?;
            }
            _ => {}
        }
    }
}

fn draw_confirm_modal(
    frame: &mut Frame<'_>,
    _services: &ActionServices<'_>,
    state: &ChatUiState,
    command: &str,
    options: &[WriteDecision],
    selected: usize,
) {
    let palette = palette_by_index(state.theme_idx);
    draw_modal_overlay(frame, palette);
    let modal = centered_rect(frame.area(), 74, 12);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_confirm_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, modal);
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(4),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    frame.render_widget(
        Paragraph::new(ui_text_confirm_message()).style(Style::default().fg(palette.text)),
        inner[0],
    );
    let preview = normalize_command_for_modal_preview(command, 220);
    frame.render_widget(
        Paragraph::new(preview)
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(palette.text))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_confirm_command_preview_title())
                    .border_style(Style::default().fg(palette.border)),
            ),
        inner[1],
    );
    let option_line = options
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            if idx == selected {
                format!("[{}]", write_decision_label(*item))
            } else {
                write_decision_label(*item).to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("   ");
    frame.render_widget(
        Paragraph::new(option_line).style(Style::default().fg(palette.accent)),
        inner[2],
    );
    frame.render_widget(
        Paragraph::new(ui_text_confirm_hint()).style(Style::default().fg(palette.muted)),
        inner[3],
    );
}

fn draw_edit_modal(
    frame: &mut Frame<'_>,
    _services: &ActionServices<'_>,
    state: &ChatUiState,
    edit: &mut InputBuffer,
) {
    let palette = palette_by_index(state.theme_idx);
    draw_modal_overlay(frame, palette);
    let modal = centered_rect(frame.area(), 76, 9);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_edit_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, modal);
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    frame.render_widget(
        Paragraph::new(ui_text_edit_hint()).style(Style::default().fg(palette.muted)),
        inner[0],
    );
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(palette.border_focus));
    frame.render_widget(input_block, inner[1]);
    let input_inner = inner[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let (visible, cursor_col) = project_input_view(edit, input_inner.width as usize);
    frame.render_widget(
        Paragraph::new(visible).style(Style::default().fg(palette.text)),
        input_inner,
    );
    frame.set_cursor_position((input_inner.x + cursor_col as u16, input_inner.y));
    frame.render_widget(
        Paragraph::new(ui_text_edit_footer()).style(Style::default().fg(palette.muted)),
        inner[2],
    );
}

fn draw_delete_message_confirm_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    palette: ThemePalette,
) {
    let Some(modal_state) = state.pending_delete_confirm.as_ref() else {
        return;
    };
    let Some(message) = state.messages.get(modal_state.message_index) else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = delete_message_modal_rect(compute_layout(frame.area()));
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_message_delete_modal_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(Color::Rgb(255, 118, 118)));
    frame.render_widget(outer, modal);
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    frame.render_widget(
        Paragraph::new(ui_text_message_delete_modal_question())
            .style(Style::default().fg(palette.text)),
        body[0],
    );
    frame.render_widget(
        Paragraph::new(trim_ui_text(message.text.trim(), 220))
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(palette.text))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_message_delete_preview_title())
                    .border_style(Style::default().fg(palette.border)),
            ),
        body[1],
    );
    let (confirm_rect, cancel_rect) = delete_message_modal_button_rects(modal);
    let confirm_style = if modal_state.selected == 0 {
        Style::default()
            .fg(Color::Rgb(255, 238, 238))
            .bg(Color::Rgb(168, 56, 56))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Rgb(255, 148, 148))
    };
    let cancel_style = if modal_state.selected == 1 {
        Style::default()
            .fg(palette.panel_bg)
            .bg(palette.border_focus)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette.text)
    };
    frame.render_widget(
        Paragraph::new(ui_text_message_delete_yes()).style(confirm_style),
        confirm_rect,
    );
    frame.render_widget(
        Paragraph::new(ui_text_message_delete_no()).style(cancel_style),
        cancel_rect,
    );
    frame.render_widget(
        Paragraph::new(ui_text_message_delete_modal_hint())
            .style(Style::default().fg(palette.muted)),
        body[3],
    );
}

fn draw_thread_delete_confirm_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(modal_state) = state.pending_thread_delete_confirm.as_ref() else {
        return;
    };
    let Some(target) = state.threads.get(modal_state.thread_index) else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = thread_delete_modal_rect(layout);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_thread_delete_modal_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(Color::Rgb(255, 118, 118)));
    frame.render_widget(outer, modal);
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    frame.render_widget(
        Paragraph::new(ui_text_thread_delete_modal_question())
            .style(Style::default().fg(palette.text)),
        body[0],
    );
    frame.render_widget(
        Paragraph::new(trim_ui_text(
            format!(
                "{} ({})",
                target.session_name,
                short_session_id(target.session_id.as_str())
            )
            .as_str(),
            220,
        ))
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(palette.text))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_thread_delete_preview_title())
                .border_style(Style::default().fg(palette.border)),
        ),
        body[1],
    );
    let (confirm_rect, cancel_rect) = thread_delete_modal_button_rects(modal);
    let confirm_style = if modal_state.selected == 0 {
        Style::default()
            .fg(Color::Rgb(255, 238, 238))
            .bg(Color::Rgb(168, 56, 56))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Rgb(255, 148, 148))
    };
    let cancel_style = if modal_state.selected == 1 {
        Style::default()
            .fg(palette.panel_bg)
            .bg(palette.border_focus)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette.text)
    };
    frame.render_widget(
        Paragraph::new(ui_text_thread_delete_yes()).style(confirm_style),
        confirm_rect,
    );
    frame.render_widget(
        Paragraph::new(ui_text_thread_delete_no()).style(cancel_style),
        cancel_rect,
    );
    frame.render_widget(
        Paragraph::new(ui_text_thread_delete_modal_hint())
            .style(Style::default().fg(palette.muted)),
        body[3],
    );
}

fn draw_thread_metadata_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(modal_state) = state.pending_thread_metadata_modal.as_ref() else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = thread_metadata_modal_rect(layout);
    frame.render_widget(Clear, modal);
    let pulse = status_pulse_on();
    let border = if pulse {
        palette.accent
    } else {
        palette.border_focus
    };
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(ui_text_thread_metadata_modal_title())
            .style(Style::default().bg(palette.panel_bg))
            .border_style(Style::default().fg(border)),
        modal,
    );
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    let head_line = format!(
        "{} · {}",
        trim_ui_text(
            modal_state.session_name.as_str(),
            body[0].width.max(1) as usize
        ),
        short_session_id(modal_state.session_id.as_str())
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                head_line,
                Style::default()
                    .fg(palette.text)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                ui_text_thread_metadata_modal_subtitle(),
                Style::default().fg(palette.muted),
            )),
        ]),
        body[0],
    );
    let table_visible = body[1].height.saturating_sub(3).max(1) as usize;
    let max_scroll = modal_state.rows.len().saturating_sub(table_visible);
    let scroll = modal_state.scroll.min(max_scroll);
    let rows = if modal_state.rows.is_empty() {
        vec![Row::new(vec![
            Cell::from(ui_text_na()),
            Cell::from(ui_text_na()),
        ])]
    } else {
        modal_state
            .rows
            .iter()
            .skip(scroll)
            .take(table_visible)
            .enumerate()
            .map(|(idx, (k, v))| {
                let row_style = if idx % 2 == 0 {
                    Style::default().fg(palette.text)
                } else {
                    Style::default().fg(Color::Rgb(196, 218, 255))
                };
                Row::new(vec![
                    Cell::from(trim_ui_text(k, 24))
                        .style(Style::default().fg(border).add_modifier(Modifier::BOLD)),
                    Cell::from(trim_ui_text(v, 220)).style(row_style),
                ])
            })
            .collect::<Vec<_>>()
    };
    let table = Table::new(rows, [Constraint::Length(24), Constraint::Min(20)])
        .header(
            Row::new(vec![
                Cell::from(ui_text_thread_metadata_field()),
                Cell::from(ui_text_thread_metadata_value()),
            ])
            .style(
                Style::default()
                    .fg(palette.text)
                    .bg(Color::Rgb(34, 52, 84))
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .column_spacing(1)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(palette.border)),
        );
    frame.render_widget(table, body[1]);
    let footer = if max_scroll == 0 {
        ui_text_thread_metadata_modal_hint().to_string()
    } else {
        format!(
            "{} · {}/{}",
            ui_text_thread_metadata_modal_hint(),
            scroll + 1,
            max_scroll + 1
        )
    };
    frame.render_widget(
        Paragraph::new(trim_ui_text(footer.as_str(), body[2].width.max(1) as usize))
            .style(Style::default().fg(palette.muted)),
        body[2],
    );
}

fn draw_tool_result_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(modal_state) = state.pending_tool_result_modal.as_ref() else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = tool_result_modal_rect(layout);
    frame.render_widget(Clear, modal);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(ui_text_message_tool_result_modal_title())
            .style(Style::default().bg(palette.panel_bg))
            .border_style(Style::default().fg(palette.border_focus)),
        modal,
    );
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(1)])
        .margin(1)
        .split(modal);
    let max_scroll = tool_result_modal_scroll_max(modal_state, layout);
    let scroll = modal_state.scroll.min(max_scroll);
    let content = modal_state.lines.join("\n");
    frame.render_widget(
        Paragraph::new(content)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .style(Style::default().fg(palette.text))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(palette.border)),
            ),
        body[0],
    );
    let hint = if max_scroll == 0 {
        ui_text_message_tool_result_modal_hint().to_string()
    } else {
        format!(
            "{} · {}/{}",
            ui_text_message_tool_result_modal_hint(),
            scroll + 1,
            max_scroll + 1
        )
    };
    frame.render_widget(
        Paragraph::new(trim_ui_text(hint.as_str(), body[1].width.max(1) as usize))
            .style(Style::default().fg(palette.muted)),
        body[1],
    );
}

fn draw_thread_action_menu_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(menu_state) = state.pending_thread_action_menu.as_ref() else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = thread_action_menu_rect(layout);
    frame.render_widget(Clear, modal);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_thread_action_menu_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, modal);
    let inner = modal.inner(ratatui::layout::Margin {
        horizontal: 2,
        vertical: 2,
    });
    let options = thread_action_menu_options();
    for (idx, label) in options.iter().enumerate() {
        let row = Rect {
            x: inner.x,
            y: inner.y.saturating_add(idx as u16),
            width: inner.width.max(1),
            height: 1,
        };
        let selected = menu_state.selected == idx;
        let style = if selected {
            Style::default()
                .fg(palette.text)
                .bg(palette.app_bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(palette.text)
        };
        frame.render_widget(
            Paragraph::new(format!("{}. {}", idx + 1, label)).style(style),
            row,
        );
    }
    let hint = Rect {
        x: modal.x.saturating_add(2),
        y: modal.y.saturating_add(modal.height.saturating_sub(2)),
        width: modal.width.saturating_sub(4).max(1),
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(ui_text_thread_action_menu_hint()).style(Style::default().fg(palette.muted)),
        hint,
    );
}

fn draw_thread_rename_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(modal_state) = state.pending_thread_rename.as_ref() else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let modal = thread_rename_modal_rect(layout);
    frame.render_widget(Clear, modal);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_thread_rename_modal_title())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, modal);
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .margin(1)
        .split(modal);
    frame.render_widget(
        Paragraph::new(ui_text_thread_rename_prompt()).style(Style::default().fg(palette.muted)),
        body[0],
    );
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(palette.border_focus));
    frame.render_widget(input_block, body[1]);
    let input_inner = body[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let (visible, cursor_col) = project_input_view(&modal_state.input, input_inner.width as usize);
    frame.render_widget(
        Paragraph::new(visible).style(Style::default().fg(palette.text)),
        input_inner,
    );
    if input_inner.width > 0 && input_inner.height > 0 {
        frame.set_cursor_position((input_inner.x + cursor_col as u16, input_inner.y));
    }
    frame.render_widget(
        Paragraph::new(ui_text_thread_rename_modal_hint())
            .style(Style::default().fg(palette.muted)),
        body[2],
    );
}

fn execute_mcp_tool_call_tui(
    services: &mut ActionServices<'_>,
    group_id: &str,
    tool_call: &ToolCallRequest,
    state: &mut ChatUiState,
) -> String {
    state.push(
        UiRole::Tool,
        format!("{}: {}", ui_text_mcp_tool_call(), tool_call.name),
    );
    let started = Instant::now();
    let outcome = services
        .mcp
        .call_ai_tool(&tool_call.name, &tool_call.arguments);
    let duration_ms = started.elapsed().as_millis();
    let (payload, exit_code) = match outcome {
        Ok(content) => {
            let exit_code = 200;
            (
                json!({
                    "ok": true,
                    "tool": tool_call.name,
                    "command": tool_call.name,
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                    "content": trim_tool_text(&content, MCP_TOOL_RESULT_CONTENT_MAX_CHARS),
                })
                .to_string(),
                exit_code,
            )
        }
        Err(err) => {
            let exit_code = mcp_exit_code_from_error(&err);
            (
                json!({
                    "ok": false,
                    "tool": tool_call.name,
                    "command": tool_call.name,
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                    "error": err.to_string(),
                    "troubleshooting": mcp_troubleshooting_hints_tui(),
                })
                .to_string(),
                exit_code,
            )
        }
    };
    let extracted_command = extract_shell_command_from_tool_args(tool_call.arguments.as_str());
    let effective_command = if extracted_command.trim().is_empty() {
        tool_call.name.clone()
    } else {
        extracted_command
    };
    let tool_meta = build_tool_execution_meta(services, ToolExecutionMetaInput {
        tool_call_id: tool_call.id.as_str(),
        function_name: tool_call.name.as_str(),
        command: effective_command.as_str(),
        arguments: tool_call.arguments.as_str(),
        result_payload: payload.as_str(),
        mode: "mcp",
        label: tool_call.name.as_str(),
        exit_code: Some(exit_code),
        duration_ms,
        timed_out: false,
        interrupted: false,
        blocked: false,
    });
    persist_tool_call_record(
        services,
        state,
        group_id,
        tool_call,
        &tool_call.arguments,
        payload.as_str(),
        tool_meta.clone(),
    );
    attach_tool_meta_to_last_tool_message(state, &tool_meta);
    attach_tool_meta_to_recent_running_tool_message(state, &tool_meta);
    payload
}

fn mcp_troubleshooting_hints_tui() -> Vec<&'static str> {
    vec![
        "Check ai.tools.mcp.enabled and confirm target MCP server is enabled in config.",
        "Verify MCP endpoint/auth/header settings; for HTTP mode prefer /mcp over legacy /sse paths.",
        "Run /mcps to inspect server and tool availability, then retry with valid JSON arguments.",
    ]
}

fn mcp_exit_code_from_error(err: &AppError) -> i32 {
    extract_http_status_code(err.to_string().as_str()).unwrap_or(500)
}

fn extract_http_status_code(raw: &str) -> Option<i32> {
    let marker_idx = raw.find("status=")?;
    let status = raw[marker_idx + "status=".len()..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if status.is_empty() {
        return None;
    }
    status.parse::<i32>().ok()
}

fn format_live_tool_label(name: &str) -> String {
    if name == "run_shell_command" {
        return "bash.run_shell_command".to_string();
    }
    if name.starts_with("mcp__") {
        return format!("mcp.{name}");
    }
    name.to_string()
}

fn parse_builtin_command(input: &str) -> Option<BuiltinCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let mut parts = trimmed[1..].splitn(2, char::is_whitespace);
    let name = parts.next()?.to_ascii_lowercase();
    let arg = parts.next().unwrap_or_default().trim().to_string();
    match name.as_str() {
        "exit" | "quit" => Some(BuiltinCommand::Exit),
        "help" => Some(BuiltinCommand::Help),
        "stats" => Some(BuiltinCommand::Stats),
        "meta" | "session" => Some(BuiltinCommand::Meta),
        "skills" => Some(BuiltinCommand::Skills),
        "mcps" => Some(BuiltinCommand::Mcps),
        "new" => Some(BuiltinCommand::New),
        "clear" => Some(BuiltinCommand::Clear),
        "list" => Some(BuiltinCommand::List),
        "change" if !arg.is_empty() => Some(BuiltinCommand::Change(arg)),
        "name" if !arg.is_empty() => Some(BuiltinCommand::Name(arg)),
        "history" => {
            let parsed = arg.parse::<usize>().ok().unwrap_or(DEFAULT_HISTORY_LIMIT);
            Some(BuiltinCommand::History(parsed.clamp(1, 200)))
        }
        _ => None,
    }
}

fn parse_confirm_mode(raw: &str) -> ConfirmMode {
    match raw.trim().to_ascii_lowercase().as_str() {
        "deny" => ConfirmMode::Deny,
        "edit" => ConfirmMode::Edit,
        "allow-session" => ConfirmMode::AllowSession,
        _ => ConfirmMode::AllowOnce,
    }
}

fn is_key_event_actionable(key: &KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn normalize_command_for_modal_preview(command: &str, max_len: usize) -> String {
    let collapsed = command.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return ui_text_na().to_string();
    }
    trim_tool_text(&collapsed, max_len)
}

fn write_decision_label(value: WriteDecision) -> &'static str {
    match value {
        WriteDecision::Reject => ui_text_decision_reject(),
        WriteDecision::Approve => ui_text_decision_approve(),
        WriteDecision::ApproveSession => ui_text_decision_approve_session(),
        WriteDecision::Edit => ui_text_decision_edit(),
    }
}

fn format_tool_result_payload(result: &CommandResult) -> String {
    json!({
        "ok": result.success,
        "label": result.label,
        "mode": result.mode,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
        "interrupted": result.interrupted,
        "blocked": result.blocked,
        "block_reason": trim_tool_text(&result.block_reason, 300),
        "stdout": trim_tool_text(result.stdout.trim(), 3000),
        "stderr": trim_tool_text(result.stderr.trim(), 2000),
    })
    .to_string()
}

fn persist_tool_call_record(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    group_id: &str,
    tool_call: &ToolCallRequest,
    raw_arguments: &str,
    payload: &str,
    tool_meta: ToolExecutionMeta,
) {
    services.session.add_tool_message_with_meta(
        format!(
            "tool_call_id={} function={} args={} result={}",
            tool_call.id,
            tool_call.name,
            trim_tool_text(raw_arguments, 400),
            trim_tool_text(payload, 2400)
        ),
        Some(group_id.to_string()),
        Some(tool_meta),
    );
    if let Err(err) = services.session.persist() {
        report_tool_session_persist_failure(state, &err, tool_call.name.as_str());
    }
}

fn report_tool_session_persist_failure(
    state: &mut ChatUiState,
    err: &AppError,
    stage: &str,
) {
    state.push(
        UiRole::System,
        format!(
            "{}: {} ({})。{}",
            ui_text_tool_persist_failed(),
            i18n::localize_error(err),
            stage,
            ui_text_tool_persist_failed_hint()
        ),
    );
    state.status = ui_text_tool_persist_failed().to_string();
}

fn attach_tool_meta_to_last_tool_message(state: &mut ChatUiState, tool_meta: &ToolExecutionMeta) {
    for message in state.messages.iter_mut().rev() {
        if message.role == UiRole::Tool {
            message.tool_meta = Some(tool_meta.clone());
            state.conversation_dirty = true;
            return;
        }
    }
}

fn attach_tool_meta_to_recent_running_tool_message(
    state: &mut ChatUiState,
    tool_meta: &ToolExecutionMeta,
) {
    let running_prefix = format!("{}:", ui_text_tool_running());
    for message in state.messages.iter_mut().rev() {
        if message.role != UiRole::Tool {
            continue;
        }
        if message.text.starts_with(running_prefix.as_str()) {
            message.tool_meta = Some(tool_meta.clone());
            state.conversation_dirty = true;
            return;
        }
    }
}

fn build_tool_execution_meta(
    services: &ActionServices<'_>,
    input: ToolExecutionMetaInput<'_>,
) -> ToolExecutionMeta {
    let masked_command = mask_ui_sensitive(input.command);
    let masked_arguments = mask_ui_sensitive(input.arguments);
    let masked_payload = mask_ui_sensitive(input.result_payload);
    let account = resolve_runtime_account();
    let cwd = resolve_runtime_cwd();
    ToolExecutionMeta {
        tool_call_id: input.tool_call_id.to_string(),
        function_name: input.function_name.to_string(),
        command: masked_command,
        arguments: masked_arguments,
        result_payload: masked_payload,
        executed_at_epoch_ms: now_epoch_ms(),
        account: trim_tool_text(account.as_str(), 120),
        environment: trim_tool_text(services.cfg.app.env_mode.as_str(), 64),
        os_name: trim_tool_text(services.os_name, 64),
        cwd: trim_tool_text(cwd.as_str(), 260),
        mode: trim_tool_text(input.mode, 32),
        label: trim_tool_text(input.label, 120),
        exit_code: input.exit_code,
        duration_ms: input.duration_ms,
        timed_out: input.timed_out,
        interrupted: input.interrupted,
        blocked: input.blocked,
    }
}

fn resolve_runtime_account() -> String {
    const ACCOUNT_ENV_KEYS: [&str; 3] = ["USER", "LOGNAME", "USERNAME"];
    ACCOUNT_ENV_KEYS
        .iter()
        .find_map(|key| {
            env::var(key)
                .ok()
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
        })
        .unwrap_or_default()
}

fn resolve_runtime_cwd() -> String {
    std::env::current_dir()
        .ok()
        .map(|item| item.display().to_string())
        .unwrap_or_default()
}

fn should_require_tool_call(message: &str) -> bool {
    const FORCE_TOOL_KEYWORDS: [&str; 12] = [
        "执行", "检查", "排查", "inspect", "check", "run", "execute", "memory", "cpu", "disk",
        "network", "process",
    ];
    let lowered = message.to_ascii_lowercase();
    FORCE_TOOL_KEYWORDS
        .iter()
        .any(|keyword| lowered.contains(&keyword.to_ascii_lowercase()))
}

fn collect_cpu_metrics_local(
    os_type: OsType,
    previous: &InspectMetrics,
    refresh_details: bool,
) -> InspectMetrics {
    let mut metrics = previous.clone();
    let (usage_cmd, info_cmd, temp_cmd) = match os_type {
        OsType::MacOS => (
            "top -l 1 -n 0 2>/dev/null | head -n 18",
            "printf 'model='; sysctl -n machdep.cpu.brand_string 2>/dev/null; printf '\nlogical='; sysctl -n hw.logicalcpu 2>/dev/null; printf '\nphysical='; sysctl -n hw.physicalcpu 2>/dev/null; printf '\nfreq='; sysctl -n hw.cpufrequency 2>/dev/null",
            "sysctl -a 2>/dev/null | grep -i -E 'cpu.*temp|temperature' | head -n 5",
        ),
        OsType::Linux => (
            "top -bn1 2>/dev/null | head -n 8",
            "printf 'model='; lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2-; printf '\nlogical='; nproc 2>/dev/null; printf '\nphysical='; lscpu 2>/dev/null | grep -m1 '^Core\\(s\\) per socket:' | awk -F: '{print $2}'; printf '\nfreq='; lscpu 2>/dev/null | grep -m1 'CPU MHz' | awk -F: '{print $2}'",
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null",
        ),
        OsType::Windows => (
            "powershell -NoProfile -Command \"Get-Counter '\\Processor(_Total)\\% Processor Time' | Select -Expand CounterSamples | Select -Expand CookedValue\"",
            "powershell -NoProfile -Command \"$cpu=Get-CimInstance Win32_Processor | Select -First 1; Write-Output ('model=' + $cpu.Name); Write-Output ('logical=' + $cpu.NumberOfLogicalProcessors); Write-Output ('physical=' + $cpu.NumberOfCores); Write-Output ('freq=' + $cpu.MaxClockSpeed)\"",
            "powershell -NoProfile -Command \"Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi | Select -First 1 -Expand CurrentTemperature\"",
        ),
        OsType::Other => (
            "top -bn1 2>/dev/null | head -n 8",
            "lscpu 2>/dev/null | head -n 30",
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null",
        ),
    };
    let usage_out =
        run_inspect_read_command_local("inspect_cpu_usage", usage_cmd, Duration::from_millis(350));
    let (user, system, idle) = parse_cpu_breakdown(&usage_out);
    metrics.user_percent = user;
    metrics.system_percent = system;
    metrics.idle_percent = idle;
    metrics.usage_percent = (100.0 - idle).clamp(0.0, 100.0);
    let mut info_out = String::new();
    let mut temp_out = String::new();
    if refresh_details {
        info_out = run_inspect_read_command_local(
            "inspect_cpu_info",
            info_cmd,
            Duration::from_millis(700),
        );
        temp_out = run_inspect_read_command_local(
            "inspect_cpu_temp",
            temp_cmd,
            Duration::from_millis(500),
        );
        if let Some(value) = parse_key_value(&info_out, "model") {
            replace_if_non_empty(&mut metrics.model, value);
        } else if let Some(value) = guess_lscpu_value(&info_out, "Model name") {
            replace_if_non_empty(&mut metrics.model, value);
        }
        if let Some(value) =
            parse_key_value(&info_out, "logical").or_else(|| guess_lscpu_value(&info_out, "CPU(s)"))
        {
            replace_if_non_empty(&mut metrics.logical_cores, value);
        }
        if let Some(value) = parse_key_value(&info_out, "physical")
            .or_else(|| guess_lscpu_value(&info_out, "Core(s) per socket"))
        {
            replace_if_non_empty(&mut metrics.physical_cores, value);
        }
        if let Some(value) =
            parse_key_value(&info_out, "freq").or_else(|| guess_lscpu_value(&info_out, "CPU MHz"))
        {
            replace_if_non_empty(&mut metrics.freq_mhz, normalize_frequency(&value));
        }
        if !temp_out.trim().is_empty() {
            replace_if_non_empty(&mut metrics.temperature, normalize_temperature(&temp_out));
        }
    }
    if metrics.model.trim().is_empty() {
        metrics.model = ui_text_na().to_string();
    }
    if metrics.logical_cores.trim().is_empty() {
        metrics.logical_cores = ui_text_na().to_string();
    }
    if metrics.physical_cores.trim().is_empty() {
        metrics.physical_cores = ui_text_na().to_string();
    }
    if metrics.freq_mhz.trim().is_empty() {
        metrics.freq_mhz = ui_text_na().to_string();
    }
    if metrics.temperature.trim().is_empty() {
        metrics.temperature = ui_text_na().to_string();
    }
    metrics.health = cpu_health_label(metrics.usage_percent).to_string();
    metrics.last_updated = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let mut raw = format!(
        "{}\n{}",
        ui_text_inspect_raw_usage(),
        trim_tool_text(usage_out.trim(), 1400),
    );
    if refresh_details {
        raw.push('\n');
        raw.push_str(ui_text_inspect_raw_info());
        raw.push('\n');
        raw.push_str(&trim_tool_text(info_out.trim(), 900));
        if !temp_out.trim().is_empty() {
            raw.push('\n');
            raw.push_str(ui_text_inspect_raw_temperature());
            raw.push('\n');
            raw.push_str(&trim_tool_text(temp_out.trim(), 300));
        }
    }
    metrics.raw_output = raw;
    metrics
}

fn collect_generic_inspect_metrics_local(os: OsType, target: InspectTarget) -> InspectMetrics {
    let mut metrics = InspectMetrics::default();
    let (label, cmd) = inspect_target_command(os, target);
    let out = run_inspect_read_command_local(label, cmd, Duration::from_secs(1));
    metrics.model = inspect_target_label(target).to_string();
    metrics.last_updated = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    metrics.health = ui_text_inspect_health_unknown().to_string();
    metrics.raw_output = trim_tool_text(out.trim(), 2400);
    metrics
}

fn inspect_target_command(os: OsType, target: InspectTarget) -> (&'static str, &'static str) {
    match os {
        OsType::Windows => match target {
            InspectTarget::Cpu => (
                "inspect_cpu",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed\"",
            ),
            InspectTarget::Memory => (
                "inspect_memory",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory | Format-List\"",
            ),
            InspectTarget::Disk => (
                "inspect_disk",
                "powershell -NoProfile -Command \"Get-PSDrive -PSProvider FileSystem | Format-Table -AutoSize\"",
            ),
            InspectTarget::Os => (
                "inspect_os",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version,BuildNumber\"",
            ),
            InspectTarget::Process => (
                "inspect_process",
                "powershell -NoProfile -Command \"Get-Process | Sort-Object CPU -Descending | Select -First 30 | Format-Table -AutoSize\"",
            ),
            InspectTarget::Filesystem => (
                "inspect_filesystem",
                "powershell -NoProfile -Command \"Get-Volume | Format-Table -AutoSize\"",
            ),
            InspectTarget::Hardware => (
                "inspect_hardware",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_ComputerSystem | Format-List Manufacturer,Model,TotalPhysicalMemory\"",
            ),
            InspectTarget::Logs => (
                "inspect_logs",
                "powershell -NoProfile -Command \"Get-WinEvent -LogName System -MaxEvents 20 | Format-Table -Wrap\"",
            ),
            InspectTarget::Network => (
                "inspect_network",
                "powershell -NoProfile -Command \"Get-NetIPConfiguration | Format-List\"",
            ),
            InspectTarget::All => (
                "inspect_all",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Format-List; Get-CimInstance Win32_Processor | Format-List; Get-PSDrive -PSProvider FileSystem | Format-Table -AutoSize\"",
            ),
        },
        _ => match target {
            InspectTarget::Cpu => (
                "inspect_cpu",
                "top -l 1 -n 0 2>/dev/null | head -n 18 || top -bn1 2>/dev/null | head -n 18",
            ),
            InspectTarget::Memory => (
                "inspect_memory",
                "vm_stat 2>/dev/null || free -h 2>/dev/null",
            ),
            InspectTarget::Disk => ("inspect_disk", "df -h"),
            InspectTarget::Os => ("inspect_os", "uname -a"),
            InspectTarget::Process => ("inspect_process", "ps aux | head -n 40"),
            InspectTarget::Filesystem => ("inspect_filesystem", "mount | head -n 40"),
            InspectTarget::Hardware => (
                "inspect_hardware",
                "system_profiler SPHardwareDataType 2>/dev/null | head -n 40 || lshw -short 2>/dev/null | head -n 40",
            ),
            InspectTarget::Logs => (
                "inspect_logs",
                "ls -lah /var/log 2>/dev/null | head -n 30 || journalctl -n 30 --no-pager 2>/dev/null",
            ),
            InspectTarget::Network => (
                "inspect_network",
                "ifconfig 2>/dev/null | head -n 70 || ip addr 2>/dev/null | head -n 70",
            ),
            InspectTarget::All => (
                "inspect_all",
                "uname -a; uptime; df -h | head -n 20; vm_stat 2>/dev/null | head -n 20 || free -h 2>/dev/null; ifconfig 2>/dev/null | head -n 40 || ip addr 2>/dev/null | head -n 40",
            ),
        },
    }
}

fn run_inspect_read_command_local(label: &str, cmd: &str, timeout: Duration) -> String {
    let spawn_result = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", cmd])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    } else {
        std::process::Command::new("sh")
            .args(["-lc", cmd])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    };
    let mut child = match spawn_result {
        Ok(proc) => proc,
        Err(err) => return format!("[{label}] {err}"),
    };
    match child.wait_timeout(timeout) {
        Ok(Some(_)) => match child.wait_with_output() {
            Ok(output) => {
                let mut out = String::new();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stdout.trim().is_empty() {
                    out.push_str(stdout.trim());
                }
                if !stderr.trim().is_empty() {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(stderr.trim());
                }
                if out.is_empty() {
                    out = ui_text_inspect_no_data().to_string();
                }
                out
            }
            Err(err) => format!("[{label}] {err}"),
        },
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
            format!("[{label}] {}", ui_text_inspect_timeout_hint())
        }
        Err(err) => format!("[{label}] {err}"),
    }
}

fn replace_if_non_empty(dst: &mut String, next: String) {
    if !next.trim().is_empty() {
        *dst = next;
    }
}

fn parse_cpu_breakdown(raw: &str) -> (f64, f64, f64) {
    let mut user = None;
    let mut sys = None;
    let mut idle = None;
    for line in raw.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("cpu usage") {
            user = parse_number_before(line, "% user");
            sys = parse_number_before(line, "% sys");
            idle = parse_number_before(line, "% idle");
            break;
        }
        if lower.contains("%cpu(s)") || lower.starts_with("cpu(s)") {
            user = parse_number_before(line, " us");
            sys = parse_number_before(line, " sy");
            idle = parse_number_before(line, " id");
            break;
        }
    }
    let user = user.unwrap_or(0.0).clamp(0.0, 100.0);
    let sys = sys.unwrap_or(0.0).clamp(0.0, 100.0);
    let mut idle = idle.unwrap_or((100.0 - user - sys).clamp(0.0, 100.0));
    if idle > 100.0 {
        idle = 100.0;
    }
    (user, sys, idle)
}

fn parse_number_before(line: &str, marker: &str) -> Option<f64> {
    let idx = line
        .to_ascii_lowercase()
        .find(&marker.to_ascii_lowercase())?;
    let prefix = line[..idx].trim_end().trim_end_matches(',');
    let token = prefix
        .split_whitespace()
        .last()?
        .trim()
        .trim_end_matches('%')
        .trim_end_matches(',');
    token.parse::<f64>().ok()
}

fn parse_key_value(raw: &str, key: &str) -> Option<String> {
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&format!("{key}=")) {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn guess_lscpu_value(raw: &str, key: &str) -> Option<String> {
    for line in raw.lines() {
        if line.contains(key) {
            let value = line.split(':').nth(1).unwrap_or_default().trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn normalize_frequency(raw: &str) -> String {
    let text = raw.trim();
    if text.is_empty() {
        return ui_text_na().to_string();
    }
    if let Ok(value) = text.parse::<f64>() {
        if value > 100_000.0 {
            return format!("{:.0} MHz", value / 1_000_000.0);
        }
        return format!("{value:.0} MHz");
    }
    format!("{} MHz", trim_ui_text(text, 18))
}

fn normalize_temperature(raw: &str) -> String {
    let text = raw.trim();
    if text.is_empty() {
        return ui_text_na().to_string();
    }
    if let Ok(value) = text.parse::<f64>() {
        if value > 1000.0 {
            return format!("{:.1} C", value / 1000.0);
        }
        if value > 0.0 {
            return format!("{value:.1} C");
        }
    }
    if let Some(num) = text
        .split(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .find(|item| !item.trim().is_empty())
        .and_then(|item| item.parse::<f64>().ok())
    {
        if num > 1000.0 {
            return format!("{:.1} C", num / 1000.0);
        }
        return format!("{num:.1} C");
    }
    ui_text_na().to_string()
}

fn cpu_health_label(usage: f64) -> &'static str {
    if usage >= 90.0 {
        ui_text_inspect_health_high()
    } else if usage >= 70.0 {
        ui_text_inspect_health_warn()
    } else {
        ui_text_inspect_health_ok()
    }
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_millis())
        .unwrap_or_default()
}

fn mask_ui_sensitive(raw: &str) -> String {
    mask::mask_sensitive(raw)
}

fn draw_ui(frame: &mut Frame<'_>, services: &ActionServices<'_>, state: &mut ChatUiState) {
    state.tick_token_display_animation();
    let palette = palette_by_index(state.theme_idx);
    let layout = compute_layout(frame.area());
    state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
    state.ensure_conversation_cache();
    if state.mode == UiMode::Chat {
        state.clamp_scroll(layout.conversation_body.height.max(1));
    }
    let root = Block::default().style(Style::default().bg(palette.app_bg));
    frame.render_widget(root, frame.area());
    draw_sidebar(frame, services, state, layout, palette);
    draw_main_panel(frame, services, state, layout, palette);
    if state.mode == UiMode::Skills {
        draw_skill_doc_modal(frame, state, layout, palette);
    }
    if state.inspect_menu_open {
        draw_inspect_target_menu(frame, state, layout, palette);
    }
    if state.pending_delete_confirm.is_some() {
        draw_delete_message_confirm_modal(frame, state, palette);
    }
    if state.pending_thread_delete_confirm.is_some() {
        draw_thread_delete_confirm_modal(frame, state, layout, palette);
    }
    if state.pending_thread_action_menu.is_some() {
        draw_thread_action_menu_modal(frame, state, layout, palette);
    }
    if state.pending_thread_rename.is_some() {
        draw_thread_rename_modal(frame, state, layout, palette);
    }
    if state.pending_thread_metadata_modal.is_some() {
        draw_thread_metadata_modal(frame, state, layout, palette);
    }
    if state.pending_tool_result_modal.is_some() {
        draw_tool_result_modal(frame, state, layout, palette);
    }
}

fn draw_sidebar(
    frame: &mut Frame<'_>,
    _services: &ActionServices<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let sidebar = Block::default()
        .style(Style::default().bg(palette.sidebar_bg))
        .borders(Borders::RIGHT)
        .border_style(Style::default().fg(palette.border));
    frame.render_widget(sidebar, layout.sidebar);

    let nav_items = vec![
        ListItem::new(Line::from(format!("  {}", ui_text_nav_new_thread()))),
        ListItem::new(Line::from(format!("  {}", ui_text_nav_skills()))),
        ListItem::new(Line::from(format!("  {}", ui_text_nav_mcp()))),
        ListItem::new(Line::from(format!("  {}", ui_text_nav_inspect()))),
        ListItem::new(Line::from(format!("  {}", ui_text_nav_config()))),
    ];
    let mut nav_state = ListState::default();
    nav_state.select(Some(
        state.nav_selected.min(NAV_ITEMS_COUNT.saturating_sub(1)),
    ));
    frame.render_stateful_widget(
        List::new(nav_items)
            .style(Style::default().fg(palette.muted))
            .highlight_style(
                Style::default()
                    .fg(palette.text)
                    .bg(palette.panel_bg)
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .title(format!(
                        "{} · {}",
                        ui_text_app_name(),
                        ui_mode_label(state.mode)
                    ))
                    .borders(Borders::BOTTOM)
                    .border_style(
                        Style::default()
                            .fg(panel_border_color(state.focus == FocusPanel::Nav, palette)),
                    ),
            ),
        layout.nav,
        &mut nav_state,
    );

    let mut thread_state = ListState::default();
    if !state.threads.is_empty() {
        thread_state.select(Some(state.thread_selected.min(state.threads.len() - 1)));
    }
    let thread_items = if state.threads.is_empty() {
        vec![ListItem::new(Line::from(format!(
            "  {}",
            ui_text_no_sessions()
        )))]
    } else {
        state
            .threads
            .iter()
            .take(layout.threads_body.height as usize)
            .map(|item| {
                ListItem::new(Line::from(format!(
                    "  {}",
                    trim_ui_text(&item.session_name, 26)
                )))
            })
            .collect::<Vec<_>>()
    };
    let mut threads_block = Block::default()
        .title(ui_text_panel_threads())
        .borders(Borders::TOP)
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Threads,
            palette,
        )));
    if state.focus == FocusPanel::Threads {
        threads_block = threads_block.style(Style::default().bg(palette.panel_bg));
    }
    frame.render_stateful_widget(
        List::new(thread_items)
            .style(Style::default().fg(palette.muted))
            .highlight_style(
                Style::default()
                    .fg(palette.text)
                    .bg(palette.panel_bg)
                    .add_modifier(Modifier::BOLD),
            )
            .block(threads_block),
        layout.threads,
        &mut thread_state,
    );

    frame.render_widget(
        Paragraph::new(format!(
            "{}  {}  {}",
            ui_text_footer_help(),
            ui_text_focus_hint(state.focus),
            ui_text_theme_hint()
        ))
        .style(Style::default().fg(palette.muted))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(palette.border)),
        ),
        layout.footer,
    );
}

fn draw_main_panel(
    frame: &mut Frame<'_>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    draw_main_header(frame, services, state, layout, palette);

    if state.mode == UiMode::Inspect {
        draw_inspect_panel(frame, state, layout, palette);
        draw_status_bar(frame, state, layout, palette);
        return;
    }
    if state.mode == UiMode::Skills {
        draw_skills_panel(frame, services, state, layout, palette);
        draw_status_bar(frame, state, layout, palette);
        return;
    }
    if state.mode == UiMode::Mcp {
        draw_mcp_panel(frame, services, state, layout, palette);
        draw_input_panel(frame, state, layout, palette);
        draw_status_bar(frame, state, layout, palette);
        return;
    }
    if state.mode == UiMode::Config {
        draw_config_panel(frame, state, layout, palette);
        draw_input_panel(frame, state, layout, palette);
        draw_status_bar(frame, state, layout, palette);
        return;
    }

    let conversation = Paragraph::new(conversation_text(state, palette))
        .scroll((state.conversation_scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_panel_conversation())
                .border_style(Style::default().fg(panel_border_color(
                    state.focus == FocusPanel::Conversation,
                    palette,
                ))),
        );
    frame.render_widget(Clear, layout.conversation);
    frame.render_widget(conversation, layout.conversation);
    draw_conversation_hover_copy_button(frame, state, layout, palette);

    draw_input_panel(frame, state, layout, palette);
    draw_status_bar(frame, state, layout, palette);
}

fn draw_main_header(
    frame: &mut Frame<'_>,
    services: &ActionServices<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let inner = layout.header.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 0,
    });
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let cols = if state.mode == UiMode::Chat {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(61), Constraint::Percentage(39)])
            .split(inner)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)])
            .split(inner)
    };

    let (left_title, left_subtitle) = match state.mode {
        UiMode::Inspect => (
            format!(
                "{} · {}",
                ui_text_inspect_panel_title(),
                inspect_target_label(state.inspect.target)
            ),
            ui_text_inspect_target_menu_hint().to_string(),
        ),
        UiMode::Skills => (
            ui_text_skills_panel_title().to_string(),
            format!("{}: {}", ui_text_skills_count(), services.skills.len()),
        ),
        UiMode::Mcp => (
            ui_text_mcp_panel_title().to_string(),
            format!(
                "{}: {}",
                ui_text_mcp_servers_count(),
                state.mcp_ui.servers.len()
            ),
        ),
        UiMode::Config => (
            ui_text_config_panel_title().to_string(),
            services.config_path.display().to_string(),
        ),
        UiMode::Chat => (
            services.session.session_name().to_string(),
            format!("{}: {}", ui_text_model_label(), services.cfg.ai.model),
        ),
    };
    let pulse_on = status_pulse_on();
    let accent_icon = if pulse_on { "✦" } else { "✧" };
    let left_meta = if state.mode == UiMode::Chat {
        format!(
            "{} {} · {}",
            accent_icon,
            ui_mode_label(state.mode),
            short_session_id(services.session.session_id())
        )
    } else {
        format!("{} {}", accent_icon, ui_mode_label(state.mode))
    };
    let meta_color = if pulse_on {
        Color::Rgb(122, 212, 255)
    } else {
        Color::Rgb(164, 126, 255)
    };
    let left_width = cols[0].width.saturating_sub(2) as usize;
    let left_lines = vec![
        Line::from(Span::styled(
            trim_ui_text(&left_title, left_width),
            Style::default()
                .fg(palette.text)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            trim_ui_text(&left_subtitle, left_width),
            Style::default().fg(palette.muted),
        )),
        Line::from(Span::styled(
            trim_ui_text(&left_meta, left_width),
            Style::default().fg(meta_color).add_modifier(Modifier::BOLD),
        )),
    ];
    frame.render_widget(
        Paragraph::new(left_lines).block(Block::default().borders(Borders::ALL).border_style(
            Style::default().fg(panel_border_color(
                state.focus == FocusPanel::Conversation || state.focus == FocusPanel::Input,
                palette,
            )),
        )),
        cols[0],
    );

    if state.mode != UiMode::Chat {
        return;
    }
    if cols.len() < 2 || cols[1].width < 18 {
        return;
    }
    let current = current_thread_overview(state, services);
    let summary = if let Some(item) = current {
        format!(
            "{}  msg:{}  user:{}  ai:{}",
            short_session_id(item.session_id.as_str()),
            item.message_count,
            item.user_count,
            item.assistant_count
        )
    } else {
        format!(
            "{}  msg:{}",
            short_session_id(services.session.session_id()),
            services.session.message_count()
        )
    };
    let updated = current
        .map(|item| format_epoch_ms(item.last_updated_epoch_ms))
        .unwrap_or_else(|| ui_text_na().to_string());
    let token_live_hint = if state.ai_live.is_some() && state.token_live_estimate > 0 {
        format!(" (+{})", format_u64_compact(state.token_live_estimate))
    } else {
        String::new()
    };
    let token_prefix = state
        .ai_live
        .as_ref()
        .map(|live| format!("{} ", spinner_frame(live.started_at)))
        .unwrap_or_default();
    let token_line = format!(
        "{}{}: {}{}",
        token_prefix,
        ui_text_session_card_tokens(),
        format_u64_compact(state.token_display_value),
        token_live_hint
    );
    let right_width = cols[1].width.saturating_sub(2) as usize;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{} ", ui_text_session_card_title()),
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    trim_ui_text(&summary, right_width),
                    Style::default().fg(palette.text),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("{} ", ui_text_session_card_updated()),
                    Style::default().fg(palette.muted),
                ),
                Span::styled(
                    trim_ui_text(&updated, right_width),
                    Style::default().fg(palette.text),
                ),
            ]),
            Line::from(Span::styled(
                trim_ui_text(&token_line, right_width),
                Style::default()
                    .fg(Color::Rgb(255, 212, 127))
                    .add_modifier(Modifier::BOLD),
            )),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(palette.accent)),
        ),
        cols[1],
    );
}

fn conversation_text(state: &ChatUiState, palette: ThemePalette) -> Text<'static> {
    let mut lines = Vec::<Line<'static>>::new();
    for line in &state.conversation_lines {
        match line.kind {
            ConversationLineKind::Header => {
                lines.push(Line::from(Span::styled(
                    line.text.clone(),
                    Style::default()
                        .fg(role_tag_color(line.role, palette))
                        .add_modifier(Modifier::BOLD),
                )));
            }
            ConversationLineKind::BubbleBorder => {
                lines.push(Line::from(Span::styled(
                    line.text.clone(),
                    Style::default().fg(role_border_color(line.role, palette)),
                )));
            }
            ConversationLineKind::Body => {
                lines.push(styled_bubble_body_line(
                    line.text.as_str(),
                    line.role,
                    palette,
                ));
            }
            ConversationLineKind::Spacer => lines.push(Line::from(String::new())),
        }
    }
    Text::from(lines)
}

fn styled_bubble_body_line(text: &str, role: UiRole, palette: ThemePalette) -> Line<'static> {
    let border_char = '│';
    let Some(left_idx) = text.find(border_char) else {
        return Line::from(Span::styled(
            text.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ));
    };
    let Some(right_idx) = text.rfind(border_char) else {
        return Line::from(Span::styled(
            text.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ));
    };
    if left_idx >= right_idx {
        return Line::from(Span::styled(
            text.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ));
    }
    let border_len = border_char.len_utf8();
    let leading = &text[..left_idx];
    let middle = &text[left_idx + border_len..right_idx];
    let trailing = &text[right_idx + border_len..];
    Line::from(vec![
        Span::styled(
            leading.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ),
        Span::styled(
            border_char.to_string(),
            Style::default().fg(role_border_color(role, palette)),
        ),
        Span::styled(
            middle.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ),
        Span::styled(
            border_char.to_string(),
            Style::default().fg(role_border_color(role, palette)),
        ),
        Span::styled(
            trailing.to_string(),
            Style::default().fg(role_body_color(role, palette)),
        ),
    ])
}

fn draw_conversation_hover_copy_button(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(buttons) = conversation_hover_buttons_data(state, layout) else {
        return;
    };
    let copy_label = ui_text_message_copy_button();
    frame.render_widget(
        Paragraph::new(copy_label).style(
            Style::default()
                .fg(palette.text)
                .bg(palette.panel_bg)
                .add_modifier(Modifier::BOLD),
        ),
        buttons.copy_rect,
    );
    if let Some(result_rect) = buttons.result_rect {
        let result_label = ui_text_message_tool_result_button();
        frame.render_widget(
            Paragraph::new(result_label).style(
                Style::default()
                    .fg(Color::Rgb(130, 205, 255))
                    .bg(palette.panel_bg)
                    .add_modifier(Modifier::BOLD),
            ),
            result_rect,
        );
    }
    let delete_label = ui_text_message_delete_button();
    frame.render_widget(
        Paragraph::new(delete_label).style(
            Style::default()
                .fg(Color::Rgb(255, 148, 148))
                .bg(palette.panel_bg)
                .add_modifier(Modifier::BOLD),
        ),
        buttons.delete_rect,
    );
}

fn conversation_copy_button_data(
    state: &ChatUiState,
    layout: UiLayout,
) -> Option<ConversationCopyButton> {
    let buttons = conversation_hover_buttons_data(state, layout)?;
    Some(ConversationCopyButton {
        message_index: buttons.message_index,
        rect: buttons.copy_rect,
    })
}

fn conversation_delete_button_data(
    state: &ChatUiState,
    layout: UiLayout,
) -> Option<ConversationDeleteButton> {
    let buttons = conversation_hover_buttons_data(state, layout)?;
    Some(ConversationDeleteButton {
        message_index: buttons.message_index,
        rect: buttons.delete_rect,
    })
}

fn conversation_tool_result_button_data(
    state: &ChatUiState,
    layout: UiLayout,
) -> Option<ConversationToolResultButton> {
    let buttons = conversation_hover_buttons_data(state, layout)?;
    let result_rect = buttons.result_rect?;
    Some(ConversationToolResultButton {
        message_index: buttons.message_index,
        rect: result_rect,
    })
}

fn conversation_hover_buttons_data(
    state: &ChatUiState,
    layout: UiLayout,
) -> Option<ConversationHoverButtons> {
    let hovered = state.hovered_message_idx?;
    if hovered >= state.messages.len() || state.conversation_lines.is_empty() {
        return None;
    }
    let mut last_line_idx = None::<usize>;
    let mut role = UiRole::Assistant;
    for (idx, line) in state.conversation_lines.iter().enumerate() {
        if line.message_index == Some(hovered) {
            last_line_idx = Some(idx);
            role = line.role;
        }
    }
    let message_end_idx = last_line_idx?;
    let button_line_idx = message_end_idx.saturating_add(1);
    let visible_start = state.conversation_scroll as usize;
    let viewport_height = layout.conversation_body.height.max(1) as usize;
    let visible_end = visible_start.saturating_add(viewport_height);
    if button_line_idx < visible_start || button_line_idx >= visible_end {
        return None;
    }
    let copy_label = ui_text_message_copy_button();
    let result_label = ui_text_message_tool_result_button();
    let delete_label = ui_text_message_delete_button();
    let copy_width = text_display_width(copy_label) as u16;
    let result_width = text_display_width(result_label) as u16;
    let delete_width = text_display_width(delete_label) as u16;
    let has_result = state
        .messages
        .get(hovered)
        .and_then(tool_result_detail_from_message)
        .is_some();
    let total_width = if has_result {
        copy_width
            .saturating_add(1)
            .saturating_add(result_width)
            .saturating_add(1)
            .saturating_add(delete_width)
            .max(1)
    } else {
        copy_width
            .saturating_add(1)
            .saturating_add(delete_width)
            .max(1)
    };
    if total_width > layout.conversation_body.width {
        return None;
    }
    let x = if role == UiRole::User {
        layout
            .conversation_body
            .x
            .saturating_add(layout.conversation_body.width.saturating_sub(total_width))
    } else {
        layout.conversation_body.x
    };
    let y = layout
        .conversation_body
        .y
        .saturating_add((button_line_idx.saturating_sub(visible_start)) as u16);
    let copy_rect = Rect {
        x,
        y,
        width: copy_width.max(1),
        height: 1,
    };
    let result_rect = if has_result {
        Some(Rect {
            x: x.saturating_add(copy_rect.width).saturating_add(1),
            y,
            width: result_width.max(1),
            height: 1,
        })
    } else {
        None
    };
    let delete_x = if let Some(rect) = result_rect {
        rect.x.saturating_add(rect.width).saturating_add(1)
    } else {
        x.saturating_add(copy_rect.width).saturating_add(1)
    };
    let delete_rect = Rect {
        x: delete_x,
        y,
        width: delete_width.max(1),
        height: 1,
    };
    Some(ConversationHoverButtons {
        message_index: hovered,
        copy_rect,
        result_rect,
        delete_rect,
    })
}

fn refresh_conversation_cache_for_layout(state: &mut ChatUiState, layout: UiLayout) {
    state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
    state.ensure_conversation_cache();
    state.clamp_scroll(layout.conversation_body.height.max(1));
}

fn hovered_message_index_from_mouse(
    state: &ChatUiState,
    layout: UiLayout,
    mouse: MouseEvent,
) -> Option<usize> {
    if !rect_contains(layout.conversation_body, mouse.column, mouse.row) {
        return None;
    }
    let row_offset = mouse.row.saturating_sub(layout.conversation_body.y) as usize;
    let line_idx = state.conversation_scroll as usize + row_offset;
    state
        .conversation_lines
        .get(line_idx)
        .and_then(|line| line.message_index)
}

fn refresh_hovered_message_from_mouse(
    state: &mut ChatUiState,
    layout: UiLayout,
    mouse: MouseEvent,
) {
    if let Some(idx) = hovered_message_index_from_mouse(state, layout, mouse) {
        state.hovered_message_idx = Some(idx);
        return;
    }
    if let Some(button) = conversation_copy_button_data(state, layout)
        && rect_contains(button.rect, mouse.column, mouse.row)
    {
        return;
    }
    if let Some(button) = conversation_delete_button_data(state, layout)
        && rect_contains(button.rect, mouse.column, mouse.row)
    {
        return;
    }
    if let Some(button) = conversation_tool_result_button_data(state, layout)
        && rect_contains(button.rect, mouse.column, mouse.row)
    {
        return;
    }
    state.hovered_message_idx = None;
}

fn draw_status_bar(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let width = layout.status.width.max(1) as usize;
    let pulse_on = status_pulse_on();
    let chip_color = if pulse_on {
        Color::Rgb(255, 206, 109)
    } else {
        palette.accent
    };
    let detail_color = if pulse_on {
        Color::Rgb(214, 238, 255)
    } else {
        palette.text
    };
    if let Some(live) = state.ai_live.as_ref() {
        let elapsed = live.started_at.elapsed().as_secs_f32();
        let badge = if live.cancel_requested {
            zh_or_en("◉ 取消中", "◉ Cancelling")
        } else if live.last_tool_label.is_empty() {
            zh_or_en("◉ 思考中", "◉ Thinking")
        } else {
            zh_or_en("◉ 工具中", "◉ Tooling")
        };
        let detail = if live.last_tool_label.is_empty() {
            let phase = if live.cancel_requested {
                ui_text_status_ai_cancelling()
            } else {
                ui_text_status_ai_thinking()
            };
            format!("{:.1}s · {}", elapsed, phase)
        } else {
            format!(
                "{:.1}s · {} · tools={} · {}",
                elapsed,
                ui_text_status_ai_tooling(),
                live.tool_calls,
                trim_ui_text(&live.last_tool_label, 30)
            )
        };
        let badge_width = text_display_width(badge);
        let detail_width = width.saturating_sub(badge_width + 1);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    badge,
                    Style::default().fg(chip_color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    trim_ui_text(&detail, detail_width),
                    Style::default().fg(detail_color),
                ),
            ])),
            layout.status,
        );
        return;
    }
    let badge = if pulse_on {
        format!("◆ {}", ui_text_header_status())
    } else {
        format!("◇ {}", ui_text_header_status())
    };
    let badge_width = text_display_width(&badge);
    let detail_width = width.saturating_sub(badge_width + 1);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                badge,
                Style::default().fg(chip_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                trim_ui_text(&state.status, detail_width),
                Style::default().fg(palette.status),
            ),
        ])),
        layout.status,
    );
}

fn status_pulse_on() -> bool {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|item| item.as_millis())
        .unwrap_or(0);
    (millis / 450).is_multiple_of(2)
}

fn draw_config_panel(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_config_panel_title())
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Conversation,
            palette,
        )));
    frame.render_widget(outer, layout.conversation);
    let inner = layout.conversation.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(26), Constraint::Min(20)])
        .split(inner);

    let mut cat_state = ListState::default();
    if !state.config_ui.categories.is_empty() {
        cat_state.select(Some(
            state
                .config_ui
                .selected_category
                .min(state.config_ui.categories.len().saturating_sub(1)),
        ));
    }
    let category_items = state
        .config_ui
        .categories
        .iter()
        .map(|item| ListItem::new(format!(" {}", item.label)))
        .collect::<Vec<_>>();
    frame.render_stateful_widget(
        List::new(category_items)
            .highlight_style(
                Style::default()
                    .fg(palette.text)
                    .bg(palette.panel_bg)
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_config_category()),
            ),
        cols[0],
        &mut cat_state,
    );

    let visible = config_visible_field_indices(state);
    if visible.is_empty() {
        frame.render_widget(
            Paragraph::new(ui_text_config_empty_fields())
                .style(Style::default().fg(palette.muted))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(ui_text_config_fields()),
                ),
            cols[1],
        );
        return;
    }
    let selected_row = state
        .config_ui
        .selected_field_row
        .min(visible.len().saturating_sub(1));
    let selected_idx = visible[selected_row];
    let table_rows = visible
        .iter()
        .map(|idx| {
            let field = &state.config_ui.fields[*idx];
            let type_label = config_kind_label(field.kind);
            let required = if field.required { "*" } else { "" };
            let dirty_mark = if field.dirty { "●" } else { "" };
            Row::new(vec![
                Cell::from(format!("{}{}", field.label, required)),
                Cell::from(trim_ui_text(&field.value, 46)),
                Cell::from(type_label),
                Cell::from(dirty_mark),
            ])
        })
        .collect::<Vec<_>>();
    let mut table_state = TableState::default();
    table_state.select(Some(selected_row));
    let table = Table::new(
        table_rows,
        [
            Constraint::Length(30),
            Constraint::Min(18),
            Constraint::Length(8),
            Constraint::Length(2),
        ],
    )
    .header(
        Row::new(vec![
            Cell::from(ui_text_config_col_key()),
            Cell::from(ui_text_config_col_value()),
            Cell::from(ui_text_config_col_type()),
            Cell::from(" "),
        ])
        .style(
            Style::default()
                .fg(palette.text)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .row_highlight_style(
        Style::default()
            .fg(palette.text)
            .bg(palette.panel_bg)
            .add_modifier(Modifier::BOLD),
    )
    .block(Block::default().borders(Borders::ALL).title(format!(
        "{} · {}",
        ui_text_config_fields(),
        state.config_ui.fields[selected_idx].key
    )));
    frame.render_stateful_widget(table, cols[1], &mut table_state);
}

fn draw_config_input_panel(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let title = format!(
        "{} ({})",
        ui_text_panel_input(),
        ui_text_config_input_hint()
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Input,
            palette,
        )));
    frame.render_widget(block, layout.input);
    let inner = layout.input_body;
    let Some(field_idx) = config_selected_field_index(state) else {
        frame.render_widget(
            Paragraph::new(ui_text_config_empty_fields()).style(Style::default().fg(palette.muted)),
            inner,
        );
        return;
    };
    let field = &state.config_ui.fields[field_idx];
    let required = if field.required {
        ui_text_config_required()
    } else {
        ""
    };
    let meta = format!(
        "{} [{}] {}",
        field.key,
        config_kind_label(field.kind),
        required
    );
    let dirty = format!(
        "{} {}",
        ui_text_config_dirty_count(),
        state.config_ui.dirty_count
    );
    let save_button = if state.config_ui.dirty_count > 0 {
        format!("[ {} ]", ui_text_config_save_button())
    } else {
        format!("[ {} ]", ui_text_config_save_no_change())
    };
    let selectable_hint = if config_field_is_selectable(field) {
        let option_line = if field.options.is_empty() {
            trim_ui_text(&field.value, 22)
        } else {
            trim_ui_text(&field.options.join(" | "), 22)
        };
        format!("<{}> {}", field.value, option_line)
    } else {
        String::new()
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);
    frame.render_widget(
        Paragraph::new(meta).style(Style::default().fg(palette.muted)),
        rows[0],
    );
    if state.config_ui.editing {
        let (visible, cursor_col) =
            project_input_view(&state.config_ui.edit_buffer, rows[1].width.max(1) as usize);
        frame.render_widget(
            Paragraph::new(visible).style(Style::default().fg(palette.text)),
            rows[1],
        );
        if state.focus == FocusPanel::Input {
            frame.set_cursor_position((rows[1].x + cursor_col as u16, rows[1].y));
        }
    } else {
        let value_display = if config_field_is_selectable(field) {
            trim_ui_text(
                format!("< {} >", field.value).as_str(),
                rows[1].width.max(1) as usize,
            )
        } else {
            trim_ui_text(&field.value, rows[1].width as usize)
        };
        frame.render_widget(
            Paragraph::new(value_display).style(Style::default().fg(palette.text)),
            rows[1],
        );
    }
    frame.render_widget(
        Paragraph::new(trim_ui_text(&selectable_hint, rows[2].width as usize))
            .style(Style::default().fg(palette.muted)),
        rows[2],
    );
    frame.render_widget(
        Paragraph::new(format!("{}    {}", save_button, dirty))
            .style(Style::default().fg(palette.accent)),
        rows[3],
    );
    frame.render_widget(
        Paragraph::new(trim_ui_text(
            &state.config_ui.config_path.display().to_string(),
            rows[4].width as usize,
        ))
        .style(Style::default().fg(palette.muted)),
        rows[4],
    );
}

fn draw_input_panel(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    if state.mode == UiMode::Config {
        draw_config_input_panel(frame, state, layout, palette);
        return;
    }
    if state.mode == UiMode::Mcp {
        draw_mcp_input_panel(frame, state, layout, palette);
        return;
    }
    let input_title = format!("{} ({})", ui_text_panel_input(), ui_text_input_hint());
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(input_title)
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Input,
            palette,
        )));
    frame.render_widget(input_block, layout.input);
    if let Some(choice) = state.pending_choice.as_ref() {
        let mut lines = vec![Line::from(ui_text_choice_prompt())];
        for (idx, item) in choice.options.iter().enumerate() {
            let prefix = if idx == choice.selected { ">" } else { " " };
            lines.push(Line::from(format!("{prefix} {}. {}", idx + 1, item)));
        }
        frame.render_widget(
            Paragraph::new(lines)
                .style(Style::default().fg(palette.text))
                .wrap(Wrap { trim: false }),
            layout.input_body,
        );
        return;
    }
    let (visible, cursor_col) = project_input_view(&state.input, layout.input_body.width as usize);
    frame.render_widget(
        Paragraph::new(visible).style(Style::default().fg(palette.text)),
        layout.input_body,
    );
    if state.focus == FocusPanel::Input {
        frame.set_cursor_position((layout.input_body.x + cursor_col as u16, layout.input_body.y));
    }
}

fn draw_skills_panel(
    frame: &mut Frame<'_>,
    services: &ActionServices<'_>,
    state: &mut ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let panel = inspect_panel_rect(layout);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_skills_panel_title())
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, panel);
    let inner = panel.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(8)])
        .split(inner);
    let skills_dir = expand_tilde(&services.cfg.ai.tools.skills.dir);
    frame.render_widget(
        Paragraph::new(format!(
            "{}: {}   {}: {}",
            ui_text_skills_count(),
            services.skills.len(),
            ui_text_skills_dir(),
            trim_ui_text(&skills_dir.display().to_string(), rows[0].width as usize)
        ))
        .style(Style::default().fg(palette.muted)),
        rows[0],
    );
    let skill_rows = build_skill_panel_rows(skills_dir.as_path(), services.skills);
    if skill_rows.is_empty() {
        state.skills_selected_row = 0;
        frame.render_widget(
            Paragraph::new(ui_text_skills_empty())
                .style(Style::default().fg(palette.muted))
                .block(Block::default().borders(Borders::ALL)),
            rows[1],
        );
        return;
    }
    state.skills_selected_row = state
        .skills_selected_row
        .min(skill_rows.len().saturating_sub(1));
    let table_rows = skill_rows
        .iter()
        .map(|item| {
            Row::new(vec![
                Cell::from(item.name.as_str()).style(Style::default().fg(palette.accent)),
                Cell::from(item.summary.as_str()),
                Cell::from(item.path.as_str()).style(Style::default().fg(palette.muted)),
            ])
        })
        .collect::<Vec<_>>();
    let table = Table::new(
        table_rows,
        [
            Constraint::Length(20),
            Constraint::Min(20),
            Constraint::Min(26),
        ],
    )
    .header(
        Row::new(vec![
            Cell::from(ui_text_skills_name()),
            Cell::from(ui_text_skills_purpose()),
            Cell::from(ui_text_skills_path()),
        ])
        .style(
            Style::default()
                .fg(palette.text)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .column_spacing(1)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(ui_text_skills_list()),
    )
    .style(Style::default().fg(palette.text))
    .row_highlight_style(
        Style::default()
            .fg(palette.text)
            .bg(palette.panel_bg)
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("▶ ");
    let mut table_state = TableState::default();
    table_state.select(Some(state.skills_selected_row));
    frame.render_stateful_widget(table, rows[1], &mut table_state);
}

fn draw_skill_doc_modal(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let Some(modal) = state.skill_doc_modal.as_ref() else {
        return;
    };
    draw_modal_overlay(frame, palette);
    let rect = skill_doc_modal_rect(layout);
    frame.render_widget(Clear, rect);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            "{} · {}",
            ui_text_skills_doc_title(),
            modal.skill_name
        ))
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(outer, rect);
    let inner = rect.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(inner);
    let path_line = format!(
        "{}: {}",
        ui_text_skills_path(),
        trim_ui_text(
            &modal.file_path.display().to_string(),
            rows[0].width as usize
        )
    );
    frame.render_widget(
        Paragraph::new(path_line).style(Style::default().fg(palette.muted)),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(modal.rendered_content.as_str())
            .wrap(Wrap { trim: false })
            .scroll((
                modal.scroll.min(skill_doc_modal_scroll_max(modal, layout)),
                0,
            ))
            .style(Style::default().fg(palette.text))
            .block(Block::default().borders(Borders::ALL)),
        rows[1],
    );
    frame.render_widget(
        Paragraph::new(ui_text_skills_doc_hint()).style(Style::default().fg(palette.accent)),
        rows[2],
    );
}

fn draw_mcp_panel(
    frame: &mut Frame<'_>,
    services: &ActionServices<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let panel = layout.conversation;
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_mcp_panel_title())
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Conversation,
            palette,
        )));
    frame.render_widget(outer, panel);
    let inner = panel.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(8)])
        .split(inner);
    let meta_line = format!(
        "{}: {}    {}: {}",
        ui_text_mcp_servers_count(),
        state.mcp_ui.servers.len(),
        ui_text_mcp_config_file(),
        trim_ui_text(
            &state.mcp_ui.config_file_path.display().to_string(),
            rows[0].width.saturating_sub(24) as usize
        )
    );
    frame.render_widget(
        Paragraph::new(meta_line).style(Style::default().fg(palette.muted)),
        rows[0],
    );
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(26)])
        .split(rows[1]);
    let mut server_state = ListState::default();
    if !state.mcp_ui.servers.is_empty() {
        server_state.select(Some(
            state
                .mcp_ui
                .selected_server
                .min(state.mcp_ui.servers.len().saturating_sub(1)),
        ));
    }
    let server_items = if state.mcp_ui.servers.is_empty() {
        vec![ListItem::new(Line::from(format!(
            "  {}",
            ui_text_mcp_empty()
        )))]
    } else {
        state
            .mcp_ui
            .servers
            .iter()
            .map(|item| {
                let marker = if item.config.enabled { "●" } else { "○" };
                let dirty = if item.dirty { "*" } else { "" };
                ListItem::new(Line::from(format!("  {marker} {}{dirty}", item.name)))
            })
            .collect::<Vec<_>>()
    };
    frame.render_stateful_widget(
        List::new(server_items)
            .style(Style::default().fg(palette.text))
            .highlight_style(
                Style::default()
                    .fg(palette.text)
                    .bg(palette.panel_bg)
                    .add_modifier(Modifier::BOLD),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_mcp_servers_list())
                    .border_style(Style::default().fg(panel_border_color(
                        state.focus == FocusPanel::Conversation && state.mcp_ui.focus_servers,
                        palette,
                    ))),
            ),
        cols[0],
        &mut server_state,
    );

    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(7), Constraint::Length(5)])
        .split(cols[1]);
    if let Some(server) = mcp_selected_server(state) {
        let defs = mcp_field_defs();
        let table_rows = defs
            .iter()
            .map(|def| {
                Row::new(vec![
                    Cell::from(def.label),
                    Cell::from(trim_ui_text(
                        &mcp_field_value(server, def.id),
                        right_rows[0].width.saturating_sub(22) as usize,
                    )),
                    Cell::from(config_kind_label(def.kind)),
                ])
            })
            .collect::<Vec<_>>();
        let mut table_state = TableState::default();
        table_state.select(Some(
            state
                .mcp_ui
                .selected_field
                .min(mcp_field_defs().len().saturating_sub(1)),
        ));
        let table = Table::new(
            table_rows,
            [
                Constraint::Length(16),
                Constraint::Min(20),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec![
                Cell::from(ui_text_config_col_key()),
                Cell::from(ui_text_config_col_value()),
                Cell::from(ui_text_config_col_type()),
            ])
            .style(
                Style::default()
                    .fg(palette.text)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .row_highlight_style(
            Style::default()
                .fg(palette.text)
                .bg(palette.panel_bg)
                .add_modifier(Modifier::BOLD),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("{} · {}", ui_text_mcp_server_detail(), server.name))
                .border_style(Style::default().fg(panel_border_color(
                    state.focus == FocusPanel::Conversation && !state.mcp_ui.focus_servers,
                    palette,
                ))),
        );
        frame.render_stateful_widget(table, right_rows[0], &mut table_state);

        let service_status = services
            .mcp
            .service_statuses()
            .into_iter()
            .find(|item| item.name == server.name);
        let tool_names = services
            .mcp
            .tool_statuses()
            .into_iter()
            .filter(|item| item.server_name == server.name)
            .map(|item| item.remote_name)
            .collect::<Vec<_>>();
        let tool_line = if tool_names.is_empty() {
            if let Some(status) = service_status {
                if let Some(err) = status.error {
                    trim_ui_text(
                        &format!("{}: {}", ui_text_mcp_error(), err),
                        right_rows[1].width as usize,
                    )
                } else if !status.available {
                    trim_ui_text(
                        &format!(
                            "{} ({})",
                            ui_text_mcp_tools_empty(),
                            status.summary.unwrap_or_else(|| ui_text_na().to_string())
                        ),
                        right_rows[1].width as usize,
                    )
                } else {
                    ui_text_mcp_tools_empty().to_string()
                }
            } else {
                ui_text_mcp_tools_empty().to_string()
            }
        } else {
            trim_ui_text(&tool_names.join("  "), right_rows[1].width as usize)
        };
        frame.render_widget(
            Paragraph::new(format!("{}\n{}", ui_text_mcp_tools_label(), tool_line))
                .style(Style::default().fg(palette.muted))
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(ui_text_mcp_tools_panel()),
                ),
            right_rows[1],
        );
    } else {
        frame.render_widget(
            Paragraph::new(ui_text_mcp_empty())
                .style(Style::default().fg(palette.muted))
                .block(Block::default().borders(Borders::ALL)),
            right_rows[0],
        );
        frame.render_widget(
            Paragraph::new(ui_text_mcp_no_tools())
                .style(Style::default().fg(palette.muted))
                .block(Block::default().borders(Borders::ALL)),
            right_rows[1],
        );
    }
}

fn draw_mcp_input_panel(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let title = format!("{} ({})", ui_text_panel_input(), ui_text_mcp_input_hint());
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Input,
            palette,
        )));
    frame.render_widget(block, layout.input);
    let inner = layout.input_body;
    let Some(server) = mcp_selected_server(state) else {
        frame.render_widget(
            Paragraph::new(ui_text_mcp_empty_fields()).style(Style::default().fg(palette.muted)),
            inner,
        );
        return;
    };
    let Some(field) = mcp_selected_field_def(state) else {
        frame.render_widget(
            Paragraph::new(ui_text_mcp_empty_fields()).style(Style::default().fg(palette.muted)),
            inner,
        );
        return;
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);
    frame.render_widget(
        Paragraph::new(format!(
            "{}.{} [{}]",
            server.name,
            field.label,
            config_kind_label(field.kind)
        ))
        .style(Style::default().fg(palette.muted)),
        rows[0],
    );
    if state.mcp_ui.editing {
        let (visible, cursor_col) =
            project_input_view(&state.mcp_ui.edit_buffer, rows[1].width.max(1) as usize);
        frame.render_widget(
            Paragraph::new(visible).style(Style::default().fg(palette.text)),
            rows[1],
        );
        if state.focus == FocusPanel::Input {
            frame.set_cursor_position((rows[1].x + cursor_col as u16, rows[1].y));
        }
    } else {
        frame.render_widget(
            Paragraph::new(trim_ui_text(
                &mcp_field_value(server, field.id),
                rows[1].width as usize,
            ))
            .style(Style::default().fg(palette.text)),
            rows[1],
        );
    }
    let save_button = if state.mcp_ui.dirty_count > 0 {
        format!("[ {} ]", ui_text_mcp_save_button())
    } else {
        format!("[ {} ]", ui_text_mcp_save_no_change())
    };
    frame.render_widget(
        Paragraph::new(format!(
            "{}    {} {}",
            save_button,
            ui_text_mcp_dirty_count(),
            state.mcp_ui.dirty_count
        ))
        .style(Style::default().fg(palette.accent)),
        rows[2],
    );
    let tail = if let Some(err) = state.mcp_ui.last_error.as_deref() {
        format!(
            "{}: {}",
            ui_text_mcp_error(),
            trim_ui_text(err, rows[3].width as usize)
        )
    } else {
        trim_ui_text(
            &state.mcp_ui.config_file_path.display().to_string(),
            rows[3].width as usize,
        )
    };
    frame.render_widget(
        Paragraph::new(tail).style(Style::default().fg(palette.muted)),
        rows[3],
    );
}

fn inspect_panel_rect(layout: UiLayout) -> Rect {
    Rect {
        x: layout.conversation.x,
        y: layout.conversation.y,
        width: layout.conversation.width,
        height: layout
            .conversation
            .height
            .saturating_add(layout.input.height)
            .max(layout.conversation.height),
    }
}

fn skill_doc_modal_rect(layout: UiLayout) -> Rect {
    let panel = inspect_panel_rect(layout);
    let target_height = panel.height.saturating_sub(2).clamp(12, 34);
    centered_rect(panel, 92, target_height)
}

fn clamp_skill_doc_modal_scroll(state: &mut ChatUiState, layout: UiLayout) {
    if let Some(modal) = state.skill_doc_modal.as_mut() {
        modal.scroll = modal.scroll.min(skill_doc_modal_scroll_max(modal, layout));
    }
}

fn skill_doc_modal_scroll_max(modal: &SkillDocModalState, layout: UiLayout) -> u16 {
    let rect = skill_doc_modal_rect(layout);
    let inner = rect.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(inner);
    let text_width = rows[1].width.saturating_sub(2).max(1) as usize;
    let viewport_height = rows[1].height.saturating_sub(2).max(1) as usize;
    let wrapped_lines = modal
        .rendered_content
        .lines()
        .flat_map(|line| wrap_text_by_display_width(line, text_width))
        .count()
        .max(1);
    wrapped_lines
        .saturating_sub(viewport_height)
        .min(u16::MAX as usize) as u16
}

fn skill_row_index_from_mouse(
    layout: UiLayout,
    mouse: MouseEvent,
    services: &ActionServices<'_>,
) -> Option<usize> {
    let skills_dir = expand_tilde(&services.cfg.ai.tools.skills.dir);
    let rows = build_skill_panel_rows(skills_dir.as_path(), services.skills);
    if rows.is_empty() {
        return None;
    }
    let panel = inspect_panel_rect(layout);
    let inner = panel.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(8)])
        .split(inner);
    let table_area = split[1];
    if !rect_contains(table_area, mouse.column, mouse.row) {
        return None;
    }
    let content = table_area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if !rect_contains(content, mouse.column, mouse.row) {
        return None;
    }
    let row = mouse.row.saturating_sub(content.y) as usize;
    if row == 0 {
        return None;
    }
    let idx = row.saturating_sub(1);
    if idx >= rows.len() {
        return None;
    }
    Some(idx)
}

fn draw_inspect_panel(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    let panel = inspect_panel_rect(layout);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            "{} · {}",
            ui_text_inspect_panel_title(),
            inspect_target_label(state.inspect.target)
        ))
        .border_style(Style::default().fg(panel_border_color(
            state.focus == FocusPanel::Conversation,
            palette,
        )));
    frame.render_widget(outer, panel);
    let inner = panel.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if state.inspect.target.as_str() != InspectTarget::Cpu.as_str() {
        draw_generic_inspect_content(frame, state, palette, inner);
        return;
    }
    draw_cpu_inspect_content(frame, state, palette, inner);
}

fn draw_cpu_inspect_content(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    palette: ThemePalette,
    inner: Rect,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(8),
            Constraint::Min(4),
        ])
        .split(inner);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(26), Constraint::Min(12)])
        .split(rows[0]);
    let usage = state.inspect.metrics.usage_percent.clamp(0.0, 100.0);
    let gauge = Gauge::default()
        .label(format!("{usage:.1}%"))
        .percent(usage as u16)
        .style(Style::default().fg(palette.accent))
        .gauge_style(Style::default().fg(palette.accent).bg(palette.panel_bg))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_inspect_usage()),
        );
    frame.render_widget(gauge, top[0]);
    let history_data = if state.inspect.usage_history.is_empty() {
        vec![0u64]
    } else {
        state
            .inspect
            .usage_history
            .iter()
            .copied()
            .collect::<Vec<_>>()
    };
    let sparkline = Sparkline::default()
        .data(&history_data)
        .style(Style::default().fg(palette.accent))
        .max(100)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_inspect_trend()),
        );
    frame.render_widget(sparkline, top[1]);

    let info_rows = vec![
        Row::new(vec![
            Cell::from(ui_text_inspect_model()),
            Cell::from(state.inspect.metrics.model.as_str()),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_cores()),
            Cell::from(format!(
                "{} / {}",
                state.inspect.metrics.physical_cores, state.inspect.metrics.logical_cores
            )),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_frequency()),
            Cell::from(state.inspect.metrics.freq_mhz.as_str()),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_temperature()),
            Cell::from(state.inspect.metrics.temperature.as_str()),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_health()),
            Cell::from(state.inspect.metrics.health.as_str()),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_breakdown()),
            Cell::from(format!(
                "usr {:.1}% | sys {:.1}% | idle {:.1}%",
                state.inspect.metrics.user_percent,
                state.inspect.metrics.system_percent,
                state.inspect.metrics.idle_percent
            )),
        ]),
        Row::new(vec![
            Cell::from(ui_text_inspect_updated()),
            Cell::from(state.inspect.metrics.last_updated.as_str()),
        ]),
    ];
    let info_table = Table::new(info_rows, [Constraint::Length(15), Constraint::Min(20)])
        .column_spacing(1)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_inspect_details()),
        )
        .style(Style::default().fg(palette.text));
    frame.render_widget(info_table, rows[1]);
    let raw = if state.inspect.metrics.raw_output.trim().is_empty() {
        ui_text_inspect_no_data().to_string()
    } else {
        state.inspect.metrics.raw_output.clone()
    };
    frame.render_widget(
        Paragraph::new(raw)
            .style(Style::default().fg(palette.muted))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_inspect_raw()),
            ),
        rows[2],
    );
}

fn draw_generic_inspect_content(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    palette: ThemePalette,
    inner: Rect,
) {
    let raw = if state.inspect.metrics.raw_output.trim().is_empty() {
        ui_text_inspect_no_data().to_string()
    } else {
        state.inspect.metrics.raw_output.clone()
    };
    let line_count = raw.lines().filter(|line| !line.trim().is_empty()).count();
    let byte_count = raw.len();
    let signal = if raw.to_ascii_lowercase().contains("timed out")
        || raw.to_ascii_lowercase().contains("not found")
        || raw.to_ascii_lowercase().contains("error")
    {
        ui_text_inspect_signal_warn()
    } else {
        ui_text_inspect_signal_ok()
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(8),
            Constraint::Min(4),
        ])
        .split(inner);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Min(18),
        ])
        .split(rows[0]);
    frame.render_widget(
        Paragraph::new(format!(
            "{}\n{}",
            ui_text_inspect_target(),
            inspect_target_label(state.inspect.target)
        ))
        .style(Style::default().fg(palette.text))
        .block(Block::default().borders(Borders::ALL)),
        top[0],
    );
    frame.render_widget(
        Paragraph::new(format!("{}\n{}", ui_text_inspect_line_count(), line_count))
            .style(Style::default().fg(palette.text))
            .block(Block::default().borders(Borders::ALL)),
        top[1],
    );
    frame.render_widget(
        Paragraph::new(format!("{}\n{}", ui_text_inspect_byte_count(), byte_count))
            .style(Style::default().fg(palette.text))
            .block(Block::default().borders(Borders::ALL)),
        top[2],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "{}\n{} · {}",
            ui_text_inspect_health(),
            signal,
            state.inspect.metrics.last_updated
        ))
        .style(Style::default().fg(palette.accent))
        .block(Block::default().borders(Borders::ALL)),
        top[3],
    );

    let detail_rows = extract_inspect_highlight_rows(raw.as_str())
        .into_iter()
        .map(|(key, value)| Row::new(vec![Cell::from(key), Cell::from(value)]))
        .collect::<Vec<_>>();
    let details = Table::new(detail_rows, [Constraint::Length(24), Constraint::Min(20)])
        .column_spacing(1)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(ui_text_inspect_highlights()),
        )
        .style(Style::default().fg(palette.text));
    frame.render_widget(details, rows[1]);

    frame.render_widget(
        Paragraph::new(raw)
            .style(Style::default().fg(palette.muted))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(ui_text_inspect_raw()),
            ),
        rows[2],
    );
}

fn extract_inspect_highlight_rows(raw: &str) -> Vec<(String, String)> {
    let mut out = Vec::<(String, String)>::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let k = trim_tool_text(key.trim(), 22);
            let v = trim_tool_text(value.trim(), 84);
            if !k.is_empty() && !v.is_empty() {
                out.push((k, v));
            }
        } else if out.len() < 8 {
            out.push((
                ui_text_inspect_raw_line().to_string(),
                trim_tool_text(trimmed, 84),
            ));
        }
        if out.len() >= 8 {
            break;
        }
    }
    if out.is_empty() {
        out.push((
            ui_text_inspect_highlights_empty().to_string(),
            ui_text_na().to_string(),
        ));
    }
    out
}

fn inspect_menu_rect(layout: UiLayout) -> Rect {
    centered_rect(inspect_panel_rect(layout), 52, 14)
}

fn draw_inspect_target_menu(
    frame: &mut Frame<'_>,
    state: &ChatUiState,
    layout: UiLayout,
    palette: ThemePalette,
) {
    draw_modal_overlay(frame, palette);
    let menu = inspect_menu_rect(layout);
    frame.render_widget(Clear, menu);
    let menu_block = Block::default()
        .borders(Borders::ALL)
        .title(ui_text_inspect_target_menu())
        .style(Style::default().bg(palette.panel_bg))
        .border_style(Style::default().fg(palette.accent));
    frame.render_widget(menu_block, menu);
    let inner = menu.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let mut stateful = ListState::default();
    stateful.select(Some(
        state
            .inspect_menu_selected
            .min(inspect_targets().len().saturating_sub(1)),
    ));
    let items = inspect_targets()
        .iter()
        .enumerate()
        .map(|(idx, target)| {
            ListItem::new(format!(" {}. {}", idx + 1, inspect_target_label(*target)))
        })
        .collect::<Vec<_>>();
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(
                Style::default()
                    .fg(palette.text)
                    .bg(palette.app_bg)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::NONE)),
        inner,
        &mut stateful,
    );
}

fn build_skill_panel_rows(skills_dir: &Path, skills: &[String]) -> Vec<SkillPanelRow> {
    let mut names = skills.to_vec();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let path = skills_dir.join(&name);
            SkillPanelRow {
                name,
                summary: read_skill_summary_for_panel(path.as_path())
                    .unwrap_or_else(|| ui_text_na().to_string()),
                path: path.display().to_string(),
            }
        })
        .collect()
}

fn read_skill_summary_for_panel(skill_dir: &Path) -> Option<String> {
    let content = fs::read_to_string(skill_dir.join("SKILL.md")).ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("```")
            || trimmed == "---"
        {
            continue;
        }
        let normalized = trimmed.trim_start_matches('-').trim();
        if !normalized.is_empty() {
            return Some(trim_tool_text(normalized, 72));
        }
    }
    None
}

fn project_input_view(input: &InputBuffer, max_width: usize) -> (String, usize) {
    if max_width == 0 {
        return (String::new(), 0);
    }
    let mut start = input.view_char_offset;
    let cursor_col_from_start = display_width_between(&input.text, start, input.cursor_char);
    let target_max = max_width.saturating_sub(1);
    if cursor_col_from_start > target_max {
        while start < input.cursor_char
            && display_width_between(&input.text, start, input.cursor_char) > target_max
        {
            start += 1;
        }
    }
    let mut out = String::new();
    for (idx, ch) in input.text.chars().enumerate().skip(start) {
        out.push(ch);
        if text_display_width(&out) > max_width {
            out.pop();
            break;
        }
        if idx + 1 >= input.char_count() {
            break;
        }
    }
    let cursor = display_width_between(&input.text, start, input.cursor_char).min(max_width);
    (out, cursor)
}

fn build_config_ui_state(cfg: &AppConfig, config_path: &Path) -> ConfigUiState {
    let fields = config_seed_fields()
        .into_iter()
        .map(|seed| ConfigField {
            key: seed.key.to_string(),
            label: seed.label.to_string(),
            category: seed.category.to_string(),
            kind: seed.kind,
            value: config_value_from_cfg(cfg, seed.key).unwrap_or_default(),
            required: seed.required,
            options: seed.options.iter().map(|item| item.to_string()).collect(),
            dirty: false,
        })
        .collect::<Vec<_>>();
    let mut categories = Vec::<ConfigCategory>::new();
    for field in &fields {
        if categories.iter().any(|item| item.id == field.category) {
            continue;
        }
        categories.push(ConfigCategory {
            id: field.category.clone(),
            label: config_category_label(&field.category),
        });
    }
    ConfigUiState {
        categories,
        selected_category: 0,
        selected_field_row: 0,
        editing: false,
        edit_buffer: InputBuffer::new(),
        fields,
        dirty_count: 0,
        config_path: config_path.to_path_buf(),
    }
}

fn mcp_field_defs() -> &'static [McpFieldDef] {
    static DEFS: [McpFieldDef; 12] = [
        McpFieldDef {
            id: McpFieldId::Name,
            label: "name",
            kind: ConfigFieldKind::String,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Enabled,
            label: "enabled",
            kind: ConfigFieldKind::Bool,
            options: &["false", "true"],
        },
        McpFieldDef {
            id: McpFieldId::Transport,
            label: "type",
            kind: ConfigFieldKind::Enum,
            options: &["", "http", "streamable_http", "sse", "stdio"],
        },
        McpFieldDef {
            id: McpFieldId::Url,
            label: "url",
            kind: ConfigFieldKind::OptionalString,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Endpoint,
            label: "endpoint",
            kind: ConfigFieldKind::OptionalString,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Command,
            label: "command",
            kind: ConfigFieldKind::OptionalString,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Args,
            label: "args",
            kind: ConfigFieldKind::StringList,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Env,
            label: "env",
            kind: ConfigFieldKind::StringMap,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::Headers,
            label: "headers",
            kind: ConfigFieldKind::StringMap,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::AuthType,
            label: "authType",
            kind: ConfigFieldKind::OptionalString,
            options: &["", "bearer"],
        },
        McpFieldDef {
            id: McpFieldId::AuthToken,
            label: "authToken",
            kind: ConfigFieldKind::OptionalString,
            options: &[],
        },
        McpFieldDef {
            id: McpFieldId::TimeoutSeconds,
            label: "timeoutSeconds",
            kind: ConfigFieldKind::OptionalU64,
            options: &[],
        },
    ];
    &DEFS
}

fn build_mcp_ui_state(cfg: &AppConfig, config_path: &Path) -> McpUiState {
    let config_file_path = mcp::resolve_mcp_servers_file_path(&cfg.ai.tools.mcp, config_path);
    let (servers, last_error) = match mcp::load_mcp_server_records(&cfg.ai.tools.mcp, config_path) {
        Ok(items) => (
            items
                .into_iter()
                .map(|item| McpUiServer {
                    name: item.name,
                    config: item.config,
                    dirty: false,
                })
                .collect::<Vec<_>>(),
            None,
        ),
        Err(err) => (Vec::new(), Some(mask_ui_sensitive(&err.to_string()))),
    };
    McpUiState {
        servers,
        selected_server: 0,
        selected_field: 0,
        focus_servers: true,
        editing: false,
        edit_buffer: InputBuffer::new(),
        dirty_count: 0,
        structural_dirty: false,
        config_file_path,
        last_error,
    }
}

fn config_seed_fields() -> Vec<ConfigFieldSeed> {
    vec![
        ConfigFieldSeed {
            key: "app.language",
            label: "language",
            category: "app",
            kind: ConfigFieldKind::OptionalString,
            required: false,
            options: &["zh-CN", "zh-TW", "en", "fr", "de", "ja"],
        },
        ConfigFieldSeed {
            key: "app.env-mode",
            label: "env-mode",
            category: "app",
            kind: ConfigFieldKind::Enum,
            required: false,
            options: &["prod", "test", "dev"],
        },
        ConfigFieldSeed {
            key: "ai.type",
            label: "type",
            category: "ai",
            kind: ConfigFieldKind::Enum,
            required: false,
            options: &["openai", "claude", "gemini", "deepseek", "qwen", "ollama"],
        },
        ConfigFieldSeed {
            key: "ai.base-url",
            label: "base-url",
            category: "ai",
            kind: ConfigFieldKind::String,
            required: true,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.token",
            label: "token",
            category: "ai",
            kind: ConfigFieldKind::String,
            required: true,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.model",
            label: "model",
            category: "ai",
            kind: ConfigFieldKind::String,
            required: true,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.debug",
            label: "debug",
            category: "ai",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.connectivity-check",
            label: "connectivity-check",
            category: "ai",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.input-price-per-million",
            label: "input-price-per-million",
            category: "ai",
            kind: ConfigFieldKind::F64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.output-price-per-million",
            label: "output-price-per-million",
            category: "ai",
            kind: ConfigFieldKind::F64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.retry.max-retries",
            label: "max-retries",
            category: "ai.retry",
            kind: ConfigFieldKind::U32,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.retry.backoff-millis",
            label: "backoff-millis",
            category: "ai.retry",
            kind: ConfigFieldKind::U64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-tool",
            label: "show-tool",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-tool-ok",
            label: "show-tool-ok",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-tool-err",
            label: "show-tool-err",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-tool-timeout",
            label: "show-tool-timeout",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-tips",
            label: "show-tips",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.command-cache-ttl-seconds",
            label: "command-cache-ttl-seconds",
            category: "ai.chat",
            kind: ConfigFieldKind::U64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-round-metrics",
            label: "show-round-metrics",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.show-token-cost",
            label: "show-token-cost",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.skip-model-price-check",
            label: "skip-model-price-check",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.model-price-check-mode",
            label: "model-price-check-mode",
            category: "ai.chat",
            kind: ConfigFieldKind::Enum,
            required: false,
            options: &["sync", "async"],
        },
        ConfigFieldSeed {
            key: "ai.chat.context-warn-percent",
            label: "context-warn-percent",
            category: "ai.chat",
            kind: ConfigFieldKind::U8,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.context-critical-percent",
            label: "context-critical-percent",
            category: "ai.chat",
            kind: ConfigFieldKind::U8,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.stream-output",
            label: "stream-output",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.output-multilines",
            label: "output-multilines",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.skip-env-profile",
            label: "skip-env-profile",
            category: "ai.chat",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.chat.cmd-run-timout",
            label: "cmd-run-timout",
            category: "ai.chat",
            kind: ConfigFieldKind::U64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.max-tool-rounds",
            label: "max-tool-rounds",
            category: "ai.chat",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.max-total-tool-calls",
            label: "max-total-tool-calls",
            category: "ai.chat",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.compression.max-history-messages",
            label: "max-history-messages",
            category: "ai.chat.compression",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.chat.compression.max-chars-count",
            label: "max-chars-count",
            category: "ai.chat.compression",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.write-cmd-run-confirm",
            label: "write-cmd-run-confirm",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.command-timeout-seconds",
            label: "command-timeout-seconds",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::U64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.command-timeout-kill-after-seconds",
            label: "command-timeout-kill-after-seconds",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::U64,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.write-cmd-confirm-mode",
            label: "write-cmd-confirm-mode",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::Enum,
            required: false,
            options: &["deny", "edit", "allow-once", "allow-session"],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.allow-cmd-list",
            label: "allow-cmd-list",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::StringList,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.deny-cmd-list",
            label: "deny-cmd-list",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::StringList,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.write-cmd-allow-patterns",
            label: "write-cmd-allow-patterns",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::StringList,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.write-cmd-deny-patterns",
            label: "write-cmd-deny-patterns",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::StringList,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.bash.command-output-max-bytes",
            label: "command-output-max-bytes",
            category: "ai.tools.bash",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.skills.enabled",
            label: "enabled",
            category: "ai.tools.skills",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.tools.skills.dir",
            label: "dir",
            category: "ai.tools.skills",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "ai.tools.mcp.enabled",
            label: "enabled",
            category: "ai.tools.mcp",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "ai.tools.mcp.mcp-availability-check-mode",
            label: "mcp-availability-check-mode",
            category: "ai.tools.mcp",
            kind: ConfigFieldKind::Enum,
            required: false,
            options: &["rsync", "async"],
        },
        ConfigFieldSeed {
            key: "ai.tools.mcp.dir",
            label: "dir",
            category: "ai.tools.mcp",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "console.colorful",
            label: "colorful",
            category: "console",
            kind: ConfigFieldKind::Bool,
            required: false,
            options: &["false", "true"],
        },
        ConfigFieldSeed {
            key: "log.dir",
            label: "dir",
            category: "log",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "log.log-file-name",
            label: "log-file-name",
            category: "log",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "log.max-file-size",
            label: "max-file-size",
            category: "log",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "log.max-save-time",
            label: "max-save-time",
            category: "log",
            kind: ConfigFieldKind::String,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "session.recent_messages",
            label: "recent_messages",
            category: "session",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
        ConfigFieldSeed {
            key: "session.max_messages",
            label: "max_messages",
            category: "session",
            kind: ConfigFieldKind::Usize,
            required: false,
            options: &[],
        },
    ]
}

fn config_value_from_cfg(cfg: &AppConfig, key: &str) -> Option<String> {
    Some(match key {
        "app.language" => cfg.app.language.clone().unwrap_or_default(),
        "app.env-mode" => cfg.app.env_mode.clone(),
        "ai.type" => cfg.ai.r#type.clone(),
        "ai.base-url" => cfg.ai.base_url.clone(),
        "ai.token" => cfg.ai.token.clone(),
        "ai.model" => cfg.ai.model.clone(),
        "ai.debug" => bool_to_text(cfg.ai.debug),
        "ai.connectivity-check" => bool_to_text(cfg.ai.connectivity_check),
        "ai.input-price-per-million" => cfg.ai.input_price_per_million.to_string(),
        "ai.output-price-per-million" => cfg.ai.output_price_per_million.to_string(),
        "ai.retry.max-retries" => cfg.ai.retry.max_retries.to_string(),
        "ai.retry.backoff-millis" => cfg.ai.retry.backoff_millis.to_string(),
        "ai.chat.show-tool" => bool_to_text(cfg.ai.chat.show_tool),
        "ai.chat.show-tool-ok" => bool_to_text(cfg.ai.chat.show_tool_ok),
        "ai.chat.show-tool-err" => bool_to_text(cfg.ai.chat.show_tool_err),
        "ai.chat.show-tool-timeout" => bool_to_text(cfg.ai.chat.show_tool_timeout),
        "ai.chat.show-tips" => bool_to_text(cfg.ai.chat.show_tips),
        "ai.chat.command-cache-ttl-seconds" => cfg.ai.chat.command_cache_ttl_seconds.to_string(),
        "ai.chat.show-round-metrics" => bool_to_text(cfg.ai.chat.show_round_metrics),
        "ai.chat.show-token-cost" => bool_to_text(cfg.ai.chat.show_token_cost),
        "ai.chat.skip-model-price-check" => bool_to_text(cfg.ai.chat.skip_model_price_check),
        "ai.chat.model-price-check-mode" => cfg.ai.chat.model_price_check_mode.clone(),
        "ai.chat.context-warn-percent" => cfg.ai.chat.context_warn_percent.to_string(),
        "ai.chat.context-critical-percent" => cfg.ai.chat.context_critical_percent.to_string(),
        "ai.chat.stream-output" => bool_to_text(cfg.ai.chat.stream_output),
        "ai.chat.output-multilines" => bool_to_text(cfg.ai.chat.output_multilines),
        "ai.chat.skip-env-profile" => bool_to_text(cfg.ai.chat.skip_env_profile),
        "ai.chat.cmd-run-timout" => cfg.ai.chat.cmd_run_timout.to_string(),
        "ai.chat.max-tool-rounds" => cfg.ai.chat.max_tool_rounds.to_string(),
        "ai.chat.max-total-tool-calls" => cfg.ai.chat.max_total_tool_calls.to_string(),
        "ai.chat.compression.max-history-messages" => {
            cfg.ai.chat.compression.max_history_messages.to_string()
        }
        "ai.chat.compression.max-chars-count" => {
            cfg.ai.chat.compression.max_chars_count.to_string()
        }
        "ai.tools.bash.write-cmd-run-confirm" => {
            bool_to_text(cfg.ai.tools.bash.write_cmd_run_confirm)
        }
        "ai.tools.bash.command-timeout-seconds" => {
            cfg.ai.tools.bash.command_timeout_seconds.to_string()
        }
        "ai.tools.bash.command-timeout-kill-after-seconds" => cfg
            .ai
            .tools
            .bash
            .command_timeout_kill_after_seconds
            .to_string(),
        "ai.tools.bash.write-cmd-confirm-mode" => cfg.ai.tools.bash.write_cmd_confirm_mode.clone(),
        "ai.tools.bash.allow-cmd-list" => to_toml_string_array(&cfg.ai.tools.bash.allow_cmd_list),
        "ai.tools.bash.deny-cmd-list" => to_toml_string_array(&cfg.ai.tools.bash.deny_cmd_list),
        "ai.tools.bash.write-cmd-allow-patterns" => {
            to_toml_string_array(&cfg.ai.tools.bash.write_cmd_allow_patterns)
        }
        "ai.tools.bash.write-cmd-deny-patterns" => {
            to_toml_string_array(&cfg.ai.tools.bash.write_cmd_deny_patterns)
        }
        "ai.tools.bash.command-output-max-bytes" => {
            cfg.ai.tools.bash.command_output_max_bytes.to_string()
        }
        "ai.tools.skills.enabled" => bool_to_text(cfg.ai.tools.skills.enabled),
        "ai.tools.skills.dir" => cfg.ai.tools.skills.dir.clone(),
        "ai.tools.mcp.enabled" => bool_to_text(cfg.ai.tools.mcp.enabled),
        "ai.tools.mcp.mcp-availability-check-mode" => {
            cfg.ai.tools.mcp.mcp_availability_check_mode.clone()
        }
        "ai.tools.mcp.dir" => cfg.ai.tools.mcp.dir.clone(),
        "console.colorful" => bool_to_text(cfg.console.colorful),
        "log.dir" => cfg.log.dir.clone(),
        "log.log-file-name" => cfg.log.log_file_name.clone(),
        "log.max-file-size" => cfg.log.max_file_size.clone(),
        "log.max-save-time" => cfg.log.max_save_time.clone(),
        "session.recent_messages" => cfg.session.recent_messages.to_string(),
        "session.max_messages" => cfg.session.max_messages.to_string(),
        _ => return None,
    })
}

fn bool_to_text(value: bool) -> String {
    if value { "true" } else { "false" }.to_string()
}

fn to_toml_string_array(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let body = values
        .iter()
        .map(|item| format!("\"{}\"", item.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn to_toml_string_map(values: &std::collections::BTreeMap<String, String>) -> String {
    if values.is_empty() {
        return "{}".to_string();
    }
    let body = values
        .iter()
        .map(|(k, v)| {
            format!(
                "\"{}\" = \"{}\"",
                k.replace('\\', "\\\\").replace('"', "\\\""),
                v.replace('\\', "\\\\").replace('"', "\\\"")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {body} }}")
}

fn config_category_label(category: &str) -> String {
    match category {
        "app" => ui_text_config_category_app().to_string(),
        "ai" => ui_text_config_category_ai().to_string(),
        "ai.retry" => ui_text_config_category_ai_retry().to_string(),
        "ai.chat" => ui_text_config_category_ai_chat().to_string(),
        "ai.chat.compression" => ui_text_config_category_ai_compression().to_string(),
        "ai.tools.bash" => ui_text_config_category_ai_tools_bash().to_string(),
        "ai.tools.skills" => ui_text_config_category_skills().to_string(),
        "ai.tools.mcp" => ui_text_config_category_mcp().to_string(),
        "console" => ui_text_config_category_console().to_string(),
        "log" => ui_text_config_category_log().to_string(),
        "session" => ui_text_config_category_session().to_string(),
        _ => category.to_string(),
    }
}

fn config_visible_field_indices(state: &ChatUiState) -> Vec<usize> {
    let Some(category) = state
        .config_ui
        .categories
        .get(state.config_ui.selected_category)
        .map(|item| item.id.as_str())
    else {
        return Vec::new();
    };
    state
        .config_ui
        .fields
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| (field.category == category).then_some(idx))
        .collect()
}

fn config_selected_field_index(state: &ChatUiState) -> Option<usize> {
    let visible = config_visible_field_indices(state);
    visible
        .get(
            state
                .config_ui
                .selected_field_row
                .min(visible.len().saturating_sub(1)),
        )
        .copied()
}

fn config_switch_category(state: &mut ChatUiState, delta: i16) {
    let len = state.config_ui.categories.len() as i32;
    if len <= 0 {
        return;
    }
    let mut next = state.config_ui.selected_category as i32 + delta as i32;
    if next < 0 {
        next = 0;
    }
    if next >= len {
        next = len - 1;
    }
    state.config_ui.selected_category = next as usize;
    state.config_ui.selected_field_row = 0;
    state.config_ui.editing = false;
}

fn config_scroll_fields(state: &mut ChatUiState, delta: i16) {
    let len = config_visible_field_indices(state).len() as i32;
    if len <= 0 {
        state.config_ui.selected_field_row = 0;
        return;
    }
    let mut next = state.config_ui.selected_field_row as i32 + delta as i32;
    if next < 0 {
        next = 0;
    }
    if next >= len {
        next = len - 1;
    }
    state.config_ui.selected_field_row = next as usize;
    state.config_ui.editing = false;
}

fn config_start_edit(state: &mut ChatUiState, field_idx: usize) {
    if field_idx >= state.config_ui.fields.len() {
        return;
    }
    if config_field_is_selectable(&state.config_ui.fields[field_idx]) {
        state.config_ui.editing = false;
        return;
    }
    state.config_ui.editing = true;
    state.config_ui.edit_buffer = InputBuffer {
        text: state.config_ui.fields[field_idx].value.clone(),
        cursor_char: state.config_ui.fields[field_idx].value.chars().count(),
        view_char_offset: 0,
    };
    state.status = ui_text_config_editing().to_string();
}

fn config_apply_edit_buffer(state: &mut ChatUiState) -> Result<(), AppError> {
    let Some(field_idx) = config_selected_field_index(state) else {
        return Ok(());
    };
    if field_idx >= state.config_ui.fields.len() {
        return Ok(());
    }
    let next = state.config_ui.edit_buffer.text.trim().to_string();
    let current = state.config_ui.fields[field_idx].value.clone();
    if next != current {
        state.config_ui.fields[field_idx].value = next;
        state.config_ui.fields[field_idx].dirty = true;
        state.config_ui.dirty_count = state.config_ui.fields.iter().filter(|f| f.dirty).count();
    }
    state.config_ui.editing = false;
    state.status = ui_text_config_edit_applied().to_string();
    Ok(())
}

fn config_cycle_field_option(state: &mut ChatUiState, field_idx: usize, step: i16) {
    if field_idx >= state.config_ui.fields.len() {
        return;
    }
    let field = &mut state.config_ui.fields[field_idx];
    if !field.options.is_empty() {
        let current = field.value.trim().to_string();
        let base = field
            .options
            .iter()
            .position(|item| item.eq_ignore_ascii_case(&current))
            .map(|idx| idx as i32)
            .unwrap_or_else(|| {
                if step >= 0 {
                    -1
                } else {
                    field.options.len() as i32
                }
            });
        let mut idx = base + step as i32;
        if idx < 0 {
            idx = 0;
        }
        if idx >= field.options.len() as i32 {
            idx = field.options.len() as i32 - 1;
        }
        field.value = field.options[idx as usize].clone();
        field.dirty = true;
    } else if field.kind == ConfigFieldKind::Bool {
        field.value = if field.value.trim().eq_ignore_ascii_case("true") {
            "false".to_string()
        } else {
            "true".to_string()
        };
        field.dirty = true;
    }
    state.config_ui.dirty_count = state.config_ui.fields.iter().filter(|f| f.dirty).count();
}

fn config_field_is_selectable(field: &ConfigField) -> bool {
    matches!(field.kind, ConfigFieldKind::Bool | ConfigFieldKind::Enum) || !field.options.is_empty()
}

fn config_step_selected_option(state: &mut ChatUiState, step: i16) -> bool {
    let Some(field_idx) = config_selected_field_index(state) else {
        return false;
    };
    if field_idx >= state.config_ui.fields.len() {
        return false;
    }
    if !config_field_is_selectable(&state.config_ui.fields[field_idx]) {
        return false;
    }
    config_cycle_field_option(state, field_idx, step);
    true
}

fn config_activate_selected_field(state: &mut ChatUiState, switch_input_for_edit: bool) {
    let Some(field_idx) = config_selected_field_index(state) else {
        return;
    };
    if field_idx >= state.config_ui.fields.len() {
        return;
    }
    if config_field_is_selectable(&state.config_ui.fields[field_idx]) {
        config_cycle_field_option(state, field_idx, 1);
        state.status = ui_text_config_edit_applied().to_string();
    } else {
        if switch_input_for_edit {
            state.focus = FocusPanel::Input;
        }
        config_start_edit(state, field_idx);
    }
}

fn config_kind_label(kind: ConfigFieldKind) -> &'static str {
    match kind {
        ConfigFieldKind::Bool => "bool",
        ConfigFieldKind::String => "string",
        ConfigFieldKind::OptionalString => "opt-str",
        ConfigFieldKind::U8 => "u8",
        ConfigFieldKind::U32 => "u32",
        ConfigFieldKind::U64 => "u64",
        ConfigFieldKind::Usize => "usize",
        ConfigFieldKind::F64 => "f64",
        ConfigFieldKind::StringList => "array",
        ConfigFieldKind::StringMap => "table",
        ConfigFieldKind::OptionalU64 => "opt-u64",
        ConfigFieldKind::Enum => "enum",
    }
}

fn save_config_ui(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
) -> Result<(), AppError> {
    if state.config_ui.dirty_count == 0 {
        state.status = ui_text_config_nothing_to_save().to_string();
        return Ok(());
    }
    let result = (|| -> Result<(), AppError> {
        let mut doc = read_config_document(services.config_path)?;
        for field in state.config_ui.fields.iter().filter(|item| item.dirty) {
            let parsed = parse_config_field_to_item(field)?;
            if let Some(item) = parsed {
                set_config_item_by_path(&mut doc, &field.key, item)?;
            } else {
                remove_config_item_by_path(&mut doc, &field.key);
            }
        }
        let raw = doc.to_string();
        let parsed_cfg =
            crate::config::parse_config_text(&raw, &services.config_path.display().to_string())
                .map_err(|err| {
                    AppError::Config(format!("failed to parse updated config: {err}"))
                })?;
        crate::config::validate_config(&parsed_cfg)?;
        write_config_document(services.config_path, &raw)?;
        Ok(())
    })();
    if let Err(err) = result {
        state.status = format!(
            "{}: {}",
            ui_text_config_save_failed(),
            i18n::localize_error(&err)
        );
        return Ok(());
    }
    for field in &mut state.config_ui.fields {
        field.dirty = false;
    }
    state.config_ui.dirty_count = 0;
    state.config_ui.editing = false;
    state.status = ui_text_config_saved().to_string();
    Ok(())
}

fn parse_config_field_to_item(field: &ConfigField) -> Result<Option<Item>, AppError> {
    let text = field.value.trim();
    let item = match field.kind {
        ConfigFieldKind::Bool => {
            let parsed = text
                .parse::<bool>()
                .map_err(|_| AppError::Config(format!("{} must be true/false", field.key)))?;
            Some(Item::Value(Value::from(parsed)))
        }
        ConfigFieldKind::String | ConfigFieldKind::Enum => {
            if field.required && text.is_empty() {
                return Err(AppError::Config(format!("{} must not be empty", field.key)));
            }
            if field.kind == ConfigFieldKind::Enum
                && !field.options.is_empty()
                && !field
                    .options
                    .iter()
                    .any(|item| item.eq_ignore_ascii_case(text))
            {
                return Err(AppError::Config(format!(
                    "{} must be one of: {}",
                    field.key,
                    field.options.join(", ")
                )));
            }
            Some(Item::Value(Value::from(text.to_string())))
        }
        ConfigFieldKind::OptionalString => {
            if text.is_empty() {
                None
            } else {
                Some(Item::Value(Value::from(text.to_string())))
            }
        }
        ConfigFieldKind::U8 => {
            let parsed = text
                .parse::<u8>()
                .map_err(|_| AppError::Config(format!("{} must be u8", field.key)))?;
            Some(Item::Value(Value::from(parsed as i64)))
        }
        ConfigFieldKind::U32 => {
            let parsed = text
                .parse::<u32>()
                .map_err(|_| AppError::Config(format!("{} must be u32", field.key)))?;
            Some(Item::Value(Value::from(parsed as i64)))
        }
        ConfigFieldKind::U64 => {
            let parsed = text
                .parse::<u64>()
                .map_err(|_| AppError::Config(format!("{} must be u64", field.key)))?;
            Some(Item::Value(Value::from(parsed as i64)))
        }
        ConfigFieldKind::Usize => {
            let parsed = text
                .parse::<usize>()
                .map_err(|_| AppError::Config(format!("{} must be usize", field.key)))?;
            Some(Item::Value(Value::from(parsed as i64)))
        }
        ConfigFieldKind::OptionalU64 => {
            if text.is_empty() {
                None
            } else {
                let parsed = text
                    .parse::<u64>()
                    .map_err(|_| AppError::Config(format!("{} must be u64", field.key)))?;
                Some(Item::Value(Value::from(parsed as i64)))
            }
        }
        ConfigFieldKind::F64 => {
            let parsed = text
                .parse::<f64>()
                .map_err(|_| AppError::Config(format!("{} must be f64", field.key)))?;
            Some(Item::Value(Value::from(parsed)))
        }
        ConfigFieldKind::StringList => {
            let literal = if text.is_empty() { "[]" } else { text };
            Some(parse_toml_literal_to_item(literal)?)
        }
        ConfigFieldKind::StringMap => {
            let literal = if text.is_empty() { "{}" } else { text };
            Some(parse_toml_literal_to_item(literal)?)
        }
    };
    Ok(item)
}

fn parse_toml_literal_to_item(literal: &str) -> Result<Item, AppError> {
    let snippet = format!("value = {literal}");
    let doc = snippet
        .parse::<DocumentMut>()
        .map_err(|err| AppError::Config(format!("failed to parse value '{literal}': {err}")))?;
    doc.get("value")
        .cloned()
        .ok_or_else(|| AppError::Config("parsed value not found".to_string()))
}

fn read_config_document(path: &Path) -> Result<DocumentMut, AppError> {
    if !path.exists() {
        return Ok(DocumentMut::new());
    }
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Config(format!("failed to read config {}: {err}", path.display()))
    })?;
    if raw.trim().is_empty() {
        return Ok(DocumentMut::new());
    }
    raw.parse::<DocumentMut>().map_err(|err| {
        AppError::Config(format!("failed to parse config {}: {err}", path.display()))
    })
}

fn write_config_document(path: &Path, content: &str) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Config(format!(
                "failed to create config directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("claw.toml");
    let tmp_path =
        path.with_file_name(format!(".{file_name}.{}.{}.tmp", std::process::id(), stamp));
    fs::write(&tmp_path, content).map_err(|err| {
        AppError::Config(format!(
            "failed to write temporary config file {}: {err}",
            tmp_path.display()
        ))
    })?;
    if let Err(rename_err) = fs::rename(&tmp_path, path) {
        if let Err(copy_err) = fs::copy(&tmp_path, path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(AppError::Config(format!(
                "failed to replace config file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            )));
        }
        let _ = fs::remove_file(&tmp_path);
    }
    Ok(())
}

fn split_config_key_path(key: &str) -> Vec<&str> {
    key.split('.')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect()
}

fn set_config_item_by_path(doc: &mut DocumentMut, key: &str, value: Item) -> Result<(), AppError> {
    let segments = split_config_key_path(key);
    if segments.is_empty() {
        return Err(AppError::Config("config key is empty".to_string()));
    }
    let mut current = doc.as_item_mut();
    for segment in &segments[..segments.len() - 1] {
        let table = current.as_table_like_mut().ok_or_else(|| {
            AppError::Config(format!(
                "config path conflict at '{}': not a table",
                segment
            ))
        })?;
        if table.get(segment).is_none() {
            table.insert(segment, Item::Table(TomlTable::new()));
        }
        let next = table.get_mut(segment).ok_or_else(|| {
            AppError::Config(format!(
                "failed to navigate config key at '{}': missing node",
                segment
            ))
        })?;
        if next.is_none() {
            *next = Item::Table(TomlTable::new());
        }
        if !next.is_table() {
            return Err(AppError::Config(format!(
                "config path conflict at '{}': target is not a table",
                segment
            )));
        }
        current = next;
    }
    let leaf = segments[segments.len() - 1];
    let table = current.as_table_like_mut().ok_or_else(|| {
        AppError::Config(format!(
            "failed to set config key '{}': parent is not table",
            key
        ))
    })?;
    table.insert(leaf, value);
    Ok(())
}

fn remove_config_item_by_path(doc: &mut DocumentMut, key: &str) {
    let segments = split_config_key_path(key);
    if segments.len() < 2 {
        return;
    }
    let mut current = doc.as_item_mut();
    for segment in &segments[..segments.len() - 1] {
        let Some(table) = current.as_table_like_mut() else {
            return;
        };
        let Some(next) = table.get_mut(segment) else {
            return;
        };
        current = next;
    }
    if let Some(table) = current.as_table_like_mut() {
        let leaf = segments[segments.len() - 1];
        let _ = table.remove(leaf);
    }
}

fn config_select_by_mouse(layout: UiLayout, mouse: MouseEvent, state: &mut ChatUiState) {
    if state.mode != UiMode::Config {
        return;
    }
    let (category_rect, fields_rect) = config_panel_regions(layout);
    if rect_contains(category_rect, mouse.column, mouse.row) {
        let inner = category_rect.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        let idx = mouse.row.saturating_sub(inner.y) as usize;
        if idx < state.config_ui.categories.len() {
            state.config_ui.selected_category = idx;
            state.config_ui.selected_field_row = 0;
            state.config_ui.editing = false;
        }
        return;
    }
    if rect_contains(fields_rect, mouse.column, mouse.row) {
        let inner = fields_rect.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        if mouse.row <= inner.y {
            return;
        }
        let row = mouse.row.saturating_sub(inner.y + 1) as usize;
        let len = config_visible_field_indices(state).len();
        if len == 0 {
            return;
        }
        state.config_ui.selected_field_row = row.min(len - 1);
        state.config_ui.editing = false;
    }
}

fn config_panel_regions(layout: UiLayout) -> (Rect, Rect) {
    let inner = layout.conversation.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(26), Constraint::Min(20)])
        .split(inner);
    (cols[0], cols[1])
}

fn mcp_panel_regions(layout: UiLayout) -> (Rect, Rect) {
    let inner = layout.conversation.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(8)])
        .split(inner);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(20)])
        .split(rows[1]);
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(7), Constraint::Length(5)])
        .split(cols[1]);
    (cols[0], right_rows[0])
}

fn mcp_selected_server(state: &ChatUiState) -> Option<&McpUiServer> {
    state.mcp_ui.servers.get(
        state
            .mcp_ui
            .selected_server
            .min(state.mcp_ui.servers.len().saturating_sub(1)),
    )
}

fn mcp_selected_field_def(state: &ChatUiState) -> Option<McpFieldDef> {
    mcp_field_defs()
        .get(
            state
                .mcp_ui
                .selected_field
                .min(mcp_field_defs().len().saturating_sub(1)),
        )
        .copied()
}

fn mcp_field_value(server: &McpUiServer, id: McpFieldId) -> String {
    match id {
        McpFieldId::Name => server.name.clone(),
        McpFieldId::Enabled => bool_to_text(server.config.enabled),
        McpFieldId::Transport => server.config.transport.clone().unwrap_or_default(),
        McpFieldId::Url => server.config.server_url.clone().unwrap_or_default(),
        McpFieldId::Endpoint => server.config.endpoint.clone().unwrap_or_default(),
        McpFieldId::Command => server.config.command.clone().unwrap_or_default(),
        McpFieldId::Args => to_toml_string_array(&server.config.args),
        McpFieldId::Env => to_toml_string_map(&server.config.env),
        McpFieldId::Headers => to_toml_string_map(&server.config.headers),
        McpFieldId::AuthType => server.config.auth_type.clone().unwrap_or_default(),
        McpFieldId::AuthToken => server.config.auth_token.clone().unwrap_or_default(),
        McpFieldId::TimeoutSeconds => server
            .config
            .timeout_seconds
            .map(|value| value.to_string())
            .unwrap_or_default(),
    }
}

fn mcp_scroll_servers(state: &mut ChatUiState, delta: i16) {
    let len = state.mcp_ui.servers.len() as i32;
    if len <= 0 {
        state.mcp_ui.selected_server = 0;
        return;
    }
    let mut next = state.mcp_ui.selected_server as i32 + delta as i32;
    if next < 0 {
        next = 0;
    }
    if next >= len {
        next = len - 1;
    }
    state.mcp_ui.selected_server = next as usize;
    state.mcp_ui.editing = false;
}

fn mcp_scroll_fields(state: &mut ChatUiState, delta: i16) {
    let len = mcp_field_defs().len() as i32;
    if len <= 0 {
        state.mcp_ui.selected_field = 0;
        return;
    }
    let mut next = state.mcp_ui.selected_field as i32 + delta as i32;
    if next < 0 {
        next = 0;
    }
    if next >= len {
        next = len - 1;
    }
    state.mcp_ui.selected_field = next as usize;
    state.mcp_ui.editing = false;
}

fn mcp_start_edit(state: &mut ChatUiState, id: McpFieldId) -> Result<(), AppError> {
    if let Some(def) = mcp_field_defs().iter().find(|item| item.id == id).copied()
        && mcp_field_is_selectable(def)
    {
        state.mcp_ui.editing = false;
        return Ok(());
    }
    let Some(current_value) = mcp_selected_server(state).map(|server| mcp_field_value(server, id))
    else {
        return Err(AppError::Config(ui_text_mcp_empty().to_string()));
    };
    state.mcp_ui.editing = true;
    state.mcp_ui.edit_buffer = InputBuffer {
        text: current_value.clone(),
        cursor_char: current_value.chars().count(),
        view_char_offset: 0,
    };
    state.status = ui_text_mcp_editing().to_string();
    Ok(())
}

fn mcp_apply_edit_buffer(state: &mut ChatUiState) -> Result<(), AppError> {
    let Some(field) = mcp_selected_field_def(state) else {
        return Ok(());
    };
    let next_text = state.mcp_ui.edit_buffer.text.trim().to_string();
    mcp_apply_field_value(state, field.id, &next_text)?;
    state.mcp_ui.editing = false;
    state.status = ui_text_mcp_edit_applied().to_string();
    Ok(())
}

fn mcp_apply_field_value(
    state: &mut ChatUiState,
    id: McpFieldId,
    next_text: &str,
) -> Result<(), AppError> {
    if state.mcp_ui.servers.is_empty() {
        return Ok(());
    }
    let selected_idx = state
        .mcp_ui
        .selected_server
        .min(state.mcp_ui.servers.len().saturating_sub(1));
    let current = mcp_field_value(&state.mcp_ui.servers[selected_idx], id);
    if current.trim() == next_text.trim() {
        return Ok(());
    }
    if matches!(id, McpFieldId::Name) {
        let new_name = next_text.trim();
        if new_name.is_empty() {
            return Err(AppError::Config(
                "MCP server name must not be empty".to_string(),
            ));
        }
        if state
            .mcp_ui
            .servers
            .iter()
            .enumerate()
            .any(|(idx, item)| idx != selected_idx && item.name == new_name)
        {
            return Err(AppError::Config(format!(
                "duplicated MCP server name: {new_name}"
            )));
        }
    }
    let Some(server) = state.mcp_ui.servers.get_mut(selected_idx) else {
        return Ok(());
    };
    match id {
        McpFieldId::Name => {
            server.name = next_text.trim().to_string();
            state.mcp_ui.structural_dirty = true;
        }
        McpFieldId::Enabled => {
            let parsed = next_text
                .parse::<bool>()
                .map_err(|_| AppError::Config("enabled must be true/false".to_string()))?;
            server.config.enabled = parsed;
        }
        McpFieldId::Transport => {
            let trimmed = next_text.trim();
            server.config.transport = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::Url => {
            let trimmed = next_text.trim();
            server.config.server_url = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::Endpoint => {
            let trimmed = next_text.trim();
            server.config.endpoint = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::Command => {
            let trimmed = next_text.trim();
            server.config.command = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::Args => {
            server.config.args = parse_toml_string_list(next_text, "args")?;
        }
        McpFieldId::Env => {
            server.config.env = parse_toml_string_map(next_text, "env")?;
        }
        McpFieldId::Headers => {
            server.config.headers = parse_toml_string_map(next_text, "headers")?;
        }
        McpFieldId::AuthType => {
            let trimmed = next_text.trim();
            server.config.auth_type = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::AuthToken => {
            let trimmed = next_text.trim();
            server.config.auth_token = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        McpFieldId::TimeoutSeconds => {
            let trimmed = next_text.trim();
            if trimmed.is_empty() {
                server.config.timeout_seconds = None;
            } else {
                let parsed = trimmed
                    .parse::<u64>()
                    .map_err(|_| AppError::Config("timeoutSeconds must be u64".to_string()))?;
                server.config.timeout_seconds = Some(parsed);
            }
        }
    }
    server.dirty = true;
    mcp_recompute_dirty_count(state);
    Ok(())
}

fn mcp_cycle_field_option(
    state: &mut ChatUiState,
    id: McpFieldId,
    step: i16,
) -> Result<(), AppError> {
    let Some(def) = mcp_selected_field_def(state) else {
        return Ok(());
    };
    if def.id != id {
        return Ok(());
    }
    if !def.options.is_empty() {
        let current = mcp_selected_server(state)
            .map(|item| mcp_field_value(item, id))
            .unwrap_or_default();
        let base = def
            .options
            .iter()
            .position(|item| item.eq_ignore_ascii_case(current.trim()))
            .map(|idx| idx as i32)
            .unwrap_or_else(|| {
                if step >= 0 {
                    -1
                } else {
                    def.options.len() as i32
                }
            });
        let mut idx = base + step as i32;
        if idx < 0 {
            idx = 0;
        }
        if idx >= def.options.len() as i32 {
            idx = def.options.len() as i32 - 1;
        }
        mcp_apply_field_value(state, id, def.options[idx as usize])
    } else if def.kind == ConfigFieldKind::Bool {
        let next = if mcp_selected_server(state).is_some_and(|item| item.config.enabled) {
            "false"
        } else {
            "true"
        };
        mcp_apply_field_value(state, id, next)
    } else {
        Ok(())
    }
}

fn mcp_field_is_selectable(def: McpFieldDef) -> bool {
    matches!(def.kind, ConfigFieldKind::Bool | ConfigFieldKind::Enum) || !def.options.is_empty()
}

fn mcp_should_refresh_runtime_metadata(
    has_enabled_servers: bool,
    has_tools: bool,
    any_service_available: bool,
    summary: &str,
) -> bool {
    if !has_enabled_servers {
        return false;
    }
    if summary.contains("availability=checking") {
        return true;
    }
    !has_tools && !any_service_available
}

fn refresh_mcp_runtime_metadata_if_needed(
    services: &mut ActionServices<'_>,
    state: &mut ChatUiState,
    report_result_to_chat: bool,
) {
    if !services.cfg.ai.tools.mcp.enabled {
        return;
    }
    let has_enabled_servers = state.mcp_ui.servers.iter().any(|item| item.config.enabled);
    let has_tools = !services.mcp.tool_statuses().is_empty();
    let any_service_available = services
        .mcp
        .service_statuses()
        .iter()
        .any(|item| item.available);
    let summary_before = services.mcp.summary();
    if !mcp_should_refresh_runtime_metadata(
        has_enabled_servers,
        has_tools,
        any_service_available,
        &summary_before,
    ) {
        return;
    }
    if report_result_to_chat && summary_before.contains("availability=checking") {
        state.push(UiRole::System, i18n::chat_mcp_availability_check_started());
    }
    match mcp::McpManager::connect(&services.cfg.ai.tools.mcp, services.config_path) {
        Ok(manager) => {
            *services.mcp = manager;
            services.mcp_summary = services.mcp.summary();
            state.mcp_ui.last_error = None;
            if report_result_to_chat {
                let tool_count = services.mcp.external_tool_definitions().len();
                state.push(
                    UiRole::System,
                    i18n::chat_mcp_availability_check_finished(tool_count, &services.mcp_summary),
                );
            }
        }
        Err(err) => {
            let detail = i18n::localize_error(&err);
            state.mcp_ui.last_error = Some(mask_ui_sensitive(&detail));
            if report_result_to_chat {
                state.push(UiRole::System, format!("MCP: {detail}"));
            }
        }
    }
}

fn mcp_add_server(state: &mut ChatUiState) {
    let next_name = mcp_next_server_name(state);
    state.mcp_ui.servers.push(McpUiServer {
        name: next_name.clone(),
        config: McpServerConfig {
            enabled: true,
            ..McpServerConfig::default()
        },
        dirty: true,
    });
    state.mcp_ui.selected_server = state.mcp_ui.servers.len().saturating_sub(1);
    state.mcp_ui.focus_servers = true;
    state.mcp_ui.structural_dirty = true;
    mcp_recompute_dirty_count(state);
    state.status = format!("{}: {}", ui_text_mcp_server_added(), next_name);
}

fn mcp_delete_selected_server(state: &mut ChatUiState) {
    if state.mcp_ui.servers.is_empty() {
        return;
    }
    let idx = state
        .mcp_ui
        .selected_server
        .min(state.mcp_ui.servers.len().saturating_sub(1));
    let removed = state.mcp_ui.servers.remove(idx);
    if state.mcp_ui.selected_server >= state.mcp_ui.servers.len()
        && !state.mcp_ui.servers.is_empty()
    {
        state.mcp_ui.selected_server = state.mcp_ui.servers.len() - 1;
    }
    state.mcp_ui.structural_dirty = true;
    mcp_recompute_dirty_count(state);
    state.status = format!("{}: {}", ui_text_mcp_server_deleted(), removed.name);
}

fn mcp_next_server_name(state: &ChatUiState) -> String {
    for idx in 1..=9_999 {
        let candidate = format!("new-mcp-{idx}");
        if !state
            .mcp_ui
            .servers
            .iter()
            .any(|item| item.name == candidate)
        {
            return candidate;
        }
    }
    format!("new-mcp-{}", now_epoch_ms())
}

fn mcp_recompute_dirty_count(state: &mut ChatUiState) {
    let field_dirty = state
        .mcp_ui
        .servers
        .iter()
        .filter(|item| item.dirty)
        .count();
    let structural = usize::from(state.mcp_ui.structural_dirty);
    state.mcp_ui.dirty_count = field_dirty + structural;
}

fn parse_toml_string_list(raw: &str, label: &str) -> Result<Vec<String>, AppError> {
    let literal = if raw.trim().is_empty() { "[]" } else { raw };
    let snippet = format!("value = {literal}");
    let parsed = toml::from_str::<toml::Table>(&snippet)
        .map_err(|err| AppError::Config(format!("failed to parse {label} as array: {err}")))?;
    let Some(value) = parsed.get("value") else {
        return Ok(Vec::new());
    };
    let Some(arr) = value.as_array() else {
        return Err(AppError::Config(format!("{label} must be array<string>")));
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(text) = item.as_str() else {
            return Err(AppError::Config(format!("{label} must be array<string>")));
        };
        out.push(text.to_string());
    }
    Ok(out)
}

fn parse_toml_string_map(raw: &str, label: &str) -> Result<BTreeMap<String, String>, AppError> {
    let literal = if raw.trim().is_empty() { "{}" } else { raw };
    let snippet = format!("value = {literal}");
    let parsed = toml::from_str::<toml::Table>(&snippet)
        .map_err(|err| AppError::Config(format!("failed to parse {label} as map: {err}")))?;
    let Some(value) = parsed.get("value") else {
        return Ok(BTreeMap::new());
    };
    let Some(table) = value.as_table() else {
        return Err(AppError::Config(format!(
            "{label} must be map<string,string>"
        )));
    };
    let mut out = BTreeMap::new();
    for (key, item) in table {
        let Some(text) = item.as_str() else {
            return Err(AppError::Config(format!(
                "{label} must be map<string,string>"
            )));
        };
        out.insert(key.to_string(), text.to_string());
    }
    Ok(out)
}

fn save_mcp_ui(services: &mut ActionServices<'_>, state: &mut ChatUiState) -> Result<(), AppError> {
    let result = (|| -> Result<PathBuf, AppError> {
        let mut records = Vec::<McpServerRecord>::new();
        for server in &state.mcp_ui.servers {
            records.push(McpServerRecord {
                name: server.name.trim().to_string(),
                config: server.config.clone(),
            });
        }
        let saved_path = mcp::save_mcp_server_records(
            &services.cfg.ai.tools.mcp,
            services.config_path,
            &records,
        )?;
        if services.cfg.ai.tools.mcp.enabled {
            *services.mcp =
                mcp::McpManager::connect(&services.cfg.ai.tools.mcp, services.config_path)?;
            services.mcp_summary = services.mcp.summary();
        }
        Ok(saved_path)
    })();
    let saved_path = match result {
        Ok(path) => path,
        Err(err) => {
            let detail = i18n::localize_error(&err);
            state.mcp_ui.last_error = Some(mask_ui_sensitive(&detail));
            state.status = format!("{}: {}", ui_text_mcp_error(), detail);
            return Ok(());
        }
    };
    state.mcp_ui.config_file_path = saved_path;
    for server in &mut state.mcp_ui.servers {
        server.dirty = false;
    }
    state.mcp_ui.dirty_count = 0;
    state.mcp_ui.structural_dirty = false;
    state.mcp_ui.editing = false;
    state.mcp_ui.last_error = None;
    state.status = ui_text_mcp_saved().to_string();
    Ok(())
}

fn mcp_select_by_mouse(layout: UiLayout, mouse: MouseEvent, state: &mut ChatUiState) {
    if state.mode != UiMode::Mcp {
        return;
    }
    let (servers_rect, fields_rect) = mcp_panel_regions(layout);
    if rect_contains(servers_rect, mouse.column, mouse.row) {
        let inner = servers_rect.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        let idx = mouse.row.saturating_sub(inner.y) as usize;
        if idx < state.mcp_ui.servers.len() {
            state.mcp_ui.selected_server = idx;
            state.mcp_ui.focus_servers = true;
            state.mcp_ui.editing = false;
        }
        return;
    }
    if rect_contains(fields_rect, mouse.column, mouse.row) {
        let inner = fields_rect.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        if mouse.row <= inner.y {
            return;
        }
        let row = mouse.row.saturating_sub(inner.y + 1) as usize;
        let len = mcp_field_defs().len();
        if len == 0 {
            return;
        }
        state.mcp_ui.selected_field = row.min(len - 1);
        state.mcp_ui.focus_servers = false;
        state.mcp_ui.editing = false;
    }
}

fn mcp_scroll_by_mouse(layout: UiLayout, mouse: MouseEvent, state: &mut ChatUiState, delta: i16) {
    if state.mode != UiMode::Mcp {
        return;
    }
    let (servers_rect, _) = mcp_panel_regions(layout);
    if rect_contains(servers_rect, mouse.column, mouse.row) {
        mcp_scroll_servers(state, delta);
        state.mcp_ui.focus_servers = true;
    } else {
        mcp_scroll_fields(state, delta);
        state.mcp_ui.focus_servers = false;
    }
}

fn compute_layout(area: Rect) -> UiLayout {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(36), Constraint::Min(40)])
        .split(area);
    let sidebar_inner = columns[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let sidebar_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(sidebar_inner);
    let main_inner = columns[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let main_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(8),
            Constraint::Length(6),
            Constraint::Length(1),
        ])
        .split(main_inner);
    UiLayout {
        sidebar: columns[0],
        nav: sidebar_rows[0],
        threads: sidebar_rows[1],
        threads_body: sidebar_rows[1].inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        }),
        footer: sidebar_rows[2],
        header: main_rows[0],
        conversation: main_rows[1],
        conversation_body: main_rows[1].inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        }),
        input: main_rows[2],
        input_body: main_rows[2].inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        }),
        status: main_rows[3],
    }
}

fn conversation_viewport_height(
    terminal: &Terminal<CrosstermBackend<Stdout>>,
) -> Result<u16, AppError> {
    let size = terminal
        .size()
        .map_err(|err| AppError::Command(format!("failed to get terminal size: {err}")))?;
    Ok(compute_layout(Rect::new(0, 0, size.width, size.height))
        .conversation_body
        .height
        .max(1))
}

fn centered_rect(area: Rect, percent_x: u16, height: u16) -> Rect {
    let width = (area.width as u32 * percent_x as u32 / 100).max(20) as u16;
    let h = height.min(area.height.saturating_sub(2)).max(3);
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    Rect {
        x,
        y,
        width: width.min(area.width.saturating_sub(1)).max(10),
        height: h,
    }
}

fn delete_message_modal_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.conversation, 74, 11)
}

fn thread_action_menu_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.threads, 86, 8)
}

fn thread_delete_modal_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.conversation, 66, 11)
}

fn thread_metadata_modal_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.conversation, 84, 18)
}

fn thread_rename_modal_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.conversation, 62, 9)
}

fn tool_result_modal_rect(layout: UiLayout) -> Rect {
    centered_rect(layout.conversation, 88, 24)
}

fn tool_result_modal_scroll_max(modal: &ToolResultModalState, layout: UiLayout) -> u16 {
    let rect = tool_result_modal_rect(layout);
    let inner = rect.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(1)])
        .split(inner);
    let text_width = rows[0].width.saturating_sub(2).max(1) as usize;
    let viewport_height = rows[0].height.saturating_sub(2).max(1) as usize;
    let wrapped_lines = modal
        .lines
        .iter()
        .map(|line| {
            line.split('\n')
                .map(|part| wrap_text_by_display_width(part, text_width).len().max(1))
                .sum::<usize>()
                .max(1)
        })
        .sum::<usize>()
        .max(1);
    wrapped_lines
        .saturating_sub(viewport_height)
        .min(u16::MAX as usize) as u16
}

fn thread_delete_modal_button_rects(modal: Rect) -> (Rect, Rect) {
    delete_message_modal_button_rects(modal)
}

fn delete_message_modal_button_rects(modal: Rect) -> (Rect, Rect) {
    let body = modal.inner(ratatui::layout::Margin {
        horizontal: 2,
        vertical: 1,
    });
    let row_y = body.y.saturating_add(
        body.height
            .saturating_sub(3)
            .min(body.height.saturating_sub(1)),
    );
    let inner_width = body.width.max(4);
    let half = inner_width / 2;
    let confirm = Rect {
        x: body.x,
        y: row_y,
        width: half.max(2),
        height: 1,
    };
    let cancel = Rect {
        x: body.x.saturating_add(half).saturating_add(1),
        y: row_y,
        width: inner_width.saturating_sub(half).saturating_sub(1).max(2),
        height: 1,
    };
    (confirm, cancel)
}

fn draw_modal_overlay(frame: &mut Frame<'_>, palette: ThemePalette) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default().style(Style::default().bg(palette.app_bg)),
        area,
    );
}

fn panel_border_color(focused: bool, palette: ThemePalette) -> Color {
    if focused {
        palette.border_focus
    } else {
        palette.border
    }
}

fn role_tag(role: UiRole) -> &'static str {
    match role {
        UiRole::User => ui_text_tag_user(),
        UiRole::Assistant => ui_text_tag_ai(),
        UiRole::Thinking => ui_text_tag_thinking(),
        UiRole::System => ui_text_tag_sys(),
        UiRole::Tool => ui_text_tag_tool(),
    }
}

fn role_tag_color(role: UiRole, palette: ThemePalette) -> Color {
    match role {
        UiRole::User => palette.accent,
        UiRole::Assistant => Color::Rgb(114, 214, 155),
        UiRole::Thinking => Color::Rgb(223, 167, 255),
        UiRole::System => Color::Rgb(255, 199, 94),
        UiRole::Tool => Color::Rgb(245, 148, 203),
    }
}

fn role_border_color(role: UiRole, palette: ThemePalette) -> Color {
    match role {
        UiRole::User => palette.accent,
        UiRole::Assistant => Color::Rgb(98, 168, 128),
        UiRole::Thinking => Color::Rgb(154, 106, 196),
        UiRole::System => Color::Rgb(201, 153, 59),
        UiRole::Tool => Color::Rgb(196, 107, 160),
    }
}

fn role_body_color(role: UiRole, palette: ThemePalette) -> Color {
    match role {
        UiRole::User => palette.text,
        UiRole::Assistant => Color::Rgb(224, 248, 232),
        UiRole::Thinking => Color::Rgb(246, 230, 255),
        UiRole::System => Color::Rgb(255, 240, 208),
        UiRole::Tool => Color::Rgb(255, 224, 244),
    }
}

fn spinner_frame(started_at: Instant) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let tick = (started_at.elapsed().as_millis() / 90) as usize;
    FRAMES[tick % FRAMES.len()]
}

fn estimate_tokens_from_text_delta(text: &str) -> u64 {
    let meaningful = text.chars().filter(|ch| !ch.is_whitespace()).count() as u64;
    if meaningful == 0 {
        0
    } else {
        meaningful.div_ceil(3).clamp(1, 512)
    }
}

fn format_u64_compact(value: u64) -> String {
    let raw = value.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3);
    for (seen, ch) in raw.chars().rev().enumerate() {
        if seen != 0 && seen.is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn thread_action_menu_options() -> [&'static str; 3] {
    [
        ui_text_thread_action_delete(),
        ui_text_thread_action_rename(),
        ui_text_thread_action_metadata(),
    ]
}

fn metadata_report_to_rows(report: &str) -> Vec<(String, String)> {
    let mut rows = Vec::<(String, String)>::new();
    for line in report.lines() {
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if let Some((key, value)) = text.split_once(':').or_else(|| text.split_once('：')) {
            let key_text = key.trim();
            let value_text = value.trim();
            if !key_text.is_empty() && !value_text.is_empty() {
                rows.push((key_text.to_string(), value_text.to_string()));
            }
        }
    }
    if rows.is_empty() {
        rows.push((ui_text_na().to_string(), trim_ui_text(report, 180)));
    }
    rows
}

fn build_session_metadata_report(services: &ActionServices<'_>, state: &ChatUiState) -> String {
    let active = current_thread_overview(state, services)
        .cloned()
        .or_else(|| {
            services.session.list_sessions().ok().and_then(|items| {
                items
                    .into_iter()
                    .find(|item| item.session_id == services.session.session_id())
            })
        });
    let archived = services.session.archived_role_counts();
    let effective = services.session.effective_context_role_counts(true);
    let compass = services.session.compass();
    let created_at = active
        .as_ref()
        .map(|item| item.created_at_epoch_ms)
        .unwrap_or(compass.created_at_epoch_ms);
    let updated_at = active
        .as_ref()
        .map(|item| item.last_updated_epoch_ms)
        .unwrap_or(compass.last_updated_epoch_ms);
    let message_count = active
        .as_ref()
        .map(|item| item.message_count)
        .unwrap_or_else(|| services.session.message_count());
    let summary_len = active
        .as_ref()
        .map(|item| item.summary_len)
        .unwrap_or_else(|| services.session.summary_len());
    let context_max = services.cfg.session.max_messages.max(1);
    let context_usage = archived
        .total
        .saturating_mul(100)
        .saturating_div(context_max)
        .min(100);
    let total_chars = services.session.total_message_chars();
    if lang_is_zh() {
        format!(
            "会话元数据\nID: {}\n名称: {}\n文件: {}\n创建时间: {}\n最近更新: {}\n消息总数: {} (用户 {} / 助手 {} / 工具 {} / 系统 {})\n有效上下文: {} (用户 {} / 助手 {} / 工具 {} / 系统 {})\n摘要长度: {} 字符\n消息总字符数: {}\n上下文占用: {}% (recent={}, max={})\n压缩统计: rounds={}, truncated={}, dropped_groups={}\n运行期 Token: committed={}, live_estimate={}, display={}",
            services.session.session_id(),
            services.session.session_name(),
            services.session.file_path().display(),
            format_epoch_ms(created_at),
            format_epoch_ms(updated_at),
            message_count,
            archived.user,
            archived.assistant,
            archived.tool,
            archived.system,
            effective.total,
            effective.user,
            effective.assistant,
            effective.tool,
            effective.system,
            summary_len,
            total_chars,
            context_usage,
            services.cfg.session.recent_messages,
            services.cfg.session.max_messages,
            compass.compression_rounds,
            compass.truncated_messages,
            compass.dropped_groups,
            format_u64_compact(state.token_usage_committed),
            format_u64_compact(state.token_live_estimate),
            format_u64_compact(state.token_display_value),
        )
    } else {
        format!(
            "Session metadata\nID: {}\nName: {}\nFile: {}\nCreated: {}\nUpdated: {}\nMessages: {} (user {} / assistant {} / tool {} / system {})\nEffective context: {} (user {} / assistant {} / tool {} / system {})\nSummary chars: {}\nMessage chars total: {}\nContext usage: {}% (recent={}, max={})\nCompression: rounds={}, truncated={}, dropped_groups={}\nRuntime token: committed={}, live_estimate={}, display={}",
            services.session.session_id(),
            services.session.session_name(),
            services.session.file_path().display(),
            format_epoch_ms(created_at),
            format_epoch_ms(updated_at),
            message_count,
            archived.user,
            archived.assistant,
            archived.tool,
            archived.system,
            effective.total,
            effective.user,
            effective.assistant,
            effective.tool,
            effective.system,
            summary_len,
            total_chars,
            context_usage,
            services.cfg.session.recent_messages,
            services.cfg.session.max_messages,
            compass.compression_rounds,
            compass.truncated_messages,
            compass.dropped_groups,
            format_u64_compact(state.token_usage_committed),
            format_u64_compact(state.token_live_estimate),
            format_u64_compact(state.token_display_value),
        )
    }
}

fn build_thread_metadata_report(
    services: &ActionServices<'_>,
    session: &SessionOverview,
) -> String {
    let context_max = services.cfg.session.max_messages.max(1);
    let context_usage = session
        .message_count
        .saturating_mul(100)
        .saturating_div(context_max)
        .min(100);
    if lang_is_zh() {
        format!(
            "会话元数据\nID: {}\n名称: {}\n文件: {}\n创建时间: {}\n最近更新: {}\n消息总数: {} (用户 {} / 助手 {} / 工具 {} / 系统 {})\n摘要长度: {} 字符\n上下文占用: {}% (recent={}, max={})\n活跃会话: 否\n说明: 该会话未激活，运行期 Token 与压缩运行态指标需切换后查看",
            session.session_id,
            session.session_name,
            session.file_path.display(),
            format_epoch_ms(session.created_at_epoch_ms),
            format_epoch_ms(session.last_updated_epoch_ms),
            session.message_count,
            session.user_count,
            session.assistant_count,
            session.tool_count,
            session.system_count,
            session.summary_len,
            context_usage,
            services.cfg.session.recent_messages,
            services.cfg.session.max_messages,
        )
    } else {
        format!(
            "Session metadata\nID: {}\nName: {}\nFile: {}\nCreated: {}\nUpdated: {}\nMessages: {} (user {} / assistant {} / tool {} / system {})\nSummary chars: {}\nContext usage: {}% (recent={}, max={})\nActive session: no\nNote: this session is inactive; runtime token and compression-live metrics are available after switching to it",
            session.session_id,
            session.session_name,
            session.file_path.display(),
            format_epoch_ms(session.created_at_epoch_ms),
            format_epoch_ms(session.last_updated_epoch_ms),
            session.message_count,
            session.user_count,
            session.assistant_count,
            session.tool_count,
            session.system_count,
            session.summary_len,
            context_usage,
            services.cfg.session.recent_messages,
            services.cfg.session.max_messages,
        )
    }
}

fn current_thread_overview<'a>(
    state: &'a ChatUiState,
    services: &'a ActionServices<'_>,
) -> Option<&'a SessionOverview> {
    let session_id = services.session.session_id();
    state
        .threads
        .iter()
        .find(|item| item.session_id == session_id)
        .or_else(|| {
            state
                .threads
                .get(state.thread_selected)
                .filter(|item| item.active)
        })
}

fn short_session_id(raw: &str) -> String {
    if raw.chars().count() <= 10 {
        return raw.to_string();
    }
    let prefix = raw.chars().take(8).collect::<String>();
    format!("{prefix}…")
}

fn format_epoch_ms(epoch_ms: u128) -> String {
    if epoch_ms == 0 {
        return ui_text_na().to_string();
    }
    let millis = epoch_ms.min(i64::MAX as u128) as i64;
    match Local.timestamp_millis_opt(millis).single() {
        Some(ts) => ts.format("%m-%d %H:%M:%S").to_string(),
        None => ui_text_na().to_string(),
    }
}

fn rect_contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x
        && x < rect.x.saturating_add(rect.width)
        && y >= rect.y
        && y < rect.y.saturating_add(rect.height)
}

fn copy_text_to_clipboard(text: &str) -> Result<(), AppError> {
    #[cfg(target_os = "macos")]
    {
        run_clipboard_command("pbcopy", &[], text)?;
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        run_clipboard_command("clip", &[], text)?;
        return Ok(());
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let candidates = [
            ("wl-copy", Vec::<&str>::new()),
            ("xclip", vec!["-selection", "clipboard"]),
            ("xsel", vec!["--clipboard", "--input"]),
        ];
        let mut last_err = String::new();
        for (program, args) in candidates {
            match run_clipboard_command(program, &args, text) {
                Ok(()) => return Ok(()),
                Err(err) => {
                    last_err = err.to_string();
                }
            }
        }
        return Err(AppError::Command(format!(
            "failed to copy text to clipboard: {last_err}"
        )));
    }
    #[allow(unreachable_code)]
    Err(AppError::Command(
        "clipboard copy is not supported on this platform".to_string(),
    ))
}

fn run_clipboard_command(program: &str, args: &[&str], input: &str) -> Result<(), AppError> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            AppError::Command(format!(
                "failed to spawn clipboard command {program}: {err}"
            ))
        })?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(input.as_bytes()).map_err(|err| {
            AppError::Command(format!(
                "failed to write clipboard input for {program}: {err}"
            ))
        })?;
    }
    let output = child.wait_with_output().map_err(|err| {
        AppError::Command(format!("failed to wait clipboard command {program}: {err}"))
    })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(AppError::Command(format!(
        "clipboard command {program} exited with status {:?}: {}",
        output.status.code(),
        if stderr.is_empty() {
            "unknown error"
        } else {
            stderr.as_str()
        }
    )))
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>, AppError> {
    enable_raw_mode()
        .map_err(|err| AppError::Command(format!("failed to enable raw mode: {err}")))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|err| AppError::Command(format!("failed to enter alternate screen: {err}")))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)
        .map_err(|err| AppError::Command(format!("failed to init terminal: {err}")))?;
    terminal
        .clear()
        .map_err(|err| AppError::Command(format!("failed to clear terminal: {err}")))?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<(), AppError> {
    disable_raw_mode()
        .map_err(|err| AppError::Command(format!("failed to disable raw mode: {err}")))?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )
    .map_err(|err| AppError::Command(format!("failed to leave alternate screen: {err}")))?;
    terminal
        .show_cursor()
        .map_err(|err| AppError::Command(format!("failed to show cursor: {err}")))?;
    Ok(())
}

fn edit_text_with_external_editor(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    initial: &str,
) -> Result<Option<String>, AppError> {
    restore_terminal(terminal)?;
    let edited = Editor::new()
        .edit(initial)
        .map_err(|err| AppError::Command(format!("failed to open external editor: {err}")));
    let mut reinitialized = init_terminal()?;
    std::mem::swap(terminal, &mut reinitialized);
    edited
}

fn load_theme_preferences() -> Result<(usize, PathBuf), AppError> {
    let path = ui_preferences_file_path()?;
    let content = fs::read_to_string(&path).ok();
    let preferences = content
        .as_deref()
        .and_then(|raw| serde_json::from_str::<UiPreferences>(raw).ok())
        .unwrap_or_default();
    let theme_idx = palette_index_by_name(preferences.theme.as_str()).unwrap_or(0);
    Ok((theme_idx, path))
}

fn save_theme_preferences(theme_idx: usize, path: &Path) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create ui preferences directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let payload = UiPreferences {
        theme: palette_by_index(theme_idx).name.to_string(),
    };
    let raw = serde_json::to_string_pretty(&payload)
        .map_err(|err| AppError::Runtime(format!("failed to serialize ui preferences: {err}")))?;
    fs::write(path, raw).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write ui preferences file {}: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

fn ui_preferences_file_path() -> Result<PathBuf, AppError> {
    let cwd = std::env::current_dir()
        .map_err(|err| AppError::Runtime(format!("failed to resolve current directory: {err}")))?;
    Ok(cwd.join(".machineclaw").join(UI_PREFS_FILE_NAME))
}

fn palettes() -> [ThemePalette; 3] {
    [
        ThemePalette {
            name: "graphite",
            app_bg: Color::Rgb(14, 16, 19),
            sidebar_bg: Color::Rgb(24, 27, 33),
            panel_bg: Color::Rgb(19, 22, 29),
            border: Color::Rgb(52, 57, 66),
            border_focus: Color::Rgb(107, 170, 255),
            text: Color::Rgb(236, 239, 244),
            muted: Color::Rgb(150, 159, 173),
            accent: Color::Rgb(116, 180, 255),
            status: Color::Rgb(162, 172, 187),
        },
        ThemePalette {
            name: "ocean",
            app_bg: Color::Rgb(9, 20, 30),
            sidebar_bg: Color::Rgb(14, 30, 45),
            panel_bg: Color::Rgb(10, 24, 37),
            border: Color::Rgb(45, 88, 112),
            border_focus: Color::Rgb(77, 195, 255),
            text: Color::Rgb(224, 241, 255),
            muted: Color::Rgb(130, 170, 198),
            accent: Color::Rgb(90, 210, 255),
            status: Color::Rgb(140, 182, 210),
        },
        ThemePalette {
            name: "paper",
            app_bg: Color::Rgb(246, 248, 252),
            sidebar_bg: Color::Rgb(238, 242, 248),
            panel_bg: Color::Rgb(252, 253, 255),
            border: Color::Rgb(182, 192, 209),
            border_focus: Color::Rgb(56, 108, 214),
            text: Color::Rgb(31, 42, 56),
            muted: Color::Rgb(95, 112, 133),
            accent: Color::Rgb(54, 102, 196),
            status: Color::Rgb(91, 108, 129),
        },
    ]
}

fn palette_by_index(index: usize) -> ThemePalette {
    let list = palettes();
    list[index % list.len()]
}

fn palette_index_by_name(name: &str) -> Option<usize> {
    palettes().iter().position(|item| item.name == name)
}

fn recent_messages_to_ui_messages(items: &[SessionMessage]) -> Vec<UiMessage> {
    items
        .iter()
        .map(|item| UiMessage {
            role: match item.role.as_str() {
                "assistant" => UiRole::Assistant,
                "thinking" => UiRole::Thinking,
                "tool" => UiRole::Tool,
                "system" => UiRole::System,
                _ => UiRole::User,
            },
            text: item.content.clone(),
            tool_meta: item.tool_meta.clone(),
        })
        .collect()
}

fn conversation_render_source_text(item: &UiMessage) -> Cow<'_, str> {
    if item.role != UiRole::Tool {
        return Cow::Borrowed(item.text.as_str());
    }
    if let Some(meta) = item.tool_meta.as_ref() {
        let raw = item.text.as_str();
        if parse_persisted_tool_message(raw).is_some()
            || format_legacy_tool_message_fallback(raw).is_some()
        {
            return Cow::Owned(format_tool_status_summary_from_meta(meta));
        }
    }
    format_persisted_tool_message_for_display(item.text.as_str())
}

fn render_conversation_item_text(item: &UiMessage, streaming_inflight: bool) -> String {
    let render_source = conversation_render_source_text(item);
    let source = render_source.trim();
    if item.role == UiRole::Tool {
        // Keep structured line breaks for persisted tool traces; markdown may collapse single '\n'.
        return normalize_conversation_markdown_for_bubble(source);
    }
    render_conversation_markdown(source, streaming_inflight)
}

#[derive(Debug, Clone, Copy)]
struct PersistedToolMessageParts<'a> {
    tool_call_id: &'a str,
    function: &'a str,
    args: &'a str,
    result: &'a str,
}

fn parse_persisted_tool_message(raw: &str) -> Option<PersistedToolMessageParts<'_>> {
    let body = raw.trim();
    let rest = body.strip_prefix("tool_call_id=")?;
    let (tool_call_id, rest) = rest.split_once(" function=")?;
    let (function, rest) = rest.split_once(" args=")?;
    let (args, result) = rest.rsplit_once(" result=")?;
    Some(PersistedToolMessageParts {
        tool_call_id: tool_call_id.trim(),
        function: function.trim(),
        args: args.trim(),
        result: result.trim(),
    })
}

fn tool_payload_to_display_text(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return trimmed.to_string();
    };
    match value {
        serde_json::Value::String(text) => text,
        other => serde_json::to_string_pretty(&other).unwrap_or_else(|_| trimmed.to_string()),
    }
}

fn format_persisted_tool_message_for_display(raw: &str) -> Cow<'_, str> {
    let Some(parts) = parse_persisted_tool_message(raw) else {
        if let Some(fallback) = format_legacy_tool_message_fallback(raw) {
            return Cow::Owned(fallback);
        }
        return Cow::Borrowed(raw);
    };
    Cow::Owned(format!(
        "tool_call_id={}\nfunction={}\nargs={}\nresult={}",
        parts.tool_call_id,
        parts.function,
        tool_payload_to_display_text(parts.args),
        tool_payload_to_display_text(parts.result)
    ))
}

fn format_legacy_tool_message_fallback(raw: &str) -> Option<String> {
    let body = raw.trim();
    if !(body.contains("tool_call_id=") && body.contains(" function=")) {
        return None;
    }
    let mut normalized = body.replace(" function=", "\nfunction=");
    normalized = normalized.replace(" args=", "\nargs=");
    normalized = normalized.replace(" result=", "\nresult=");
    normalized = normalized.replace(" cache_hit=", "\ncache_hit=");
    Some(normalized)
}

fn tool_result_detail_from_message(message: &UiMessage) -> Option<ToolResultDetail> {
    if message.role != UiRole::Tool {
        return None;
    }
    if let Some(meta) = message.tool_meta.as_ref() {
        return Some(ToolResultDetail {
            tool_call_id: meta.tool_call_id.clone(),
            function_name: meta.function_name.clone(),
            command: meta.command.clone(),
            arguments: tool_payload_to_display_text(meta.arguments.as_str()),
            result_payload: tool_result_output_for_modal(meta.result_payload.as_str()),
            executed_at_epoch_ms: meta.executed_at_epoch_ms,
            account: meta.account.clone(),
            environment: meta.environment.clone(),
            os_name: meta.os_name.clone(),
            cwd: meta.cwd.clone(),
            mode: meta.mode.clone(),
            label: meta.label.clone(),
            exit_code: meta.exit_code,
            duration_ms: meta.duration_ms,
            timed_out: meta.timed_out,
            interrupted: meta.interrupted,
            blocked: meta.blocked,
        });
    }
    let parts = parse_persisted_tool_message(message.text.as_str())?;
    let result_payload = tool_result_output_for_modal(parts.result);
    let command = {
        let extracted = extract_shell_command_from_tool_args(parts.args);
        if extracted.trim().is_empty() {
            parts.function.to_string()
        } else {
            extracted
        }
    };
    let arguments = tool_payload_to_display_text(parts.args);
    let result_value = serde_json::from_str::<serde_json::Value>(parts.result).ok();
    Some(ToolResultDetail {
        tool_call_id: parts.tool_call_id.to_string(),
        function_name: parts.function.to_string(),
        command,
        arguments,
        result_payload,
        executed_at_epoch_ms: 0,
        account: String::new(),
        environment: String::new(),
        os_name: String::new(),
        cwd: String::new(),
        mode: result_value
            .as_ref()
            .and_then(|item| item.get("mode"))
            .and_then(|item| item.as_str())
            .unwrap_or_default()
            .to_string(),
        label: result_value
            .as_ref()
            .and_then(|item| item.get("label"))
            .and_then(|item| item.as_str())
            .unwrap_or_default()
            .to_string(),
        exit_code: result_value
            .as_ref()
            .and_then(|item| item.get("exit_code"))
            .and_then(|item| item.as_i64())
            .and_then(|value| i32::try_from(value).ok()),
        duration_ms: result_value
            .as_ref()
            .and_then(|item| item.get("duration_ms"))
            .and_then(|item| item.as_u64())
            .map(|value| value as u128)
            .unwrap_or_default(),
        timed_out: result_value
            .as_ref()
            .and_then(|item| item.get("timed_out"))
            .and_then(|item| item.as_bool())
            .unwrap_or(false),
        interrupted: result_value
            .as_ref()
            .and_then(|item| item.get("interrupted"))
            .and_then(|item| item.as_bool())
            .unwrap_or(false),
        blocked: result_value
            .as_ref()
            .and_then(|item| item.get("blocked"))
            .and_then(|item| item.as_bool())
            .unwrap_or(false),
    })
}

fn extract_shell_command_from_tool_args(raw_args: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw_args) else {
        return String::new();
    };
    value
        .get("command")
        .and_then(|item| item.as_str())
        .unwrap_or_default()
        .to_string()
}

fn tool_result_output_for_modal(raw_payload: &str) -> String {
    let trimmed = raw_payload.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return trim_ui_text(trimmed, TOOL_RESULT_MODAL_MAX_RESULT_CHARS);
    };
    let stdout = value
        .get("stdout")
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    let stderr = value
        .get("stderr")
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    if !stdout.is_empty() || !stderr.is_empty() {
        if !stdout.is_empty() && stderr.is_empty() {
            return trim_ui_text(stdout, TOOL_RESULT_MODAL_MAX_RESULT_CHARS);
        }
        if stdout.is_empty() && !stderr.is_empty() {
            return trim_ui_text(stderr, TOOL_RESULT_MODAL_MAX_RESULT_CHARS);
        }
        return trim_ui_text(
            format!("stdout:\n{}\n\nstderr:\n{}", stdout, stderr).as_str(),
            TOOL_RESULT_MODAL_MAX_RESULT_CHARS,
        );
    }
    trim_ui_text(
        serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| trimmed.to_string())
            .as_str(),
        TOOL_RESULT_MODAL_MAX_RESULT_CHARS,
    )
}

fn format_tool_status_summary_from_meta(meta: &ToolExecutionMeta) -> String {
    let ok = tool_result_ok_from_payload(meta.result_payload.as_str()).unwrap_or_else(|| {
        meta.exit_code == Some(0) && !meta.timed_out && !meta.interrupted && !meta.blocked
    });
    let exit_text = meta
        .exit_code
        .map(|code| format!("Some({code})"))
        .unwrap_or_else(|| "None".to_string());
    format!(
        "{}: ok={} exit={} timeout={} blocked={}",
        if ok {
            ui_text_tool_finished()
        } else {
            ui_text_tool_error()
        },
        ok,
        exit_text,
        meta.timed_out,
        meta.blocked
    )
}

fn tool_result_ok_from_payload(raw_payload: &str) -> Option<bool> {
    let trimmed = raw_payload.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(trimmed)
        .ok()
        .and_then(|item| item.get("ok").and_then(|v| v.as_bool()))
}

fn char_to_byte_idx(text: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn text_display_width(text: &str) -> usize {
    text.chars().map(char_display_width).sum::<usize>()
}

fn char_display_width(ch: char) -> usize {
    if ch == '\n' || ch == '\r' || is_zero_width_char(ch) {
        return 0;
    }
    if is_emoji_char(ch) {
        return 1;
    }
    UnicodeWidthChar::width(ch).unwrap_or(1)
}

fn is_zero_width_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x200C
            | 0x200D
            | 0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
            | 0xFE00..=0xFE0F
            | 0x1F3FB..=0x1F3FF
            | 0xE0100..=0xE01EF
            | 0xE0020..=0xE007F
    )
}

fn is_emoji_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F000..=0x1FAFF
            | 0x2600..=0x26FF
            | 0x2700..=0x27BF
            | 0x2B50
            | 0x2B55
    )
}

fn normalize_conversation_line(text: &str) -> String {
    text.replace('\r', "").replace('\t', "    ")
}

fn render_conversation_markdown(raw: &str, streaming_inflight: bool) -> String {
    if streaming_inflight && !might_need_markdown_render(raw) {
        return normalize_conversation_markdown_for_bubble(raw);
    }
    if streaming_inflight
        && (has_inline_markdown_link_syntax(raw) || contains_plain_url_candidate(raw))
    {
        return normalize_conversation_markdown_for_bubble(raw);
    }
    let rendered = render::render_markdown_for_terminal(raw, false);
    if streaming_inflight {
        if rendered.trim().is_empty() && !raw.trim().is_empty() {
            return normalize_conversation_markdown_for_bubble(raw);
        }
        if has_unfinished_inline_markdown_link(raw) {
            return normalize_conversation_markdown_for_bubble(raw);
        }
    }
    normalize_conversation_markdown_for_bubble(&rendered)
}

fn might_need_markdown_render(text: &str) -> bool {
    text.bytes().any(|b| {
        matches!(
            b,
            b'`' | b'*'
                | b'_'
                | b'['
                | b']'
                | b'('
                | b')'
                | b'#'
                | b'>'
                | b'!'
                | b'~'
                | b'|'
                | b'<'
                | b'-'
        )
    })
}

fn has_unfinished_inline_markdown_link(text: &str) -> bool {
    if !text.contains("](") {
        return false;
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() < 4 {
        return false;
    }
    let mut label_stack = Vec::<usize>::new();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' {
            i = (i + 2).min(chars.len());
            continue;
        }
        if ch == '`' {
            let mut tick_len = 1usize;
            while i + tick_len < chars.len() && chars[i + tick_len] == '`' {
                tick_len += 1;
            }
            i += tick_len;
            while i < chars.len() {
                if chars[i] == '\\' {
                    i = (i + 2).min(chars.len());
                    continue;
                }
                let mut closing_ticks = 0usize;
                while i + closing_ticks < chars.len() && chars[i + closing_ticks] == '`' {
                    closing_ticks += 1;
                }
                if closing_ticks >= tick_len {
                    i += tick_len;
                    break;
                }
                i += 1;
            }
            continue;
        }
        match ch {
            '[' => label_stack.push(i),
            ']' => {
                let has_label = label_stack.pop().is_some();
                let starts_link = i + 1 < chars.len() && chars[i + 1] == '(';
                if has_label && starts_link {
                    let mut depth = 1usize;
                    i += 2;
                    while i < chars.len() {
                        let current = chars[i];
                        if current == '\\' {
                            i = (i + 2).min(chars.len());
                            continue;
                        }
                        if current == '(' {
                            depth += 1;
                        } else if current == ')' {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                break;
                            }
                        }
                        i += 1;
                    }
                    if depth > 0 {
                        return true;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn has_inline_markdown_link_syntax(text: &str) -> bool {
    if !text.contains("](") {
        return false;
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() < 4 {
        return false;
    }
    let mut label_stack = Vec::<usize>::new();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' {
            i = (i + 2).min(chars.len());
            continue;
        }
        if ch == '`' {
            let mut tick_len = 1usize;
            while i + tick_len < chars.len() && chars[i + tick_len] == '`' {
                tick_len += 1;
            }
            i += tick_len;
            while i < chars.len() {
                if chars[i] == '\\' {
                    i = (i + 2).min(chars.len());
                    continue;
                }
                let mut closing_ticks = 0usize;
                while i + closing_ticks < chars.len() && chars[i + closing_ticks] == '`' {
                    closing_ticks += 1;
                }
                if closing_ticks >= tick_len {
                    i += tick_len;
                    break;
                }
                i += 1;
            }
            continue;
        }
        match ch {
            '[' => label_stack.push(i),
            ']' => {
                let has_label = label_stack.pop().is_some();
                let starts_link = i + 1 < chars.len() && chars[i + 1] == '(';
                if has_label && starts_link {
                    return true;
                }
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn contains_plain_url_candidate(text: &str) -> bool {
    text.contains("https://") || text.contains("http://")
}

fn normalize_conversation_markdown_for_bubble(rendered: &str) -> String {
    if rendered.is_empty() {
        return String::new();
    }
    let mut lines = Vec::<String>::new();
    let mut in_code_fence = false;
    for raw_line in rendered.lines() {
        let trimmed = raw_line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            lines.push(raw_line.to_string());
            continue;
        }
        lines.push(strip_markdown_heading_prefix(raw_line));
    }
    lines.join("\n")
}

fn strip_markdown_heading_prefix(line: &str) -> String {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return line.to_string();
    }
    let mut marker_len = 0usize;
    for ch in trimmed.chars() {
        if ch == '#' {
            marker_len += 1;
        } else {
            break;
        }
    }
    if marker_len == 0 || marker_len > 6 {
        return line.to_string();
    }
    let Some(after_markers) = trimmed.get(marker_len..) else {
        return line.to_string();
    };
    if !after_markers.starts_with(' ')
        && (marker_len < 2 || after_markers.starts_with('#') || after_markers.is_empty())
    {
        return line.to_string();
    }
    let leading_spaces = line.len().saturating_sub(trimmed.len());
    format!(
        "{}{}",
        " ".repeat(leading_spaces),
        after_markers.trim_start()
    )
}

fn conversation_bubble_max_inner_width(wrap_width: u16) -> usize {
    if wrap_width <= 1 {
        return 12;
    }
    let width = wrap_width as usize;
    let hard_cap = width.saturating_sub(4).max(1);
    let preferred = width.saturating_sub(10).max(1);
    preferred.max(12).min(hard_cap)
}

fn wrap_text_by_display_width(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::<String>::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in text.chars() {
        let ch_width = char_display_width(ch);
        if ch_width == 0 {
            current.push(ch);
            continue;
        }
        if current_width + ch_width > max_width {
            if !current.is_empty() {
                out.push(current);
                current = String::new();
                current_width = 0;
            }
            if ch_width > max_width {
                out.push(ch.to_string());
                continue;
            }
        }
        current.push(ch);
        current_width += ch_width;
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn display_width_between(text: &str, from_char: usize, to_char: usize) -> usize {
    if to_char <= from_char {
        return 0;
    }
    let from_byte = char_to_byte_idx(text, from_char);
    let to_byte = char_to_byte_idx(text, to_char);
    text_display_width(&text[from_byte..to_byte])
}

fn trim_ui_text(text: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_len {
        return text.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    chars.into_iter().take(max_len - 3).collect::<String>() + "..."
}

fn trim_tool_text(text: &str, max_len: usize) -> String {
    trim_ui_text(text, max_len)
}

fn lang_is_zh() -> bool {
    matches!(
        i18n::current_language(),
        i18n::Language::ZhCn | i18n::Language::ZhTw
    )
}

fn zh_or_en(zh: &'static str, en: &'static str) -> &'static str {
    if lang_is_zh() { zh } else { en }
}

fn ui_text_app_name() -> &'static str {
    "MachineClaw"
}
fn ui_text_nav_new_thread() -> &'static str {
    zh_or_en("新线程", "New Thread")
}
fn ui_text_nav_skills() -> &'static str {
    zh_or_en("技能", "Skills")
}
fn ui_text_nav_mcp() -> &'static str {
    "MCP"
}
fn ui_text_nav_inspect() -> &'static str {
    zh_or_en("巡检", "Inspect")
}
fn ui_text_nav_config() -> &'static str {
    zh_or_en("配置", "Config")
}
fn ui_text_skills_panel_title() -> &'static str {
    zh_or_en("技能面板", "Skills Panel")
}
fn ui_text_mcp_panel_title() -> &'static str {
    zh_or_en("MCP 服务管理", "MCP Servers")
}
fn ui_text_mcp_panel_ready() -> &'static str {
    zh_or_en("已切换到 MCP 管理页面", "Switched to MCP panel")
}
fn ui_text_mcp_servers_count() -> &'static str {
    zh_or_en("服务数量", "Servers")
}
fn ui_text_mcp_config_file() -> &'static str {
    zh_or_en("配置文件", "Config File")
}
fn ui_text_mcp_servers_list() -> &'static str {
    zh_or_en("已安装服务", "Installed Servers")
}
fn ui_text_mcp_server_detail() -> &'static str {
    zh_or_en("服务详情", "Server Detail")
}
fn ui_text_mcp_tools_panel() -> &'static str {
    zh_or_en("工具能力", "Tools")
}
fn ui_text_mcp_tools_label() -> &'static str {
    zh_or_en("工具列表", "Tool List")
}
fn ui_text_mcp_tools_empty() -> &'static str {
    zh_or_en("未发现工具", "No tools discovered")
}
fn ui_text_mcp_no_tools() -> &'static str {
    zh_or_en("暂无可展示工具", "No tools available")
}
fn ui_text_mcp_input_hint() -> &'static str {
    zh_or_en(
        "Enter切换/编辑/保存, ←/→切换选项, Ctrl+S保存, A新增, D删除",
        "Enter select/edit/save, Left/Right choose option, Ctrl+S save, A add, D delete",
    )
}
fn ui_text_mcp_input_ready() -> &'static str {
    zh_or_en("已切换到 MCP 输入编辑区", "MCP input is ready")
}
fn ui_text_mcp_dirty_count() -> &'static str {
    zh_or_en("待保存", "Pending")
}
fn ui_text_mcp_save_button() -> &'static str {
    zh_or_en("确认保存", "Save")
}
fn ui_text_mcp_save_no_change() -> &'static str {
    zh_or_en("无变更", "No changes")
}
fn ui_text_mcp_saved() -> &'static str {
    zh_or_en(
        "MCP 配置已保存并刷新",
        "MCP configuration saved and refreshed",
    )
}
fn ui_text_mcp_editing() -> &'static str {
    zh_or_en("正在编辑 MCP 字段", "Editing MCP field")
}
fn ui_text_mcp_edit_applied() -> &'static str {
    zh_or_en(
        "MCP 字段已更新（待保存）",
        "MCP field updated (pending save)",
    )
}
fn ui_text_mcp_edit_cancelled() -> &'static str {
    zh_or_en("已取消 MCP 编辑", "MCP edit cancelled")
}
fn ui_text_mcp_server_added() -> &'static str {
    zh_or_en("已新增 MCP 服务", "Added MCP server")
}
fn ui_text_mcp_server_deleted() -> &'static str {
    zh_or_en("已删除 MCP 服务", "Deleted MCP server")
}
fn ui_text_mcp_focus_servers() -> &'static str {
    zh_or_en("焦点: MCP 服务列表", "Focus: MCP server list")
}
fn ui_text_mcp_focus_fields() -> &'static str {
    zh_or_en("焦点: MCP 字段详情", "Focus: MCP field details")
}
fn ui_text_mcp_error() -> &'static str {
    zh_or_en("错误", "Error")
}
fn ui_text_mcp_empty() -> &'static str {
    zh_or_en(
        "暂无 MCP 服务，按 A 新增",
        "No MCP server yet, press A to add",
    )
}
fn ui_text_mcp_empty_fields() -> &'static str {
    zh_or_en("请先选择 MCP 服务", "Select an MCP server first")
}
fn ui_text_config_panel_title() -> &'static str {
    zh_or_en("配置管理", "Configuration")
}
fn ui_text_config_mode_ready() -> &'static str {
    zh_or_en("已切换到配置管理页面", "Switched to configuration panel")
}
fn ui_text_config_category() -> &'static str {
    zh_or_en("分类", "Categories")
}
fn ui_text_config_fields() -> &'static str {
    zh_or_en("字段", "Fields")
}
fn ui_text_config_col_key() -> &'static str {
    zh_or_en("字段", "Key")
}
fn ui_text_config_col_value() -> &'static str {
    zh_or_en("值", "Value")
}
fn ui_text_config_col_type() -> &'static str {
    zh_or_en("类型", "Type")
}
fn ui_text_config_empty_fields() -> &'static str {
    zh_or_en("当前分类暂无字段", "No fields in this category")
}
fn ui_text_config_input_hint() -> &'static str {
    zh_or_en(
        "Enter切换/编辑/保存, ←/→切换选项, Ctrl+S保存, Esc返回",
        "Enter select/edit/save, Left/Right choose option, Ctrl+S save, Esc back",
    )
}
fn ui_text_config_required() -> &'static str {
    zh_or_en("(必填)", "(required)")
}
fn ui_text_config_dirty_count() -> &'static str {
    zh_or_en("待保存", "Pending")
}
fn ui_text_config_save_button() -> &'static str {
    zh_or_en("确认保存", "Save")
}
fn ui_text_config_save_no_change() -> &'static str {
    zh_or_en("无变更", "No changes")
}
fn ui_text_config_nothing_to_save() -> &'static str {
    zh_or_en("没有可保存的配置变更", "No config changes to save")
}
fn ui_text_config_saved() -> &'static str {
    zh_or_en("配置已保存并写回配置文件", "Configuration saved to file")
}
fn ui_text_config_save_failed() -> &'static str {
    zh_or_en("配置保存失败", "Configuration save failed")
}
fn ui_text_config_editing() -> &'static str {
    zh_or_en("正在编辑配置字段", "Editing configuration field")
}
fn ui_text_config_edit_applied() -> &'static str {
    zh_or_en("字段值已更新（待保存）", "Field updated (pending save)")
}
fn ui_text_config_edit_cancelled() -> &'static str {
    zh_or_en("已取消编辑", "Edit cancelled")
}
fn ui_text_config_category_app() -> &'static str {
    zh_or_en("应用", "App")
}
fn ui_text_config_category_ai() -> &'static str {
    zh_or_en("AI基础", "AI")
}
fn ui_text_config_category_ai_retry() -> &'static str {
    zh_or_en("AI重试", "AI Retry")
}
fn ui_text_config_category_ai_chat() -> &'static str {
    zh_or_en("AI聊天", "AI Chat")
}
fn ui_text_config_category_ai_compression() -> &'static str {
    zh_or_en("聊天压缩", "Chat Compression")
}
fn ui_text_config_category_ai_tools_bash() -> &'static str {
    zh_or_en("AI工具(Bash)", "AI Tools (Bash)")
}
fn ui_text_config_category_skills() -> &'static str {
    zh_or_en("技能", "Skills")
}
fn ui_text_config_category_mcp() -> &'static str {
    zh_or_en("MCP", "MCP")
}
fn ui_text_config_category_console() -> &'static str {
    zh_or_en("控制台", "Console")
}
fn ui_text_config_category_log() -> &'static str {
    zh_or_en("日志", "Log")
}
fn ui_text_config_category_session() -> &'static str {
    zh_or_en("会话", "Session")
}
fn ui_text_skills_panel_ready() -> &'static str {
    zh_or_en("已切换到技能面板", "Switched to skills panel")
}
fn ui_text_skills_count() -> &'static str {
    zh_or_en("技能数量", "Skill Count")
}
fn ui_text_skills_dir() -> &'static str {
    zh_or_en("扫描目录", "Skills Directory")
}
fn ui_text_skills_list() -> &'static str {
    zh_or_en("技能列表", "Skill List")
}
fn ui_text_skills_name() -> &'static str {
    zh_or_en("名称", "Name")
}
fn ui_text_skills_purpose() -> &'static str {
    zh_or_en("用途", "Purpose")
}
fn ui_text_skills_path() -> &'static str {
    zh_or_en("路径", "Path")
}
fn ui_text_skills_empty() -> &'static str {
    zh_or_en("未检测到可用技能", "No skills detected")
}
fn ui_text_skills_doc_title() -> &'static str {
    zh_or_en("技能文档", "Skill Document")
}
fn ui_text_skills_doc_hint() -> &'static str {
    zh_or_en(
        "Esc关闭 · E/Enter编辑并保存到文件 · ↑↓/PgUp/PgDn滚动",
        "Esc close · E/Enter edit and save file · ↑↓/PgUp/PgDn scroll",
    )
}
fn ui_text_skills_doc_opened() -> &'static str {
    zh_or_en("已打开技能文档", "Skill document opened")
}
fn ui_text_skills_doc_saved() -> &'static str {
    zh_or_en("技能文档已保存", "Skill document saved")
}
fn ui_text_skills_doc_save_failed() -> &'static str {
    zh_or_en("技能文档保存失败", "Skill document save failed")
}
fn ui_text_skills_doc_closed() -> &'static str {
    zh_or_en("已关闭技能文档", "Skill document closed")
}
fn ui_text_skills_doc_open_failed() -> &'static str {
    zh_or_en("技能文档打开失败", "Skill document open failed")
}
fn ui_text_skills_doc_edit_cancelled() -> &'static str {
    zh_or_en("已取消文档编辑", "Skill document edit cancelled")
}
fn ui_text_skills_doc_unchanged() -> &'static str {
    zh_or_en("文档无变化，未保存", "Document unchanged, not saved")
}
fn ui_text_panel_threads() -> &'static str {
    zh_or_en("线程", "Threads")
}
fn ui_text_thread_action_menu_title() -> &'static str {
    zh_or_en("线程操作", "Thread Actions")
}
fn ui_text_thread_action_delete() -> &'static str {
    zh_or_en("删除", "Delete")
}
fn ui_text_thread_action_rename() -> &'static str {
    zh_or_en("重命名", "Rename")
}
fn ui_text_thread_action_metadata() -> &'static str {
    zh_or_en("元数据", "Metadata")
}
fn ui_text_thread_metadata_modal_title() -> &'static str {
    zh_or_en("线程元数据", "Thread Metadata")
}
fn ui_text_thread_metadata_modal_subtitle() -> &'static str {
    zh_or_en("只读视图 · 结构化统计", "Read-only view · structured stats")
}
fn ui_text_thread_metadata_field() -> &'static str {
    zh_or_en("字段", "Field")
}
fn ui_text_thread_metadata_value() -> &'static str {
    zh_or_en("值", "Value")
}
fn ui_text_thread_metadata_modal_hint() -> &'static str {
    zh_or_en(
        "Esc/Enter关闭 · ↑↓/PgUp/PgDn滚动 · 鼠标滚轮支持",
        "Esc/Enter close · ↑↓/PgUp/PgDn scroll · mouse wheel supported",
    )
}
fn ui_text_thread_action_menu_hint() -> &'static str {
    zh_or_en(
        "双击线程已弹出菜单，↑↓切换，Enter确认，Esc关闭",
        "Thread menu opened, use ↑↓ to select, Enter to apply, Esc to close",
    )
}
fn ui_text_thread_rename_modal_title() -> &'static str {
    zh_or_en("重命名线程", "Rename Thread")
}
fn ui_text_thread_rename_modal_hint() -> &'static str {
    zh_or_en(
        "输入新名称，Enter保存，Esc取消",
        "Type new name, Enter to save, Esc to cancel",
    )
}
fn ui_text_thread_delete_confirm_prompt() -> &'static str {
    zh_or_en("请确认是否删除该线程", "Confirm thread deletion")
}
fn ui_text_thread_delete_modal_title() -> &'static str {
    zh_or_en("删除线程确认", "Delete Thread")
}
fn ui_text_thread_delete_modal_question() -> &'static str {
    zh_or_en(
        "删除后将同步移除会话文件，是否继续？",
        "Delete this thread and remove its session file?",
    )
}
fn ui_text_thread_delete_preview_title() -> &'static str {
    zh_or_en("线程预览", "Thread Preview")
}
fn ui_text_thread_delete_modal_hint() -> &'static str {
    zh_or_en(
        "方向键/Tab切换，Enter确认，Esc取消",
        "Use arrows/Tab to switch, Enter to confirm, Esc to cancel",
    )
}
fn ui_text_thread_delete_yes() -> &'static str {
    zh_or_en("确认删除", "Delete")
}
fn ui_text_thread_delete_no() -> &'static str {
    zh_or_en("取消", "Cancel")
}
fn ui_text_panel_conversation() -> &'static str {
    zh_or_en("会话", "Conversation")
}
fn ui_text_panel_input() -> &'static str {
    zh_or_en("输入", "Input")
}
fn ui_text_session_card_title() -> &'static str {
    zh_or_en("会话信息", "Session Info")
}
fn ui_text_session_card_updated() -> &'static str {
    zh_or_en("最近更新", "Updated")
}
fn ui_text_session_card_tokens() -> &'static str {
    zh_or_en("Token消耗", "Token Usage")
}
fn ui_text_model_label() -> &'static str {
    zh_or_en("模型", "Model")
}
fn ui_text_header_status() -> &'static str {
    zh_or_en("状态", "Status")
}
fn ui_text_input_hint() -> &'static str {
    zh_or_en(
        "Enter发送, Tab切换面板, Ctrl+C/Ctrl+Z退出",
        "Enter send, Tab switch panel, Ctrl+C/Ctrl+Z exit",
    )
}
fn ui_text_theme_hint() -> &'static str {
    zh_or_en("F2切换主题", "F2 switch theme")
}
fn ui_text_ready() -> &'static str {
    zh_or_en("就绪", "Ready")
}
fn ui_text_tui_ready() -> &'static str {
    zh_or_en("TUI已就绪", "TUI ready")
}
fn ui_text_startup_splash_hint() -> &'static str {
    zh_or_en("初始化中，请稍候…", "Initializing, please wait...")
}
fn ui_text_thinking() -> &'static str {
    zh_or_en("AI思考中...", "AI thinking...")
}
fn ui_text_stream_fallback_notice() -> &'static str {
    zh_or_en(
        "提示：本轮未收到流式分片，已按非流式回显（可能是服务端/模型兼容性回退）。",
        "Notice: no stream chunks were received in this round; output was shown in non-streaming mode (possibly due to server/model compatibility fallback).",
    )
}
fn ui_text_status_ai_thinking() -> &'static str {
    zh_or_en("[AI] 深度思考中", "[AI] reasoning")
}
fn ui_text_status_ai_tooling() -> &'static str {
    zh_or_en("[AI] 工具执行中(Bash/MCP)", "[AI] running tools (Bash/MCP)")
}
fn ui_text_status_ai_cancelling() -> &'static str {
    zh_or_en("[AI] 正在取消请求", "[AI] cancelling request")
}
fn ui_text_ai_failed() -> &'static str {
    zh_or_en("AI请求失败", "AI request failed")
}
fn ui_text_ai_channel_closed() -> &'static str {
    zh_or_en("AI通信通道已关闭", "AI response channel closed")
}
fn ui_text_no_sessions() -> &'static str {
    zh_or_en("(无会话)", "(no sessions)")
}
fn ui_text_no_messages() -> &'static str {
    zh_or_en("暂无消息", "No messages yet")
}
fn ui_text_footer_help() -> &'static str {
    "/help"
}
fn ui_text_help_commands() -> &'static str {
    zh_or_en(
        "命令: /help /stats /meta /skills /mcps /new /list /change <id|name> /name <new-name> /history [n] /clear /exit",
        "Commands: /help /stats /meta /skills /mcps /new /list /change <id|name> /name <new-name> /history [n] /clear /exit",
    )
}
fn ui_text_focus_hint(focus: FocusPanel) -> &'static str {
    match focus {
        FocusPanel::Nav => zh_or_en("当前焦点: 功能导航", "Focus: Navigation"),
        FocusPanel::Threads => zh_or_en("当前焦点: 线程面板", "Focus: Threads"),
        FocusPanel::Conversation => zh_or_en("当前焦点: 会话面板", "Focus: Conversation"),
        FocusPanel::Input => zh_or_en("当前焦点: 输入面板", "Focus: Input"),
    }
}
fn ui_text_choice_prompt() -> &'static str {
    zh_or_en(
        "请选择一个选项（方向键+Enter）",
        "Pick one option (arrows + Enter)",
    )
}
fn ui_text_choice_ready() -> &'static str {
    zh_or_en(
        "检测到可选项，已切换到选项输入",
        "Choice options detected in input panel",
    )
}
fn ui_text_choice_cancelled() -> &'static str {
    zh_or_en("已退出选项选择", "Choice selection cancelled")
}
fn ui_text_na() -> &'static str {
    zh_or_en("N/A", "N/A")
}
fn ui_text_inspect_target_menu() -> &'static str {
    zh_or_en("选择巡检目标", "Choose Inspect Target")
}
fn ui_text_inspect_target_menu_hint() -> &'static str {
    zh_or_en("请选择巡检目标", "Select inspect target")
}
fn ui_text_inspect_target_changed() -> &'static str {
    zh_or_en("巡检目标已切换", "Inspect target switched")
}
fn ui_text_inspect_panel_title() -> &'static str {
    zh_or_en("巡检面板", "Inspect Panel")
}
fn ui_text_inspect_usage() -> &'static str {
    zh_or_en("CPU使用率", "CPU Usage")
}
fn ui_text_inspect_trend() -> &'static str {
    zh_or_en("负载趋势", "Usage Trend")
}
fn ui_text_inspect_details() -> &'static str {
    zh_or_en("详细参数", "Details")
}
fn ui_text_inspect_raw() -> &'static str {
    zh_or_en("原始数据", "Raw Data")
}
fn ui_text_inspect_no_data() -> &'static str {
    zh_or_en("暂无数据", "No data")
}
fn ui_text_inspect_timeout_hint() -> &'static str {
    zh_or_en("命令超时", "Command timed out")
}
fn ui_text_inspect_model() -> &'static str {
    zh_or_en("CPU型号", "CPU Model")
}
fn ui_text_inspect_cores() -> &'static str {
    zh_or_en("核数(物理/逻辑)", "Cores (P/L)")
}
fn ui_text_inspect_frequency() -> &'static str {
    zh_or_en("频率", "Frequency")
}
fn ui_text_inspect_temperature() -> &'static str {
    zh_or_en("温度", "Temperature")
}
fn ui_text_inspect_health() -> &'static str {
    zh_or_en("健康状态", "Health")
}
fn ui_text_inspect_breakdown() -> &'static str {
    zh_or_en("占比拆分", "Breakdown")
}
fn ui_text_inspect_updated() -> &'static str {
    zh_or_en("更新时间", "Updated At")
}
fn ui_text_inspect_target() -> &'static str {
    zh_or_en("巡检目标", "Target")
}
fn ui_text_inspect_line_count() -> &'static str {
    zh_or_en("有效行数", "Line Count")
}
fn ui_text_inspect_byte_count() -> &'static str {
    zh_or_en("字符数", "Chars")
}
fn ui_text_inspect_signal_ok() -> &'static str {
    zh_or_en("状态正常", "Healthy")
}
fn ui_text_inspect_signal_warn() -> &'static str {
    zh_or_en("存在异常信号", "Warning Signals")
}
fn ui_text_inspect_highlights() -> &'static str {
    zh_or_en("关键字段", "Highlights")
}
fn ui_text_inspect_highlights_empty() -> &'static str {
    zh_or_en("无可提取字段", "No parsed fields")
}
fn ui_text_inspect_raw_line() -> &'static str {
    zh_or_en("原始行", "Raw Line")
}
fn ui_text_inspect_raw_usage() -> &'static str {
    zh_or_en("[CPU使用原始输出]", "[CPU usage raw]")
}
fn ui_text_inspect_raw_info() -> &'static str {
    zh_or_en("[CPU信息原始输出]", "[CPU info raw]")
}
fn ui_text_inspect_raw_temperature() -> &'static str {
    zh_or_en("[温度原始输出]", "[Temperature raw]")
}
fn ui_text_inspect_health_unknown() -> &'static str {
    zh_or_en("未知", "Unknown")
}
fn ui_text_inspect_health_ok() -> &'static str {
    zh_or_en("健康", "Healthy")
}
fn ui_text_inspect_health_warn() -> &'static str {
    zh_or_en("偏高", "Elevated")
}
fn ui_text_inspect_health_high() -> &'static str {
    zh_or_en("高负载", "High Load")
}
fn ui_text_confirm_title() -> &'static str {
    zh_or_en("写命令确认", "Write Command Confirmation")
}
fn ui_text_confirm_message() -> &'static str {
    zh_or_en(
        "是否继续执行以下写命令？",
        "Proceed with this write command?",
    )
}
fn ui_text_confirm_command_preview_title() -> &'static str {
    zh_or_en("命令预览", "Command Preview")
}
fn ui_text_confirm_hint() -> &'static str {
    zh_or_en(
        "方向键/Tab切换，Enter确认，Esc取消",
        "Use arrows/Tab to switch, Enter to confirm, Esc to cancel",
    )
}
fn ui_text_edit_title() -> &'static str {
    zh_or_en("编辑写命令", "Edit Write Command")
}
fn ui_text_edit_hint() -> &'static str {
    zh_or_en(
        "可编辑命令；Enter确认，Esc取消",
        "Edit command; Enter to apply, Esc to cancel",
    )
}
fn ui_text_edit_footer() -> &'static str {
    zh_or_en(
        "提示: 仅对当前命令生效",
        "Tip: applies to current command only",
    )
}
fn ui_text_decision_reject() -> &'static str {
    zh_or_en("拒绝", "Reject")
}
fn ui_text_decision_approve() -> &'static str {
    zh_or_en("允许一次", "Allow Once")
}
fn ui_text_decision_approve_session() -> &'static str {
    zh_or_en("本会话允许", "Allow Session")
}
fn ui_text_decision_edit() -> &'static str {
    zh_or_en("编辑命令", "Edit")
}
fn ui_text_tag_user() -> &'static str {
    zh_or_en("[你]", "[You]")
}
fn ui_text_tag_ai() -> &'static str {
    "[AI]"
}
fn ui_text_tag_thinking() -> &'static str {
    zh_or_en("[思考]", "[Thinking]")
}
fn ui_text_tag_sys() -> &'static str {
    zh_or_en("[系统]", "[Sys]")
}
fn ui_text_tag_tool() -> &'static str {
    zh_or_en("[工具]", "[Tool]")
}
fn ui_text_message_copy_button() -> &'static str {
    zh_or_en("[复制]", "[Copy]")
}
fn ui_text_message_tool_result_button() -> &'static str {
    zh_or_en("[执行结果]", "[Result]")
}
fn ui_text_message_delete_button() -> &'static str {
    zh_or_en("[删除]", "[Delete]")
}
fn ui_text_message_copy_success() -> &'static str {
    zh_or_en("消息内容已复制到剪贴板", "Message copied to clipboard")
}
fn ui_text_message_copy_failed() -> &'static str {
    zh_or_en("复制消息失败", "Failed to copy message")
}
fn ui_text_message_tool_result_modal_title() -> &'static str {
    zh_or_en("工具执行结果", "Tool Execution Result")
}
fn ui_text_message_tool_result_modal_subtitle() -> &'static str {
    zh_or_en("工具函数:", "Function:")
}
fn ui_text_message_tool_result_modal_hint() -> &'static str {
    zh_or_en(
        "方向键/PageUp/PageDown滚动，Esc关闭",
        "Use arrows/PageUp/PageDown to scroll, Esc to close",
    )
}
fn ui_text_message_tool_result_modal_opened() -> &'static str {
    zh_or_en("已打开执行结果弹窗", "Execution result modal opened")
}
fn ui_text_message_tool_result_modal_closed() -> &'static str {
    zh_or_en("已关闭执行结果弹窗", "Execution result modal closed")
}
fn ui_text_message_tool_result_unavailable() -> &'static str {
    zh_or_en("当前消息没有可展示的执行结果", "No execution result for this message")
}
fn ui_text_message_tool_result_meta_call_id() -> &'static str {
    zh_or_en("调用ID:", "Call ID:")
}
fn ui_text_message_tool_result_meta_time() -> &'static str {
    zh_or_en("执行时间:", "Executed At:")
}
fn ui_text_message_tool_result_meta_account() -> &'static str {
    zh_or_en("执行账号:", "Account:")
}
fn ui_text_message_tool_result_meta_env() -> &'static str {
    zh_or_en("环境:", "Environment:")
}
fn ui_text_message_tool_result_meta_os() -> &'static str {
    zh_or_en("操作系统:", "OS:")
}
fn ui_text_message_tool_result_meta_cwd() -> &'static str {
    zh_or_en("工作目录:", "CWD:")
}
fn ui_text_message_tool_result_meta_label() -> &'static str {
    zh_or_en("标签:", "Label:")
}
fn ui_text_message_tool_result_meta_mode() -> &'static str {
    zh_or_en("模式:", "Mode:")
}
fn ui_text_message_tool_result_meta_exit_code() -> &'static str {
    zh_or_en("退出码:", "Exit Code:")
}
fn ui_text_message_tool_result_meta_duration() -> &'static str {
    zh_or_en("耗时:", "Duration:")
}
fn ui_text_message_tool_result_meta_status() -> &'static str {
    zh_or_en("状态:", "Status:")
}
fn ui_text_message_tool_result_meta_command() -> &'static str {
    zh_or_en("命令", "Command")
}
fn ui_text_message_tool_result_meta_arguments() -> &'static str {
    zh_or_en("参数", "Arguments")
}
fn ui_text_message_tool_result_meta_output() -> &'static str {
    zh_or_en("输出", "Output")
}
fn ui_text_message_delete_confirm_prompt() -> &'static str {
    zh_or_en("请确认是否删除该消息", "Confirm message deletion")
}
fn ui_text_message_delete_modal_title() -> &'static str {
    zh_or_en("删除消息确认", "Delete Message")
}
fn ui_text_message_delete_modal_question() -> &'static str {
    zh_or_en(
        "删除后将同步写回会话文件，是否继续？",
        "Delete this message and persist it to the session file?",
    )
}
fn ui_text_message_delete_preview_title() -> &'static str {
    zh_or_en("消息预览", "Message Preview")
}
fn ui_text_message_delete_modal_hint() -> &'static str {
    zh_or_en(
        "方向键/Tab切换，Enter确认，Esc取消",
        "Use arrows/Tab to switch, Enter to confirm, Esc to cancel",
    )
}
fn ui_text_message_delete_yes() -> &'static str {
    zh_or_en("确认删除", "Delete")
}
fn ui_text_message_delete_no() -> &'static str {
    zh_or_en("取消", "Cancel")
}
fn ui_text_message_delete_success() -> &'static str {
    zh_or_en("消息已删除并写回会话文件", "Message deleted and persisted")
}
fn ui_text_message_delete_cancelled() -> &'static str {
    zh_or_en("已取消删除消息", "Message deletion cancelled")
}
fn ui_text_message_delete_not_found() -> &'static str {
    zh_or_en("未找到可删除的目标消息", "Target message not found")
}
fn ui_text_message_delete_not_supported() -> &'static str {
    zh_or_en(
        "该消息类型暂不支持删除",
        "This message type cannot be deleted",
    )
}
fn ui_text_thread_not_found() -> &'static str {
    zh_or_en("未找到目标线程", "Target thread not found")
}
fn ui_text_thread_rename_prompt() -> &'static str {
    zh_or_en("请输入新的线程名称", "Please input a new thread name")
}
fn ui_text_thread_rename_empty() -> &'static str {
    zh_or_en("线程名称不能为空", "Thread name cannot be empty")
}
fn ui_text_thread_rename_cancelled() -> &'static str {
    zh_or_en("已取消线程重命名", "Thread rename cancelled")
}
fn ui_text_thread_renamed() -> &'static str {
    zh_or_en("线程已重命名", "Thread renamed")
}
fn ui_text_thread_deleted() -> &'static str {
    zh_or_en("线程已删除", "Thread deleted")
}
fn ui_text_thread_delete_cancelled() -> &'static str {
    zh_or_en("已取消删除线程", "Thread deletion cancelled")
}
fn ui_text_thread_metadata_opened() -> &'static str {
    zh_or_en("已打开线程元数据弹窗", "Thread metadata modal opened")
}
fn ui_text_thread_metadata_closed() -> &'static str {
    zh_or_en("已关闭线程元数据弹窗", "Thread metadata modal closed")
}
fn ui_text_tool_running() -> &'static str {
    zh_or_en("Bash执行中", "Bash running")
}
fn ui_text_tool_finished() -> &'static str {
    zh_or_en("Bash执行完成", "Bash finished")
}
fn ui_text_tool_error() -> &'static str {
    zh_or_en("Bash执行错误", "Bash tool error")
}
fn ui_text_tool_persist_failed() -> &'static str {
    zh_or_en("工具消息写入会话失败", "Failed to persist tool message to session")
}
fn ui_text_tool_persist_failed_hint() -> &'static str {
    zh_or_en(
        "该条工具消息可能无法在重启后完整回放",
        "This tool message may not replay completely after restart",
    )
}
fn ui_text_mcp_tool_call() -> &'static str {
    zh_or_en("MCP工具调用", "MCP tool call")
}
fn ui_text_mode_read() -> &'static str {
    zh_or_en("读", "read")
}
fn ui_text_mode_write() -> &'static str {
    zh_or_en("写", "write")
}
fn ui_text_status_ai_connectivity() -> &'static str {
    zh_or_en("正在检查 AI 连通性...", "Checking AI connectivity...")
}
fn ui_text_status_ai_connectivity_background_started() -> &'static str {
    zh_or_en(
        "AI 连通性检查已在后台运行，可继续输入与切换面板；检查完成前消息不会发送",
        "AI connectivity check is running in background; you can keep typing and switching panels, but messages will not be sent until it finishes",
    )
}
fn ui_text_status_ai_connectivity_pending_send_blocked() -> &'static str {
    zh_or_en(
        "AI 连通性检查进行中，消息暂未发送",
        "AI connectivity check in progress, message not sent yet",
    )
}
fn ui_text_status_ai_pending_send_blocked() -> &'static str {
    zh_or_en(
        "AI 正在处理中，可继续输入与浏览，当前消息暂未发送",
        "AI is still processing; you can keep typing and browsing, but this message is not sent yet",
    )
}
fn ui_text_status_ai_pending_session_switch_blocked() -> &'static str {
    zh_or_en(
        "AI 正在处理中，暂不允许新建或切换会话",
        "AI is still processing; creating or switching sessions is temporarily disabled",
    )
}
fn ui_text_status_ai_connectivity_ok() -> &'static str {
    zh_or_en("AI 连通性检查通过", "AI connectivity check passed")
}
fn ui_text_status_ai_connectivity_failed() -> &'static str {
    zh_or_en("AI 连通性检查失败", "AI connectivity check failed")
}
fn ui_text_theme_switched() -> &'static str {
    zh_or_en("主题已切换", "Theme switched")
}

fn ui_mode_label(mode: UiMode) -> &'static str {
    match mode {
        UiMode::Chat => zh_or_en("聊天", "Chat"),
        UiMode::Skills => zh_or_en("技能", "Skills"),
        UiMode::Mcp => "MCP",
        UiMode::Inspect => zh_or_en("巡检", "Inspect"),
        UiMode::Config => zh_or_en("配置", "Config"),
    }
}

fn inspect_target_label(target: InspectTarget) -> &'static str {
    match target {
        InspectTarget::Cpu => zh_or_en("CPU", "CPU"),
        InspectTarget::Memory => zh_or_en("内存", "Memory"),
        InspectTarget::Disk => zh_or_en("磁盘", "Disk"),
        InspectTarget::Os => zh_or_en("系统", "OS"),
        InspectTarget::Process => zh_or_en("进程", "Process"),
        InspectTarget::Filesystem => zh_or_en("文件系统", "Filesystem"),
        InspectTarget::Hardware => zh_or_en("硬件", "Hardware"),
        InspectTarget::Logs => zh_or_en("日志", "Logs"),
        InspectTarget::Network => zh_or_en("网络", "Network"),
        InspectTarget::All => zh_or_en("全部", "All"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MessageKind;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    fn test_cfg() -> AppConfig {
        toml::from_str(
            r#"
[ai]
base-url = "https://example.invalid/v1"
token = "sk-test"
model = "test-model"
"#,
        )
        .expect("parse test cfg")
    }

    fn new_state() -> ChatUiState {
        let cfg = test_cfg();
        ChatUiState::new(
            Vec::new(),
            Vec::new(),
            "session-test".to_string(),
            0,
            PathBuf::from("/tmp/ui-preferences.json"),
            &cfg,
            Path::new("/tmp/claw.toml"),
        )
    }

    fn mock_overview(id: &str, name: &str, active: bool) -> SessionOverview {
        SessionOverview {
            session_id: id.to_string(),
            session_name: name.to_string(),
            file_path: PathBuf::from(format!("/tmp/{id}.json")),
            message_count: 0,
            summary_len: 0,
            user_count: 0,
            assistant_count: 0,
            tool_count: 0,
            system_count: 0,
            created_at_epoch_ms: 0,
            last_updated_epoch_ms: 0,
            active,
        }
    }

    #[test]
    fn focus_cycle_chat_mode_has_expected_order() {
        let mut state = new_state();
        assert!(matches!(state.focus, FocusPanel::Input));
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Nav));
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Threads));
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Conversation));
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Input));
    }

    #[test]
    fn focus_cycle_inspect_mode_skips_input_panel() {
        let mut state = new_state();
        state.mode = UiMode::Inspect;
        state.focus = FocusPanel::Conversation;
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Nav));
        state.cycle_focus_forward();
        assert!(matches!(state.focus, FocusPanel::Threads));
        state.cycle_focus_backward();
        assert!(matches!(state.focus, FocusPanel::Nav));
        state.cycle_focus_backward();
        assert!(matches!(state.focus, FocusPanel::Conversation));
    }

    #[test]
    fn connectivity_check_success_unblocks_chat_and_resets_status() {
        let mut state = new_state();
        state.ai_connectivity_checking = true;
        state.status = ui_text_status_ai_connectivity().to_string();
        let (tx, rx) = mpsc::channel::<Result<(), AppError>>();
        tx.send(Ok(())).expect("send check result");
        let mut handle = Some(AiConnectivityCheckHandle { rx });

        poll_ai_connectivity_check(&mut handle, &mut state);

        assert!(!state.ai_connectivity_checking);
        assert!(handle.is_none());
        assert_eq!(state.status, ui_text_ready());
        assert!(
            state
                .messages
                .back()
                .is_some_and(|item| item.text.contains(ui_text_status_ai_connectivity_ok()))
        );
    }

    #[test]
    fn connectivity_check_failure_unblocks_chat_and_surfaces_error() {
        let mut state = new_state();
        state.ai_connectivity_checking = true;
        state.status = ui_text_status_ai_connectivity().to_string();
        let (tx, rx) = mpsc::channel::<Result<(), AppError>>();
        tx.send(Err(AppError::Ai("network down".to_string())))
            .expect("send check result");
        let mut handle = Some(AiConnectivityCheckHandle { rx });

        poll_ai_connectivity_check(&mut handle, &mut state);

        assert!(!state.ai_connectivity_checking);
        assert!(handle.is_none());
        assert_eq!(state.status, ui_text_status_ai_connectivity_failed());
        assert!(
            state
                .messages
                .back()
                .is_some_and(|item| item.text.contains(ui_text_status_ai_connectivity_failed()))
        );
    }

    #[test]
    fn apply_selected_inspect_target_switches_mode_and_focus() {
        let mut state = new_state();
        state.mode = UiMode::Chat;
        state.focus = FocusPanel::Input;
        state.inspect_menu_selected = inspect_target_index(InspectTarget::Network);
        apply_selected_inspect_target(&mut state);
        assert!(matches!(state.mode, UiMode::Inspect));
        assert!(matches!(state.focus, FocusPanel::Conversation));
        assert_eq!(
            state.inspect.target.as_str(),
            InspectTarget::Network.as_str()
        );
        assert!(!state.inspect_menu_open);
    }

    #[test]
    fn set_threads_keeps_existing_visual_order_after_refresh() {
        let mut state = new_state();
        state.threads = vec![
            mock_overview("a", "A", false),
            mock_overview("b", "B", false),
            mock_overview("c", "C", true),
        ];
        state.thread_selected = 1;
        state.set_threads(vec![
            mock_overview("c", "C", true),
            mock_overview("a", "A", false),
            mock_overview("b", "B", false),
        ]);
        let order = state
            .threads
            .iter()
            .map(|item| item.session_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(order, vec!["a", "b", "c"]);
        assert_eq!(state.thread_selected, 1);
    }

    #[test]
    fn set_threads_prunes_orphaned_session_cache_entries() {
        let mut state = new_state();
        state.set_active_session("a".to_string());
        state.threads = vec![mock_overview("a", "A", true), mock_overview("b", "B", false)];
        state.session_conversation_cache.insert(
            "a".to_string(),
            SessionConversationCache {
                messages: vec![UiMessage {
                    role: UiRole::System,
                    text: "active".to_string(),
                    tool_meta: None,
                }],
                message_persisted: vec![true],
            },
        );
        state.session_conversation_cache.insert(
            "b".to_string(),
            SessionConversationCache {
                messages: vec![UiMessage {
                    role: UiRole::System,
                    text: "known".to_string(),
                    tool_meta: None,
                }],
                message_persisted: vec![true],
            },
        );
        state.session_conversation_cache.insert(
            "orphan".to_string(),
            SessionConversationCache {
                messages: vec![UiMessage {
                    role: UiRole::System,
                    text: "stale".to_string(),
                    tool_meta: None,
                }],
                message_persisted: vec![true],
            },
        );

        state.set_threads(vec![mock_overview("a", "A", true), mock_overview("b", "B", false)]);

        assert!(state.session_conversation_cache.contains_key("a"));
        assert!(state.session_conversation_cache.contains_key("b"));
        assert!(!state.session_conversation_cache.contains_key("orphan"));
    }

    #[test]
    fn set_threads_prunes_orphaned_auto_title_tracking() {
        let mut state = new_state();
        state.set_active_session("a".to_string());
        state.threads = vec![mock_overview("a", "A", true), mock_overview("b", "B", false)];
        state.session_auto_title_attempted.insert("a".to_string());
        state.session_auto_title_attempted.insert("b".to_string());
        state
            .session_auto_title_attempted
            .insert("orphan".to_string());
        let (_tx_keep, rx_keep) = mpsc::channel::<Result<String, AppError>>();
        state.session_auto_title_workers.push(SessionAutoTitleHandle {
            session_id: "a".to_string(),
            rx: rx_keep,
        });
        let (_tx_drop, rx_drop) = mpsc::channel::<Result<String, AppError>>();
        state.session_auto_title_workers.push(SessionAutoTitleHandle {
            session_id: "orphan".to_string(),
            rx: rx_drop,
        });

        state.set_threads(vec![mock_overview("a", "A", true), mock_overview("b", "B", false)]);

        assert!(state.session_auto_title_attempted.contains("a"));
        assert!(state.session_auto_title_attempted.contains("b"));
        assert!(!state.session_auto_title_attempted.contains("orphan"));
        assert_eq!(state.session_auto_title_workers.len(), 1);
        assert_eq!(state.session_auto_title_workers[0].session_id, "a");
    }

    #[test]
    fn mcp_delete_marks_pending_save_even_when_removed_server_was_clean() {
        let mut state = new_state();
        state.mcp_ui.servers = vec![McpUiServer {
            name: "demo".to_string(),
            config: McpServerConfig {
                enabled: true,
                ..McpServerConfig::default()
            },
            dirty: false,
        }];
        state.mcp_ui.selected_server = 0;
        state.mcp_ui.dirty_count = 0;
        state.mcp_ui.structural_dirty = false;

        mcp_delete_selected_server(&mut state);

        assert!(state.mcp_ui.servers.is_empty());
        assert!(state.mcp_ui.structural_dirty);
        assert!(state.mcp_ui.dirty_count > 0);
    }

    #[test]
    fn mcp_mouse_click_can_select_first_server_row() {
        let mut state = new_state();
        state.mode = UiMode::Mcp;
        state.mcp_ui.servers = vec![
            McpUiServer {
                name: "srv-a".to_string(),
                config: McpServerConfig::default(),
                dirty: false,
            },
            McpUiServer {
                name: "srv-b".to_string(),
                config: McpServerConfig::default(),
                dirty: false,
            },
        ];
        state.mcp_ui.selected_server = 1;
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        let (servers_rect, _) = mcp_panel_regions(layout);
        let inner = servers_rect.inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        let mouse = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: inner.x.saturating_add(1),
            row: inner.y,
            modifiers: KeyModifiers::NONE,
        };
        mcp_select_by_mouse(layout, mouse, &mut state);
        assert_eq!(state.mcp_ui.selected_server, 0);
        assert!(state.mcp_ui.focus_servers);
    }

    #[test]
    fn conversation_bubble_border_is_closed_for_long_system_line() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::System,
            text: "当前命令不强制要求 root/管理员权限，按当前用户继续运行".to_string(),
            tool_meta: None,
        });
        state.conversation_wrap_width = 48;
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let bubble_lines = state
            .conversation_lines
            .iter()
            .filter(|line| {
                !matches!(
                    line.kind,
                    ConversationLineKind::Header | ConversationLineKind::Spacer
                )
            })
            .collect::<Vec<_>>();
        assert!(!bubble_lines.is_empty());

        let top = bubble_lines.first().expect("top line");
        let bottom = bubble_lines.last().expect("bottom line");
        assert!(top.text.starts_with('╭') && top.text.ends_with('╮'));
        assert!(bottom.text.starts_with('╰') && bottom.text.ends_with('╯'));

        for line in &bubble_lines[1..bubble_lines.len().saturating_sub(1)] {
            if line.kind == ConversationLineKind::Body {
                assert!(line.text.starts_with("│ ") && line.text.ends_with(" │"));
            }
        }
    }

    #[test]
    fn conversation_thinking_bubble_right_border_stays_collinear_for_mixed_text() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::Thinking,
            text: "看起来有一些音频相关进程（如AudioToolbox），但没有明显的音乐播放器。用户可能只是想要一些时间，或者想让我等一下。\n\n考虑到我的角色是系统巡检助手，我可以提供系统状态摘要，并说明一首歌的时间大约是3-5分钟，期间系统可以继续运行。".to_string(),
            tool_meta: None,
        });
        state.conversation_wrap_width = 92;
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let mut widths = Vec::<usize>::new();
        for line in state.conversation_lines.iter().filter(|line| {
            line.message_index == Some(0)
                && matches!(
                    line.kind,
                    ConversationLineKind::BubbleBorder | ConversationLineKind::Body
                )
        }) {
            widths.push(text_display_width(&line.text));
        }
        assert!(!widths.is_empty());
        let expected = widths[0];
        assert!(widths.iter().all(|item| *item == expected));
    }

    #[test]
    fn display_width_treats_variation_selector_as_zero_width() {
        assert_eq!(text_display_width("🔍"), text_display_width("🔍\u{fe0f}"));
        assert_eq!(text_display_width("✈"), text_display_width("✈\u{fe0f}"));
    }

    #[test]
    fn display_width_treats_single_emoji_as_one_cell() {
        assert_eq!(text_display_width("😘"), 1);
        assert_eq!(text_display_width("😊"), 1);
    }

    #[test]
    fn conversation_bubble_right_border_stays_collinear_with_emoji_and_zwj() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::Assistant,
            text: "🔍️ 下一步建议\n👨‍👩‍👧‍👦 家庭表情也应保持边框对齐".to_string(),
            tool_meta: None,
        });
        state.conversation_wrap_width = 64;
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let widths = state
            .conversation_lines
            .iter()
            .filter(|line| {
                line.message_index == Some(0)
                    && matches!(
                        line.kind,
                        ConversationLineKind::BubbleBorder | ConversationLineKind::Body
                    )
            })
            .map(|line| text_display_width(&line.text))
            .collect::<Vec<_>>();
        assert!(!widths.is_empty());
        assert!(widths.iter().all(|item| *item == widths[0]));
    }

    #[test]
    fn conversation_bubble_width_is_capped_in_narrow_viewport() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::Assistant,
            text: "窄窗口也不应出现气泡宽度超出会话区".to_string(),
            tool_meta: None,
        });
        state.conversation_wrap_width = 8;
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let max_width = state
            .conversation_lines
            .iter()
            .filter(|line| line.message_index == Some(0))
            .map(|line| text_display_width(&line.text))
            .max()
            .unwrap_or(0);
        assert!(max_width <= 8);
    }

    #[test]
    fn wrap_text_by_display_width_no_leading_empty_line_when_first_char_exceeds_limit() {
        let wrapped = wrap_text_by_display_width("🔍", 1);
        assert_eq!(wrapped, vec!["🔍".to_string()]);
    }

    #[test]
    fn conversation_user_header_right_edge_matches_bubble_width() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::User,
            text: "给我写一个冒泡排序！".to_string(),
            tool_meta: None,
        });
        state.conversation_wrap_width = 54;
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let header = state
            .conversation_lines
            .iter()
            .find(|line| line.message_index == Some(0) && line.kind == ConversationLineKind::Header)
            .expect("user header line");
        let top_border = state
            .conversation_lines
            .iter()
            .find(|line| {
                line.message_index == Some(0) && line.kind == ConversationLineKind::BubbleBorder
            })
            .expect("user bubble top line");
        assert_eq!(
            text_display_width(&header.text),
            text_display_width(&top_border.text)
        );
    }

    #[test]
    fn conversation_user_lines_fill_viewport_width_after_padding() {
        let mut state = new_state();
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
        state.messages.push_back(UiMessage {
            role: UiRole::User,
            text: "hello".to_string(),
            tool_meta: None,
        });
        state.messages.push_back(UiMessage {
            role: UiRole::Assistant,
            text: "world".to_string(),
            tool_meta: None,
        });
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        let rendered = conversation_text(&state, palette_by_index(0));
        assert_eq!(rendered.lines.len(), state.conversation_lines.len());
        for (rendered_line, source_line) in rendered.lines.iter().zip(&state.conversation_lines) {
            if source_line.kind == ConversationLineKind::Spacer {
                continue;
            }
            if source_line.role == UiRole::User {
                assert_eq!(
                    text_display_width(&source_line.text),
                    layout.conversation_body.width as usize
                );
            } else {
                assert_eq!(rendered_line.alignment, None);
            }
        }
    }

    #[test]
    fn conversation_non_user_header_lines_fill_viewport_width_after_padding() {
        let mut state = new_state();
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
        state.messages.push_back(UiMessage {
            role: UiRole::System,
            text: "session switched".to_string(),
            tool_meta: None,
        });
        state.messages.push_back(UiMessage {
            role: UiRole::Thinking,
            text: "reasoning text".to_string(),
            tool_meta: None,
        });
        state.messages.push_back(UiMessage {
            role: UiRole::Assistant,
            text: "reply".to_string(),
            tool_meta: None,
        });
        state.conversation_dirty = true;
        state.ensure_conversation_cache();

        for line in state
            .conversation_lines
            .iter()
            .filter(|line| line.kind == ConversationLineKind::Header)
        {
            assert_eq!(
                text_display_width(&line.text),
                layout.conversation_body.width as usize
            );
        }
    }

    #[test]
    fn conversation_body_side_borders_use_same_border_color_as_top_and_bottom() {
        let mut state = new_state();
        state.set_conversation_wrap_width(80);
        state.push(UiRole::Tool, "tool output");
        state.ensure_conversation_cache();

        let rendered = conversation_text(&state, palette_by_index(0));
        let Some((body_idx, _)) = state
            .conversation_lines
            .iter()
            .enumerate()
            .find(|(_, line)| line.kind == ConversationLineKind::Body)
        else {
            panic!("body line not found");
        };
        let spans = &rendered.lines[body_idx].spans;
        assert!(spans.len() >= 4);
        let expected = Some(role_border_color(UiRole::Tool, palette_by_index(0)));
        assert_eq!(spans[1].style.fg, expected);
        assert_eq!(spans[3].style.fg, expected);
    }

    #[test]
    fn conversation_copy_button_aligns_to_hovered_user_message() {
        let mut state = new_state();
        state.push(UiRole::User, "hello");
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
        state.ensure_conversation_cache();
        state.clamp_scroll(layout.conversation_body.height.max(1));
        state.hovered_message_idx = Some(0);

        let button = conversation_copy_button_data(&state, layout).expect("button should exist");
        let delete_button =
            conversation_delete_button_data(&state, layout).expect("delete button should exist");
        assert_eq!(button.message_index, 0);
        assert_eq!(delete_button.message_index, 0);
        assert!(button.rect.x < delete_button.rect.x);
        assert_eq!(
            delete_button
                .rect
                .x
                .saturating_add(delete_button.rect.width),
            layout
                .conversation_body
                .x
                .saturating_add(layout.conversation_body.width)
        );
    }

    #[test]
    fn conversation_delete_button_is_rendered_after_copy_button() {
        let mut state = new_state();
        state.push(UiRole::Assistant, "hello");
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
        state.ensure_conversation_cache();
        state.clamp_scroll(layout.conversation_body.height.max(1));
        state.hovered_message_idx = Some(0);

        let copy = conversation_copy_button_data(&state, layout).expect("copy button should exist");
        let delete =
            conversation_delete_button_data(&state, layout).expect("delete button should exist");
        assert_eq!(copy.message_index, 0);
        assert_eq!(delete.message_index, 0);
        assert_eq!(delete.rect.y, copy.rect.y);
        assert!(delete.rect.x > copy.rect.x);
    }

    #[test]
    fn tool_message_with_meta_renders_result_button_between_copy_and_delete() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::Tool,
            text: "Bash执行完成: ok=true exit=Some(0) timeout=false blocked=false".to_string(),
            tool_meta: Some(ToolExecutionMeta {
                tool_call_id: "call_1".to_string(),
                function_name: "run_shell_command".to_string(),
                command: "pwd".to_string(),
                arguments: "{\"command\":\"pwd\"}".to_string(),
                result_payload: "{\"ok\":true,\"stdout\":\"/tmp\"}".to_string(),
                executed_at_epoch_ms: 1,
                account: "tester".to_string(),
                environment: "dev".to_string(),
                os_name: "macOS".to_string(),
                cwd: "/tmp".to_string(),
                mode: "read".to_string(),
                label: "chat_tool".to_string(),
                exit_code: Some(0),
                duration_ms: 12,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        });
        state.message_persisted.push_back(false);
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        state.set_conversation_wrap_width(layout.conversation_body.width.max(1));
        state.ensure_conversation_cache();
        state.clamp_scroll(layout.conversation_body.height.max(1));
        state.hovered_message_idx = Some(0);

        let copy = conversation_copy_button_data(&state, layout).expect("copy button should exist");
        let result = conversation_tool_result_button_data(&state, layout)
            .expect("result button should exist");
        let delete =
            conversation_delete_button_data(&state, layout).expect("delete button should exist");
        assert_eq!(copy.rect.y, result.rect.y);
        assert_eq!(result.rect.y, delete.rect.y);
        assert!(copy.rect.x < result.rect.x);
        assert!(result.rect.x < delete.rect.x);
    }

    #[test]
    fn pending_ai_send_block_notice_is_deduplicated_and_keeps_input() {
        let mut state = new_state();
        stage_blocked_pending_message(&mut state, "next message");
        mark_pending_ai_send_blocked(&mut state);
        mark_pending_ai_send_blocked(&mut state);

        assert_eq!(state.input.text, "next message");
        let notices = state
            .messages
            .iter()
            .filter(|item| {
                item.role == UiRole::System && item.text == ui_text_status_ai_pending_send_blocked()
            })
            .count();
        assert_eq!(notices, 1);
    }

    #[test]
    fn pending_ai_input_enter_does_not_send_and_preserves_text() {
        let mut state = new_state();
        state.mode = UiMode::Chat;
        state.input.text = "queued question".to_string();
        state.input.cursor_char = state.input.char_count();
        state.focus = FocusPanel::Input;

        let handled = handle_pending_ai_input_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
        )
        .expect("pending ai input handler should not fail");
        assert!(!handled);
        assert_eq!(state.input.text, "queued question");
        assert_eq!(state.status, ui_text_status_ai_pending_send_blocked());
    }

    #[test]
    fn should_schedule_session_auto_title_requires_default_name_and_first_user_message() {
        let attempted = HashSet::<String>::new();
        assert!(should_schedule_session_auto_title(
            &attempted,
            "abc",
            "session-abc",
            1
        ));
        assert!(!should_schedule_session_auto_title(
            &attempted,
            "abc",
            "session-abc",
            2
        ));
        assert!(!should_schedule_session_auto_title(
            &attempted,
            "abc",
            "custom-name",
            1
        ));
        let mut attempted = HashSet::<String>::new();
        attempted.insert("abc".to_string());
        assert!(!should_schedule_session_auto_title(
            &attempted,
            "abc",
            "session-abc",
            1
        ));
    }

    #[test]
    fn normalize_session_auto_title_candidate_trims_prefix_and_limits_units() {
        let en = normalize_session_auto_title_candidate(
            " Title: Build a robust async thread switch regression fix with tests ",
            6,
        )
        .expect("title should be normalized");
        assert_eq!(en, "Build a robust async thread switch");

        let zh = normalize_session_auto_title_candidate(
            "标题：这是一个用于验证会话标题自动截断行为的中文标题示例",
            15,
        )
        .expect("title should be normalized");
        assert!(zh.chars().count() <= 15);
    }

    #[test]
    fn session_auto_title_prompts_follow_app_language() {
        let (_, zh_cn_prompt) =
            build_session_auto_title_prompts(i18n::Language::ZhCn, "测试消息");
        assert!(zh_cn_prompt.contains("简体中文"));

        let (_, zh_tw_prompt) =
            build_session_auto_title_prompts(i18n::Language::ZhTw, "測試訊息");
        assert!(zh_tw_prompt.contains("繁體中文"));

        let (_, en_prompt) =
            build_session_auto_title_prompts(i18n::Language::En, "test message");
        assert!(en_prompt.contains("Output language must be English"));
    }

    #[test]
    fn selected_thread_is_active_session_matches_current_session_id() {
        let mut state = new_state();
        state.threads = vec![
            mock_overview("session-a", "A", true),
            mock_overview("session-b", "B", false),
        ];
        state.set_active_session("session-a".to_string());
        state.thread_selected = 0;
        assert!(selected_thread_is_active_session(&state));
        state.thread_selected = 1;
        assert!(!selected_thread_is_active_session(&state));
    }

    #[test]
    fn pending_ai_threads_enter_allows_return_to_chat_for_active_thread_only() {
        let mut state = new_state();
        state.mode = UiMode::Skills;
        state.focus = FocusPanel::Threads;
        state.threads = vec![
            mock_overview("session-a", "A", true),
            mock_overview("session-b", "B", false),
        ];
        state.set_active_session("session-a".to_string());

        state.thread_selected = 0;
        let handled = handle_pending_ai_threads_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
        )
        .expect("pending ai threads key should not fail");
        assert!(!handled);
        assert!(matches!(state.mode, UiMode::Chat));
        assert_eq!(state.status, ui_text_focus_hint(FocusPanel::Threads));

        state.mode = UiMode::Skills;
        state.thread_selected = 1;
        let handled = handle_pending_ai_threads_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
        )
        .expect("pending ai threads key should not fail");
        assert!(!handled);
        assert!(matches!(state.mode, UiMode::Skills));
        assert_eq!(
            state.status,
            ui_text_status_ai_pending_session_switch_blocked()
        );
    }

    #[test]
    fn persisted_occurrence_counter_ignores_transient_duplicates() {
        let mut state = new_state();
        state.push_persisted(UiRole::Assistant, "dup");
        state.push(UiRole::Assistant, "dup");
        state.push_persisted(UiRole::Assistant, "dup");
        let first = same_persisted_message_occurrence_from_end(&state, 2).expect("occurrence");
        let second = same_persisted_message_occurrence_from_end(&state, 0).expect("occurrence");
        assert_eq!(first, 1);
        assert_eq!(second, 2);
    }

    #[test]
    fn streamed_assistant_message_can_be_marked_persisted_after_save() {
        let mut state = new_state();
        state.push_stream_chunk(UiRole::Assistant, "hello", None);
        assert!(!state.is_message_persisted(0));
        state.mark_last_message_persisted_if_matches(UiRole::Assistant, "hello");
        assert!(state.is_message_persisted(0));
    }

    #[test]
    fn stream_chunks_append_to_single_assistant_message() {
        let mut state = new_state();
        state.push(UiRole::User, "hello");
        state.push_stream_chunk(UiRole::Assistant, "wo", None);
        state.push_stream_chunk(UiRole::Assistant, "rld", None);

        let assistant_messages = state
            .messages
            .iter()
            .filter(|item| item.role == UiRole::Assistant)
            .collect::<Vec<_>>();
        assert_eq!(assistant_messages.len(), 1);
        assert_eq!(assistant_messages[0].text, "world");
    }

    #[test]
    fn stream_thinking_chunks_keep_single_message() {
        let mut state = new_state();
        state.push(UiRole::User, "probe");
        state.push_stream_chunk(UiRole::Thinking, "step-1", None);
        state.push_stream_chunk(UiRole::Thinking, " + step-2", None);

        let thinking_messages = state
            .messages
            .iter()
            .filter(|item| item.role == UiRole::Thinking)
            .collect::<Vec<_>>();
        assert_eq!(thinking_messages.len(), 1);
        assert_eq!(thinking_messages[0].text, "step-1 + step-2");
    }

    #[test]
    fn streamed_thinking_message_can_be_marked_persisted_by_role_batch() {
        let mut state = new_state();
        state.push_stream_chunk(UiRole::Thinking, "step-1", None);
        state.push_stream_chunk(UiRole::Assistant, "done", None);

        let marked = state.mark_all_unpersisted_messages_by_role(UiRole::Thinking);
        assert_eq!(marked, 1);
        assert!(state.is_message_persisted(0));
        assert!(!state.is_message_persisted(1));
    }

    #[test]
    fn mark_all_unpersisted_messages_by_role_only_marks_target_role() {
        let mut state = new_state();
        state.push(UiRole::Thinking, "t-1");
        state.push(UiRole::Assistant, "a-1");
        state.push(UiRole::Thinking, "t-2");
        state.push_persisted(UiRole::Thinking, "t-3");
        assert!(!state.is_message_persisted(0));
        assert!(!state.is_message_persisted(1));
        assert!(!state.is_message_persisted(2));
        assert!(state.is_message_persisted(3));

        let marked = state.mark_all_unpersisted_messages_by_role(UiRole::Thinking);
        assert_eq!(marked, 2);
        assert!(state.is_message_persisted(0));
        assert!(!state.is_message_persisted(1));
        assert!(state.is_message_persisted(2));
        assert!(state.is_message_persisted(3));
    }

    #[test]
    fn report_tool_session_persist_failure_pushes_system_notice() {
        let mut state = new_state();
        report_tool_session_persist_failure(
            &mut state,
            &AppError::Runtime("disk full".to_string()),
            "run_shell_command",
        );
        assert_eq!(state.status, ui_text_tool_persist_failed());
        let notice = state.messages.back().expect("system notice should exist");
        assert_eq!(notice.role, UiRole::System);
        assert!(notice.text.contains(ui_text_tool_persist_failed()));
        assert!(notice.text.contains("disk full"));
    }

    #[test]
    fn session_cache_restores_transient_thinking_messages() {
        let mut state = new_state();
        state.set_active_session("session-a".to_string());
        state.push_persisted(UiRole::User, "u1");
        state.push(UiRole::Thinking, "step-a");
        state.remember_current_session_messages();

        state.set_active_session("session-b".to_string());
        state.clear_conversation_viewport_only();
        state.push_persisted(UiRole::Assistant, "b1");

        let restored = state.session_messages_or_fallback(
            "session-a",
            vec![UiMessage {
                role: UiRole::User,
                text: "fallback".to_string(),
                tool_meta: None,
            }],
        );
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].0.role, UiRole::User);
        assert!(restored[0].1);
        assert_eq!(restored[1].0.role, UiRole::Thinking);
        assert!(!restored[1].1);
    }

    #[test]
    fn recent_messages_to_ui_messages_restores_thinking_role() {
        let mapped = recent_messages_to_ui_messages(&[SessionMessage {
            role: "thinking".to_string(),
            content: "reasoning".to_string(),
            kind: MessageKind::Assistant,
            group_id: None,
            created_at_epoch_ms: 1,
            tool_meta: None,
        }]);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role, UiRole::Thinking);
        assert_eq!(mapped[0].text, "reasoning");
    }

    #[test]
    fn persisted_tool_message_is_rendered_with_multiline_fields() {
        let item = UiMessage {
            role: UiRole::Tool,
            text: r#"tool_call_id=call_1 function=run_shell_command args={"command":"pwd"} result={"ok":true,"stdout":"line1\nline2"}"#.to_string(),
            tool_meta: None,
        };
        let rendered = conversation_render_source_text(&item).into_owned();
        assert!(rendered.contains("tool_call_id=call_1\nfunction=run_shell_command"));
        assert!(rendered.contains("\"command\": \"pwd\""));
        assert!(rendered.contains("\"stdout\": \"line1\\nline2\""));
    }

    #[test]
    fn persisted_tool_message_parsing_uses_last_result_delimiter() {
        let item = UiMessage {
            role: UiRole::Tool,
            text: r#"tool_call_id=call_9 function=run_shell_command args={"command":"echo \" result=marker\""} result={"ok":true,"stdout":"done"}"#.to_string(),
            tool_meta: None,
        };
        let rendered = conversation_render_source_text(&item).into_owned();
        assert!(rendered.contains("\"command\": \"echo \\\" result=marker\\\"\""));
        assert!(rendered.contains("\"stdout\": \"done\""));
    }

    #[test]
    fn tool_message_render_preserves_structured_line_breaks_after_reload_path() {
        let item = UiMessage {
            role: UiRole::Tool,
            text: r#"tool_call_id=call_2 function=run_shell_command args={"command":"echo hi"} result={"ok":true}"#.to_string(),
            tool_meta: None,
        };
        let rendered = render_conversation_item_text(&item, false);
        assert!(rendered.contains("tool_call_id=call_2"));
        assert!(rendered.contains("\nfunction=run_shell_command"));
        assert!(rendered.contains("\nargs="));
        assert!(rendered.contains("\nresult="));
    }

    #[test]
    fn tool_message_with_meta_and_raw_trace_renders_status_summary() {
        let item = UiMessage {
            role: UiRole::Tool,
            text: r#"tool_call_id=call_2 function=run_shell_command args={"command":"echo hi"} result={"ok":true,"exit_code":0}"#.to_string(),
            tool_meta: Some(ToolExecutionMeta {
                tool_call_id: "call_2".to_string(),
                function_name: "run_shell_command".to_string(),
                command: "echo hi".to_string(),
                arguments: "{\"command\":\"echo hi\"}".to_string(),
                result_payload: "{\"ok\":true,\"exit_code\":0}".to_string(),
                executed_at_epoch_ms: 1,
                account: String::new(),
                environment: String::new(),
                os_name: String::new(),
                cwd: String::new(),
                mode: "read".to_string(),
                label: "chat_tool".to_string(),
                exit_code: Some(0),
                duration_ms: 10,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        };
        let rendered = conversation_render_source_text(&item).into_owned();
        assert!(!rendered.contains("tool_call_id="));
        assert!(rendered.contains("ok=true"));
        assert!(rendered.contains("exit=Some(0)"));
    }

    #[test]
    fn legacy_tool_message_without_result_is_formatted_for_display() {
        let item = UiMessage {
            role: UiRole::Tool,
            text: r#"tool_call_id=call_cache function=run_shell_command args={"command":"ls"} cache_hit=true"#.to_string(),
            tool_meta: None,
        };
        let rendered = conversation_render_source_text(&item).into_owned();
        assert!(rendered.contains("tool_call_id=call_cache"));
        assert!(rendered.contains("\nfunction=run_shell_command"));
        assert!(rendered.contains("\nargs="));
        assert!(rendered.contains("\ncache_hit=true"));
    }

    #[test]
    fn conversation_cache_has_no_trailing_spacer_line() {
        let mut state = new_state();
        state.push(UiRole::User, "first");
        state.push(UiRole::Assistant, "second");
        state.ensure_conversation_cache();
        assert_ne!(
            state.conversation_lines.last().map(|line| line.kind),
            Some(ConversationLineKind::Spacer)
        );
    }

    #[test]
    fn conversation_markdown_heading_and_fences_are_not_shown_raw() {
        let mut state = new_state();
        state.push(
            UiRole::Assistant,
            "### 标题\n\n```rust\nfn main() {}\n```\n\n普通文本",
        );
        state.ensure_conversation_cache();
        let body = state
            .conversation_lines
            .iter()
            .filter(|line| line.kind == ConversationLineKind::Body)
            .map(|line| line.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!body.contains("### "));
        assert!(!body.contains("```"));
        assert!(body.contains("标题"));
        assert!(body.contains("main"));
        assert!(body.contains("普通文本"));
    }

    #[test]
    fn normalize_conversation_markdown_strips_hash_headings_only() {
        let normalized = normalize_conversation_markdown_for_bubble("## Hello\n#1-not-heading");
        assert!(normalized.contains("Hello"));
        assert!(!normalized.contains("## Hello"));
        assert!(normalized.contains("#1-not-heading"));
    }

    #[test]
    fn normalize_conversation_markdown_strips_compacted_multi_hash_headings() {
        let normalized = normalize_conversation_markdown_for_bubble("###1. title\n##2** item");
        assert!(normalized.contains("1. title"));
        assert!(normalized.contains("2** item"));
        assert!(!normalized.contains("###1. title"));
        assert!(!normalized.contains("##2** item"));
    }

    #[test]
    fn streaming_inflight_markdown_keeps_markdown_rendering_enabled() {
        let inflight = render_conversation_markdown("### title\n- item", true);
        let finalized = render_conversation_markdown("### title\n- item", false);
        assert!(inflight.contains("title"));
        assert!(!inflight.contains("### title"));
        assert!(inflight.contains("• item") || inflight.contains("- item"));
        assert!(finalized.contains("title"));
        assert!(finalized.contains("• item") || finalized.contains("- item"));
        assert_eq!(inflight, finalized);
    }

    #[test]
    fn streaming_inflight_plain_text_uses_normalized_raw_fast_path() {
        let raw = "plain stream line 1\nplain stream line 2";
        let inflight = render_conversation_markdown(raw, true);
        let expected = normalize_conversation_markdown_for_bubble(raw);
        assert_eq!(inflight, expected);
    }

    #[test]
    fn streaming_inflight_incomplete_inline_link_keeps_raw_text() {
        let raw = "visit [OpenAI](https://platform.openai.com/doc";
        let inflight = render_conversation_markdown(raw, true);
        assert!(inflight.contains("[OpenAI](https://platform.openai.com/doc"));
    }

    #[test]
    fn streaming_inflight_complete_inline_link_keeps_raw_until_finalized() {
        let raw = "visit [OpenAI](https://platform.openai.com/docs)";
        let inflight = render_conversation_markdown(raw, true);
        let finalized = render_conversation_markdown(raw, false);
        assert!(inflight.contains("OpenAI"));
        assert!(inflight.contains("[OpenAI]("));
        assert!(!finalized.contains("[OpenAI]("));
    }

    #[test]
    fn streaming_inflight_plain_url_keeps_raw_text_for_stability() {
        let raw = "details: https://platform.openai.com/docs/guides";
        let inflight = render_conversation_markdown(raw, true);
        assert!(inflight.contains("https://platform.openai.com/docs/guides"));
    }

    #[test]
    fn unfinished_inline_link_detector_ignores_code_span() {
        assert!(!has_unfinished_inline_markdown_link(
            "`[OpenAI](https://platform.openai.com/doc`"
        ));
    }

    #[test]
    fn skill_modal_uses_mdansi_and_hides_raw_markdown_markers() {
        let rendered = render_skill_markdown_for_modal(
            "### Skill Title\n\n```rust\nlet x = 1;\n```\n\n- item",
        );
        assert!(rendered.contains("Skill Title"));
        assert!(rendered.contains("let x = 1;"));
        assert!(rendered.contains("• item"));
        assert!(!rendered.contains("### "));
        assert!(!rendered.contains("```"));
    }

    #[test]
    fn skill_modal_scroll_max_respects_wrapped_lines() {
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        let modal = SkillDocModalState {
            skill_name: "demo".to_string(),
            file_path: PathBuf::from("/tmp/demo/SKILL.md"),
            raw_content: String::new(),
            rendered_content: std::iter::repeat_n("line", 80)
                .collect::<Vec<_>>()
                .join("\n"),
            scroll: 0,
        };
        let max_scroll = skill_doc_modal_scroll_max(&modal, layout);
        assert!(max_scroll > 0);
    }

    #[test]
    fn skill_modal_scroll_is_clamped_to_renderable_range() {
        let layout = compute_layout(Rect::new(0, 0, 120, 40));
        let mut state = new_state();
        state.skill_doc_modal = Some(SkillDocModalState {
            skill_name: "demo".to_string(),
            file_path: PathBuf::from("/tmp/demo/SKILL.md"),
            raw_content: String::new(),
            rendered_content: "x".repeat(600),
            scroll: u16::MAX,
        });
        clamp_skill_doc_modal_scroll(&mut state, layout);
        let modal = state.skill_doc_modal.as_ref().expect("modal exists");
        assert!(modal.scroll < u16::MAX);
        assert_eq!(modal.scroll, skill_doc_modal_scroll_max(modal, layout));
    }

    #[test]
    fn tool_result_modal_scroll_max_counts_empty_separator_lines() {
        let layout = compute_layout(Rect::new(0, 0, 120, 18));
        let modal = ToolResultModalState {
            message_index: 0,
            lines: std::iter::repeat_n(String::new(), 24).collect::<Vec<_>>(),
            scroll: 0,
        };
        let max_scroll = tool_result_modal_scroll_max(&modal, layout);
        assert!(max_scroll > 0);
    }

    #[test]
    fn tool_result_modal_includes_output_section_from_tool_meta() {
        let mut state = new_state();
        state.messages.push_back(UiMessage {
            role: UiRole::Tool,
            text: "Bash执行完成: ok=true exit=Some(0) timeout=false blocked=false".to_string(),
            tool_meta: Some(ToolExecutionMeta {
                tool_call_id: "call-output".to_string(),
                function_name: "run_shell_command".to_string(),
                command: "date".to_string(),
                arguments: "{\"command\":\"date\"}".to_string(),
                result_payload: "{\"ok\":true,\"stdout\":\"Sun Mar 15 12:00:00\"}".to_string(),
                executed_at_epoch_ms: 1,
                account: "tester".to_string(),
                environment: "prod".to_string(),
                os_name: "macos".to_string(),
                cwd: "/tmp".to_string(),
                mode: "read".to_string(),
                label: "获取当前时间".to_string(),
                exit_code: Some(0),
                duration_ms: 19,
                timed_out: false,
                interrupted: false,
                blocked: false,
            }),
        });
        state.message_persisted.push_back(false);
        open_tool_result_modal(&mut state, 0);
        let modal = state
            .pending_tool_result_modal
            .as_ref()
            .expect("tool result modal should be opened");
        let content = modal.lines.join("\n");
        assert!(content.contains(ui_text_message_tool_result_meta_output()));
        assert!(content.contains("Sun Mar 15 12:00:00"));
    }

    #[test]
    fn attach_tool_meta_to_recent_running_message_keeps_result_modal_consistent() {
        let mut state = new_state();
        state.push(
            UiRole::Tool,
            format!(
                "{}: demo [{}]\n{}",
                ui_text_tool_running(),
                ui_text_mode_read(),
                "pwd"
            ),
        );
        state.push(
            UiRole::Tool,
            format!(
                "{}: ok=true exit=Some(0) timeout=false blocked=false",
                ui_text_tool_finished()
            ),
        );
        let meta = ToolExecutionMeta {
            tool_call_id: "call-consistent".to_string(),
            function_name: "run_shell_command".to_string(),
            command: "pwd".to_string(),
            arguments: "{\"command\":\"pwd\"}".to_string(),
            result_payload: "{\"ok\":true,\"stdout\":\"/tmp\"}".to_string(),
            executed_at_epoch_ms: 1,
            account: "tester".to_string(),
            environment: "dev".to_string(),
            os_name: "macos".to_string(),
            cwd: "/tmp".to_string(),
            mode: "read".to_string(),
            label: "chat_tool".to_string(),
            exit_code: Some(0),
            duration_ms: 10,
            timed_out: false,
            interrupted: false,
            blocked: false,
        };

        attach_tool_meta_to_last_tool_message(&mut state, &meta);
        attach_tool_meta_to_recent_running_tool_message(&mut state, &meta);

        assert!(state.messages[0].tool_meta.is_some());
        assert!(state.messages[1].tool_meta.is_some());
        let running_detail = tool_result_detail_from_message(&state.messages[0])
            .expect("running message should have tool detail");
        let finished_detail = tool_result_detail_from_message(&state.messages[1])
            .expect("finished message should have tool detail");
        assert_eq!(running_detail.tool_call_id, finished_detail.tool_call_id);
        assert_eq!(running_detail.result_payload, finished_detail.result_payload);
    }

    #[test]
    fn extract_http_status_code_parses_mcp_error_text() {
        assert_eq!(
            extract_http_status_code(
                "MCP http request failed: status=429 Too Many Requests, body=oops"
            ),
            Some(429)
        );
        assert_eq!(
            extract_http_status_code("MCP sse post failed: status=504 Gateway Timeout"),
            Some(504)
        );
        assert_eq!(extract_http_status_code("network error"), None);
    }

    #[test]
    fn mcp_exit_code_from_error_falls_back_to_500() {
        let with_status = AppError::Runtime(
            "MCP http request failed: status=404 Not Found, body={}".to_string(),
        );
        assert_eq!(mcp_exit_code_from_error(&with_status), 404);
        let without_status = AppError::Runtime("MCP request timeout".to_string());
        assert_eq!(mcp_exit_code_from_error(&without_status), 500);
    }

    #[test]
    fn tool_result_detail_from_persisted_mcp_message_uses_function_as_command() {
        let message = UiMessage {
            role: UiRole::Tool,
            text: "tool_call_id=call-mcp function=mcp__news_headlines__get_news_list args={} result={\"ok\":true,\"tool\":\"mcp__news_headlines__get_news_list\"}".to_string(),
            tool_meta: None,
        };
        let detail = tool_result_detail_from_message(&message).expect("detail should exist");
        assert_eq!(detail.command, "mcp__news_headlines__get_news_list");
    }

    #[test]
    fn tool_result_modal_output_keeps_large_mcp_payload() {
        let payload = json!({
            "ok": true,
            "tool": "mcp__news_headlines__get_news_list",
            "content": "x".repeat(20_000)
        })
        .to_string();
        let rendered = tool_result_output_for_modal(&payload);
        assert!(rendered.chars().count() > 12_000);
        assert!(!rendered.ends_with("..."));
    }

    #[test]
    fn follow_tail_keeps_newest_user_message_visible() {
        let mut state = new_state();
        state.set_conversation_wrap_width(60);
        for idx in 0..16 {
            state.push(UiRole::Assistant, format!("history {idx}"));
        }
        state.ensure_conversation_cache();
        state.follow_tail = true;
        state.clamp_scroll(8);

        state.push(UiRole::User, "latest-user-message");
        state.ensure_conversation_cache();
        state.clamp_scroll(8);

        let start = state.conversation_scroll as usize;
        let end = (start + 8).min(state.conversation_lines.len());
        let visible = &state.conversation_lines[start..end];
        assert!(
            visible
                .iter()
                .any(|line| line.text.contains("latest-user-message"))
        );
    }

    #[test]
    fn follow_tail_reserves_one_bottom_padding_line() {
        let mut state = new_state();
        state.set_conversation_wrap_width(60);
        for idx in 0..10 {
            state.push(UiRole::Assistant, format!("item {idx}"));
        }
        state.ensure_conversation_cache();
        state.follow_tail = true;
        state.clamp_scroll(8);

        let base_max = state
            .conversation_line_count
            .saturating_sub(8)
            .min(u16::MAX as usize) as u16;
        assert_eq!(state.conversation_scroll, base_max.saturating_add(1));
    }

    #[test]
    fn config_activate_selected_field_cycles_option_field() {
        let mut state = new_state();
        let field_idx = config_selected_field_index(&state).expect("selected field idx");
        assert_eq!(state.config_ui.fields[field_idx].key, "app.language");
        assert_eq!(state.config_ui.fields[field_idx].value, "");

        config_activate_selected_field(&mut state, false);

        assert_eq!(state.config_ui.fields[field_idx].value, "zh-CN");
        assert!(state.config_ui.fields[field_idx].dirty);
        assert!(!state.config_ui.editing);
    }

    #[test]
    fn parse_config_enum_rejects_unknown_value() {
        let field = ConfigField {
            key: "ai.chat.model-price-check-mode".to_string(),
            label: "model-price-check-mode".to_string(),
            category: "ai.chat".to_string(),
            kind: ConfigFieldKind::Enum,
            value: "invalid".to_string(),
            required: false,
            options: vec!["sync".to_string(), "async".to_string()],
            dirty: true,
        };
        let err = parse_config_field_to_item(&field).expect_err("invalid enum should fail");
        assert!(err.to_string().contains("must be one of"));
    }

    #[test]
    fn mcp_cycle_field_option_supports_optional_string_options() {
        let mut state = new_state();
        state.mcp_ui.servers = vec![McpUiServer {
            name: "demo".to_string(),
            config: McpServerConfig::default(),
            dirty: false,
        }];
        state.mcp_ui.selected_server = 0;
        state.mcp_ui.selected_field = mcp_field_defs()
            .iter()
            .position(|item| item.id == McpFieldId::AuthType)
            .expect("auth type field");
        assert!(
            state
                .mcp_ui
                .servers
                .first()
                .and_then(|item| item.config.auth_type.as_deref())
                .is_none()
        );

        mcp_cycle_field_option(&mut state, McpFieldId::AuthType, 1)
            .expect("cycle mcp authType option");

        assert_eq!(
            state.mcp_ui.servers[0].config.auth_type.as_deref(),
            Some("bearer")
        );
    }

    #[test]
    fn mcp_should_refresh_metadata_when_pending_or_all_unavailable() {
        assert!(mcp_should_refresh_runtime_metadata(
            true,
            false,
            false,
            "enabled, servers=2, availability=checking(async)"
        ));
        assert!(mcp_should_refresh_runtime_metadata(
            true,
            false,
            false,
            "enabled, servers=0"
        ));
        assert!(!mcp_should_refresh_runtime_metadata(
            true,
            false,
            true,
            "enabled, servers=1, tools=0"
        ));
        assert!(!mcp_should_refresh_runtime_metadata(
            false,
            false,
            false,
            "enabled, servers=0"
        ));
    }

    #[test]
    fn token_helpers_estimate_and_compact_format() {
        assert_eq!(estimate_tokens_from_text_delta(""), 0);
        assert_eq!(estimate_tokens_from_text_delta("abc"), 1);
        assert!(estimate_tokens_from_text_delta("abcdefghijkl") >= 4);
        assert_eq!(format_u64_compact(0), "0");
        assert_eq!(format_u64_compact(999), "999");
        assert_eq!(format_u64_compact(12_345_678), "12,345,678");
    }

    #[test]
    fn token_display_animation_converges_to_target() {
        let mut state = new_state();
        state.add_live_token_estimate(4096);
        for _ in 0..160 {
            state.tick_token_display_animation();
        }
        assert_eq!(state.token_display_value, state.token_usage_target());
    }

    #[test]
    fn should_append_final_thinking_only_when_not_rendered_before() {
        assert!(should_append_final_thinking("step-1 -> step-2", false));
        assert!(!should_append_final_thinking("step-1 -> step-2", true));
        assert!(!should_append_final_thinking("   ", false));
    }

    #[test]
    fn parse_builtin_command_supports_session_meta_aliases() {
        assert!(matches!(
            parse_builtin_command("/meta"),
            Some(BuiltinCommand::Meta)
        ));
        assert!(matches!(
            parse_builtin_command("/session"),
            Some(BuiltinCommand::Meta)
        ));
    }
}
