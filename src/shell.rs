use std::{
    io::{self, IsTerminal, Read, Write},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::mpsc,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use wait_timeout::ChildExt;

use crate::{config::CmdConfig, error::AppError, i18n, logging, mask::mask_sensitive};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static WRITE_SESSION_APPROVED: AtomicBool = AtomicBool::new(false);
static INTERACTIVE_INPUT_IDLE_REFRESH_HINT: AtomicBool = AtomicBool::new(false);
const DETACHED_CAPTURE_JOIN_GRACE: Duration = Duration::from_millis(250);
const INTERACTIVE_INPUT_IDLE_REFRESH_THRESHOLD: Duration = Duration::from_secs(45);

static DANGEROUS_PATTERNS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "rm -rf /",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "format c:",
        "diskpart",
        "reg delete hklm",
        ":(){:|:&};:",
    ]
});

static WRITE_HINT_PATTERNS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        " rm ",
        " mv ",
        " cp ",
        " chmod ",
        " chown ",
        " useradd ",
        " userdel ",
        " kill ",
        " sed -i",
        " tee ",
        " touch ",
        " mkdir ",
        " rmdir ",
        " del ",
        " move ",
        " ren ",
        " sc stop ",
        " net stop ",
    ]
});

static DETACHED_AMPERSAND_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(^|\s)&(\s|$)").expect("valid detached ampersand regex"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandMode {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub label: String,
    pub command: String,
    pub mode: CommandMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandResult {
    pub label: String,
    pub command: String,
    pub mode: String,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
    pub timed_out: bool,
    pub interrupted: bool,
    pub blocked: bool,
    pub block_reason: String,
}

pub struct ShellExecutor {
    timeout: Duration,
    kill_after: Duration,
    write_confirm: bool,
    confirm_mode: WriteConfirmMode,
    allow_cmd_list: Vec<String>,
    deny_cmd_list: Vec<String>,
    allow_patterns: Vec<String>,
    deny_patterns: Vec<String>,
    output_max_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteConfirmMode {
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
    EditAndApprove,
}

#[derive(Debug, Default)]
struct CaptureBuffer {
    bytes: Vec<u8>,
    truncated: bool,
}

struct CaptureThread {
    state: Arc<Mutex<CaptureBuffer>>,
    handle: thread::JoinHandle<()>,
}

impl ShellExecutor {
    pub fn new(cfg: &CmdConfig) -> Self {
        Self {
            timeout: Duration::from_secs(cfg.command_timeout_seconds),
            kill_after: Duration::from_secs(cfg.command_timeout_kill_after_seconds),
            write_confirm: cfg.write_cmd_run_confirm,
            confirm_mode: parse_confirm_mode(&cfg.write_cmd_confirm_mode),
            allow_cmd_list: cfg.allow_cmd_list.clone(),
            deny_cmd_list: cfg.deny_cmd_list.clone(),
            allow_patterns: cfg.write_cmd_allow_patterns.clone(),
            deny_patterns: cfg.write_cmd_deny_patterns.clone(),
            output_max_bytes: cfg.command_output_max_bytes,
        }
    }

    pub fn install_interrupt_handler() -> Result<(), AppError> {
        ctrlc::set_handler(|| {
            INTERRUPTED.store(true, Ordering::SeqCst);
        })
        .map_err(|err| AppError::Runtime(format!("failed to install interrupt handler: {err}")))
    }

    pub fn run_many(&self, specs: &[CommandSpec]) -> Result<Vec<CommandResult>, AppError> {
        Self::clear_interrupt_flag();
        if specs.len() <= 1 {
            let mut results = Vec::with_capacity(specs.len());
            for spec in specs {
                let result = self.run(spec)?;
                results.push(result);
            }
            return Ok(results);
        }

        let all_read = specs.iter().all(|spec| spec.mode == CommandMode::Read);
        if !all_read {
            let mut results = Vec::with_capacity(specs.len());
            for spec in specs {
                let result = self.run(spec)?;
                results.push(result);
            }
            return Ok(results);
        }

        let (tx, rx) = mpsc::channel::<(usize, Result<CommandResult, AppError>)>();
        thread::scope(|scope| {
            for (idx, spec) in specs.iter().enumerate() {
                let tx = tx.clone();
                scope.spawn(move || {
                    let _ = tx.send((idx, self.run(spec)));
                });
            }
        });
        drop(tx);

        let mut ordered = vec![None; specs.len()];
        for _ in 0..specs.len() {
            let (idx, item) = rx.recv().map_err(|err| {
                AppError::Command(format!("failed to collect parallel command result: {err}"))
            })?;
            match item {
                Ok(result) => {
                    ordered[idx] = Some(result);
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }

        let mut results = Vec::with_capacity(specs.len());
        for result in ordered.into_iter().flatten() {
            results.push(result);
        }
        Ok(results)
    }

    pub fn clear_interrupt_flag() {
        INTERRUPTED.store(false, Ordering::SeqCst);
    }

    pub fn run(&self, spec: &CommandSpec) -> Result<CommandResult, AppError> {
        self.run_with_effective_timeout(spec, self.timeout)
    }

    pub fn run_with_timeout(
        &self,
        spec: &CommandSpec,
        timeout: Duration,
    ) -> Result<CommandResult, AppError> {
        self.run_with_effective_timeout(spec, timeout)
    }

    fn run_with_effective_timeout(
        &self,
        spec: &CommandSpec,
        effective_timeout: Duration,
    ) -> Result<CommandResult, AppError> {
        let command = spec.command.trim();
        if command.is_empty() {
            return Err(AppError::Command(i18n::command_empty()));
        }

        let lowered = format!(" {} ", command.to_ascii_lowercase());
        let effective_mode =
            if spec.mode == CommandMode::Write || looks_like_write_command(&lowered) {
                CommandMode::Write
            } else {
                CommandMode::Read
            };

        if let Some(pattern) = self
            .deny_cmd_list
            .iter()
            .find(|pattern| command_matches_regex_rule(command, pattern))
        {
            let reason = i18n::command_blocked_by_deny_pattern(pattern);
            logging::warn(&format!(
                "blocked command by deny-cmd-list regex: {}",
                mask_sensitive(command)
            ));
            return Ok(CommandResult {
                label: spec.label.clone(),
                command: mask_sensitive(command),
                mode: mode_to_str(effective_mode).to_string(),
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: true,
                block_reason: reason,
            });
        }
        if !self.allow_cmd_list.is_empty()
            && !self
                .allow_cmd_list
                .iter()
                .any(|pattern| command_matches_regex_rule(command, pattern))
        {
            logging::warn(&format!(
                "blocked command by allow-cmd-list policy: {}",
                mask_sensitive(command)
            ));
            return Ok(CommandResult {
                label: spec.label.clone(),
                command: mask_sensitive(command),
                mode: mode_to_str(effective_mode).to_string(),
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: true,
                block_reason: i18n::command_blocked_by_allow_policy(),
            });
        }

        if let Some(pattern) = self.deny_patterns.iter().find(|pattern| {
            !pattern.trim().is_empty() && lowered.contains(&pattern.to_ascii_lowercase())
        }) {
            let reason = i18n::command_blocked_by_deny_pattern(pattern);
            logging::warn(&format!(
                "blocked command by deny pattern: {}",
                mask_sensitive(command)
            ));
            return Ok(CommandResult {
                label: spec.label.clone(),
                command: mask_sensitive(command),
                mode: mode_to_str(effective_mode).to_string(),
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: true,
                block_reason: reason,
            });
        }
        if let Some(pattern) = DANGEROUS_PATTERNS
            .iter()
            .find(|pattern| lowered.contains(&pattern.to_ascii_lowercase()))
        {
            let reason = i18n::dangerous_command_blocked(pattern);
            logging::warn(&format!("blocked command: {}", mask_sensitive(command)));
            return Ok(CommandResult {
                label: spec.label.clone(),
                command: mask_sensitive(command),
                mode: mode_to_str(effective_mode).to_string(),
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: true,
                block_reason: reason,
            });
        }

        if effective_mode == CommandMode::Write
            && !self.allow_patterns.is_empty()
            && !self.allow_patterns.iter().any(|pattern| {
                let value = pattern.trim();
                !value.is_empty() && lowered.contains(&value.to_ascii_lowercase())
            })
        {
            return Ok(CommandResult {
                label: spec.label.clone(),
                command: mask_sensitive(command),
                mode: mode_to_str(effective_mode).to_string(),
                success: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration_ms: 0,
                timed_out: false,
                interrupted: false,
                blocked: true,
                block_reason: i18n::command_blocked_by_allow_policy(),
            });
        }

        if effective_mode == CommandMode::Write && self.write_confirm {
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                return Err(AppError::Command(
                    i18n::command_write_confirm_non_interactive(),
                ));
            }
            let mut effective_command = command.to_string();
            let decision = resolve_write_decision(self.confirm_mode, &effective_command)?;
            match decision {
                WriteDecision::Reject => {
                    return Ok(CommandResult {
                        label: spec.label.clone(),
                        command: mask_sensitive(command),
                        mode: mode_to_str(effective_mode).to_string(),
                        success: false,
                        exit_code: None,
                        stdout: String::new(),
                        stderr: String::new(),
                        duration_ms: 0,
                        timed_out: false,
                        interrupted: false,
                        blocked: true,
                        block_reason: i18n::command_write_denied_by_user(),
                    });
                }
                WriteDecision::Approve => {}
                WriteDecision::ApproveSession => {
                    WRITE_SESSION_APPROVED.store(true, Ordering::SeqCst);
                }
                WriteDecision::EditAndApprove => {
                    let edited = prompt_edit_command(&effective_command)?;
                    if edited.trim().is_empty() {
                        return Ok(CommandResult {
                            label: spec.label.clone(),
                            command: mask_sensitive(command),
                            mode: mode_to_str(effective_mode).to_string(),
                            success: false,
                            exit_code: None,
                            stdout: String::new(),
                            stderr: String::new(),
                            duration_ms: 0,
                            timed_out: false,
                            interrupted: false,
                            blocked: true,
                            block_reason: i18n::command_write_denied_by_user(),
                        });
                    }
                    effective_command = edited;
                }
            }

            let lowered_effective = format!(" {} ", effective_command.to_ascii_lowercase());
            if let Some(pattern) = self
                .deny_cmd_list
                .iter()
                .find(|pattern| command_matches_regex_rule(&effective_command, pattern))
            {
                let reason = i18n::command_blocked_by_deny_pattern(pattern);
                logging::warn(&format!(
                    "blocked command by deny-cmd-list regex: {}",
                    mask_sensitive(&effective_command)
                ));
                return Ok(CommandResult {
                    label: spec.label.clone(),
                    command: mask_sensitive(&effective_command),
                    mode: mode_to_str(effective_mode).to_string(),
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: true,
                    block_reason: reason,
                });
            }
            if !self.allow_cmd_list.is_empty()
                && !self
                    .allow_cmd_list
                    .iter()
                    .any(|pattern| command_matches_regex_rule(&effective_command, pattern))
            {
                logging::warn(&format!(
                    "blocked command by allow-cmd-list policy: {}",
                    mask_sensitive(&effective_command)
                ));
                return Ok(CommandResult {
                    label: spec.label.clone(),
                    command: mask_sensitive(&effective_command),
                    mode: mode_to_str(effective_mode).to_string(),
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: true,
                    block_reason: i18n::command_blocked_by_allow_policy(),
                });
            }
            if let Some(pattern) = self.deny_patterns.iter().find(|pattern| {
                !pattern.trim().is_empty()
                    && lowered_effective.contains(&pattern.to_ascii_lowercase())
            }) {
                return Ok(CommandResult {
                    label: spec.label.clone(),
                    command: mask_sensitive(&effective_command),
                    mode: mode_to_str(effective_mode).to_string(),
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: true,
                    block_reason: i18n::command_blocked_by_deny_pattern(pattern),
                });
            }
            if let Some(pattern) = DANGEROUS_PATTERNS
                .iter()
                .find(|pattern| lowered_effective.contains(&pattern.to_ascii_lowercase()))
            {
                return Ok(CommandResult {
                    label: spec.label.clone(),
                    command: mask_sensitive(&effective_command),
                    mode: mode_to_str(effective_mode).to_string(),
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: true,
                    block_reason: i18n::dangerous_command_blocked(pattern),
                });
            }
            if !self.allow_patterns.is_empty()
                && !self.allow_patterns.iter().any(|pattern| {
                    let value = pattern.trim();
                    !value.is_empty() && lowered_effective.contains(&value.to_ascii_lowercase())
                })
            {
                return Ok(CommandResult {
                    label: spec.label.clone(),
                    command: mask_sensitive(&effective_command),
                    mode: mode_to_str(effective_mode).to_string(),
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 0,
                    timed_out: false,
                    interrupted: false,
                    blocked: true,
                    block_reason: i18n::command_blocked_by_allow_policy(),
                });
            }
            return self.run_inner(spec, &effective_command, effective_mode, effective_timeout);
        }
        self.run_inner(spec, command, effective_mode, effective_timeout)
    }

    fn run_inner(
        &self,
        spec: &CommandSpec,
        command: &str,
        effective_mode: CommandMode,
        effective_timeout: Duration,
    ) -> Result<CommandResult, AppError> {
        logging::info(&format!(
            "command start: label={}, cmd={}",
            spec.label,
            mask_sensitive(command)
        ));
        let started = Instant::now();
        let detached_command = looks_like_detached_command(command);
        let mut child = shell_command(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| {
                AppError::Command(format!("failed to spawn command [{}]: {err}", spec.label))
            })?;

        let stdout_pipe = child.stdout.take().ok_or_else(|| {
            AppError::Command(format!(
                "failed to open stdout pipe for command [{}]",
                spec.label
            ))
        })?;
        let stderr_pipe = child.stderr.take().ok_or_else(|| {
            AppError::Command(format!(
                "failed to open stderr pipe for command [{}]",
                spec.label
            ))
        })?;
        let stdout_handle = spawn_capture_thread(stdout_pipe, self.output_max_bytes);
        let stderr_handle = spawn_capture_thread(stderr_pipe, self.output_max_bytes);

        let mut timed_out = false;
        let mut interrupted = false;

        loop {
            if INTERRUPTED.load(Ordering::SeqCst) {
                interrupted = true;
                let _ = child.kill();
                break;
            }

            let elapsed = started.elapsed();
            if elapsed >= effective_timeout {
                timed_out = true;
                let _ = child.kill();
                break;
            }

            let wait_slice = std::cmp::min(effective_timeout - elapsed, Duration::from_millis(200));
            let status = child.wait_timeout(wait_slice).map_err(|err| {
                AppError::Command(format!("failed to wait command [{}]: {err}", spec.label))
            })?;
            if status.is_some() {
                break;
            }
        }

        if timed_out {
            let _ = child.wait_timeout(self.kill_after);
        }

        let status = child.wait().map_err(|err| {
            AppError::Command(format!(
                "failed to collect command output [{}]: {err}",
                spec.label
            ))
        })?;
        let (stdout, stdout_truncated, stdout_incomplete) =
            collect_capture_output(stdout_handle, &spec.label, "stdout", detached_command)?;
        let (stderr, stderr_truncated, stderr_incomplete) =
            collect_capture_output(stderr_handle, &spec.label, "stderr", detached_command)?;
        let duration_ms = started.elapsed().as_millis();
        let stdout = finalize_captured_output(stdout, stdout_truncated, stdout_incomplete);
        let stderr = finalize_captured_output(stderr, stderr_truncated, stderr_incomplete);
        let success = status.success() && !timed_out && !interrupted;

        logging::info(&format!(
            "command end: label={}, success={}, exit_code={:?}, duration_ms={duration_ms}",
            spec.label,
            success,
            status.code()
        ));

        Ok(CommandResult {
            label: spec.label.clone(),
            command: mask_sensitive(command),
            mode: mode_to_str(effective_mode).to_string(),
            success,
            exit_code: status.code(),
            stdout,
            stderr,
            duration_ms,
            timed_out,
            interrupted,
            blocked: false,
            block_reason: String::new(),
        })
    }
}

pub fn note_interactive_input_wait(wait_started: Instant) {
    if wait_started.elapsed() >= INTERACTIVE_INPUT_IDLE_REFRESH_THRESHOLD {
        INTERACTIVE_INPUT_IDLE_REFRESH_HINT.store(true, Ordering::SeqCst);
    }
}

pub fn take_interactive_input_refresh_hint() -> bool {
    INTERACTIVE_INPUT_IDLE_REFRESH_HINT.swap(false, Ordering::SeqCst)
}

fn shell_command(command: &str) -> Command {
    if cfg!(windows) {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", command]);
        cmd
    } else {
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", command]);
        cmd
    }
}

fn command_matches_regex_rule(command: &str, pattern: &str) -> bool {
    let trimmed = pattern.trim();
    if trimmed.is_empty() {
        return false;
    }
    match Regex::new(trimmed) {
        Ok(regex) => regex.is_match(command),
        Err(err) => {
            logging::warn(&format!(
                "invalid command regex pattern ignored: pattern={}, err={}",
                mask_sensitive(trimmed),
                err
            ));
            false
        }
    }
}

fn parse_confirm_mode(raw: &str) -> WriteConfirmMode {
    match raw.trim().to_ascii_lowercase().as_str() {
        "deny" => WriteConfirmMode::Deny,
        "edit" => WriteConfirmMode::Edit,
        "allow-session" => WriteConfirmMode::AllowSession,
        _ => WriteConfirmMode::AllowOnce,
    }
}

fn resolve_write_decision(
    mode: WriteConfirmMode,
    command: &str,
) -> Result<WriteDecision, AppError> {
    match mode {
        WriteConfirmMode::Deny => Ok(WriteDecision::Reject),
        WriteConfirmMode::Edit => Ok(WriteDecision::EditAndApprove),
        WriteConfirmMode::AllowSession => {
            if WRITE_SESSION_APPROVED.load(Ordering::SeqCst) {
                return Ok(WriteDecision::Approve);
            }
            prompt_write_decision(command, true)
        }
        WriteConfirmMode::AllowOnce => prompt_write_decision(command, false),
    }
}

fn prompt_write_decision(
    command: &str,
    with_session_allow: bool,
) -> Result<WriteDecision, AppError> {
    let confirm_title = i18n::prompt_write_command_confirmation(&mask_sensitive(command));
    let confirm_question = if with_session_allow {
        i18n::prompt_write_command_proceed_with_session()
    } else {
        i18n::prompt_write_command_proceed()
    };
    println!("{confirm_title}");
    loop {
        print!("{confirm_question}");
        io::stdout().flush().map_err(|err| {
            AppError::Command(format!("failed to flush confirmation prompt: {err}"))
        })?;
        let mut input = String::new();
        let wait_started = Instant::now();
        io::stdin().read_line(&mut input).map_err(|err| {
            AppError::Command(format!("failed to read confirmation input: {err}"))
        })?;
        note_interactive_input_wait(wait_started);
        if let Some(decision) = parse_write_decision_input(&input, with_session_allow) {
            return Ok(decision);
        }
        println!("{}", i18n::prompt_write_command_invalid_input());
    }
}

fn parse_write_decision_input(input: &str, with_session_allow: bool) -> Option<WriteDecision> {
    let normalized = input.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "n" || normalized == "no" {
        return Some(WriteDecision::Reject);
    }
    if normalized == "y" || normalized == "yes" {
        return Some(WriteDecision::Approve);
    }
    if with_session_allow && (normalized == "a" || normalized == "all") {
        return Some(WriteDecision::ApproveSession);
    }
    if normalized == "e" || normalized == "edit" {
        return Some(WriteDecision::EditAndApprove);
    }
    None
}

fn prompt_edit_command(original: &str) -> Result<String, AppError> {
    print!(
        "{}\n{}\n{}",
        i18n::prompt_write_command_edit_title(&mask_sensitive(original)),
        i18n::prompt_write_command_edit_hint(),
        i18n::prompt_write_command_edit_input()
    );
    io::stdout()
        .flush()
        .map_err(|err| AppError::Command(format!("failed to flush edit prompt: {err}")))?;
    let mut input = String::new();
    let wait_started = Instant::now();
    io::stdin()
        .read_line(&mut input)
        .map_err(|err| AppError::Command(format!("failed to read edited command: {err}")))?;
    note_interactive_input_wait(wait_started);
    Ok(input.trim().to_string())
}

fn spawn_capture_thread<T>(mut reader: T, cap: usize) -> CaptureThread
where
    T: Read + Send + 'static,
{
    let state = Arc::new(Mutex::new(CaptureBuffer::default()));
    let thread_state = Arc::clone(&state);
    let handle = thread::spawn(move || {
        let mut tmp = [0_u8; 4096];
        loop {
            match reader.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    let mut guard = match thread_state.lock() {
                        Ok(guard) => guard,
                        Err(_) => break,
                    };
                    let remaining = cap.saturating_sub(guard.bytes.len());
                    if remaining > 0 {
                        let take = std::cmp::min(remaining, n);
                        guard.bytes.extend_from_slice(&tmp[..take]);
                    }
                    if n > remaining || guard.bytes.len() >= cap {
                        guard.truncated = true;
                    }
                }
                Err(_) => break,
            }
        }
    });
    CaptureThread { state, handle }
}

fn collect_capture_output(
    capture: CaptureThread,
    label: &str,
    stream_name: &str,
    detached_command: bool,
) -> Result<(Vec<u8>, bool, bool), AppError> {
    if detached_command {
        let deadline = Instant::now() + DETACHED_CAPTURE_JOIN_GRACE;
        while !capture.handle.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }

    let incomplete = detached_command && !capture.handle.is_finished();
    if !incomplete {
        capture.handle.join().map_err(|_| {
            AppError::Command(format!(
                "failed to join {stream_name} capture thread for command [{label}]",
            ))
        })?;
    } else {
        logging::warn(&format!(
            "capture thread still active after command exit, command={}, stream={stream_name}",
            mask_sensitive(label)
        ));
    }

    let snapshot = capture.state.lock().map_err(|_| {
        AppError::Command(format!(
            "failed to snapshot {stream_name} capture buffer for command [{label}]",
        ))
    })?;
    Ok((snapshot.bytes.clone(), snapshot.truncated, incomplete))
}

fn finalize_captured_output(bytes: Vec<u8>, truncated: bool, incomplete: bool) -> String {
    let mut text = String::from_utf8_lossy(&bytes).to_string();
    if truncated {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("...[output truncated]");
    }
    if incomplete {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("...[output capture incomplete: detached process still holds pipe]");
    }
    text
}

fn looks_like_write_command(command_with_padding: &str) -> bool {
    WRITE_HINT_PATTERNS
        .iter()
        .any(|item| command_with_padding.contains(&item.to_ascii_lowercase()))
}

fn looks_like_detached_command(command: &str) -> bool {
    let lowered = command.trim().to_ascii_lowercase();
    lowered.contains("nohup ")
        || lowered.contains(" disown")
        || lowered.contains("setsid ")
        || lowered.contains("start /b ")
        || DETACHED_AMPERSAND_RE.is_match(&lowered)
}

fn mode_to_str(mode: CommandMode) -> &'static str {
    match mode {
        CommandMode::Read => "read",
        CommandMode::Write => "write",
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::config::CmdConfig;

    use super::{
        CommandMode, CommandSpec, ShellExecutor, WriteDecision, looks_like_detached_command,
        note_interactive_input_wait, parse_write_decision_input,
        take_interactive_input_refresh_hint,
    };

    #[test]
    fn detects_detached_background_commands_without_false_positive_on_redirects() {
        assert!(looks_like_detached_command(
            r#"cd "/tmp" && nohup npm run dev > /tmp/app.log 2>&1 & echo $!"#
        ));
        assert!(looks_like_detached_command("sleep 5 & echo ready"));
        assert!(!looks_like_detached_command("echo 2>&1"));
        assert!(!looks_like_detached_command("sleep 1 && echo ready"));
    }

    #[cfg(not(windows))]
    #[test]
    fn background_command_returns_without_waiting_for_pipe_eof() {
        let executor = ShellExecutor::new(&CmdConfig::default());
        let result = executor
            .run_with_timeout(
                &CommandSpec {
                    label: "background".to_string(),
                    command: "sleep 2 & echo ready".to_string(),
                    mode: CommandMode::Read,
                },
                Duration::from_secs(5),
            )
            .expect("background command should complete");

        assert!(result.success);
        assert!(
            result.duration_ms < 1_500,
            "duration_ms={}",
            result.duration_ms
        );
        assert!(result.stdout.contains("ready"));
    }

    #[test]
    fn parse_write_confirmation_accepts_case_insensitive_yes_no_and_rejects_invalid() {
        assert_eq!(
            parse_write_decision_input("Y\n", false),
            Some(WriteDecision::Approve)
        );
        assert_eq!(
            parse_write_decision_input("n\n", false),
            Some(WriteDecision::Reject)
        );
        assert_eq!(
            parse_write_decision_input("\n", false),
            Some(WriteDecision::Reject)
        );
        assert_eq!(parse_write_decision_input("m\n", false), None);
    }

    #[test]
    fn long_interactive_wait_sets_refresh_hint_once() {
        let _ = take_interactive_input_refresh_hint();
        note_interactive_input_wait(Instant::now() - Duration::from_secs(46));
        assert!(take_interactive_input_refresh_hint());
        assert!(!take_interactive_input_refresh_hint());
    }

    #[test]
    fn short_interactive_wait_does_not_set_refresh_hint() {
        let _ = take_interactive_input_refresh_hint();
        note_interactive_input_wait(Instant::now());
        assert!(!take_interactive_input_refresh_hint());
    }
}
