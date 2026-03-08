use std::{
    io::{self, IsTerminal, Read, Write},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use serde::Serialize;
use wait_timeout::ChildExt;

use crate::{config::CmdConfig, error::AppError, i18n, logging, mask::mask_sensitive};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static WRITE_SESSION_APPROVED: AtomicBool = AtomicBool::new(false);

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

impl ShellExecutor {
    pub fn new(cfg: &CmdConfig) -> Self {
        Self {
            timeout: Duration::from_secs(cfg.command_timeout_seconds),
            kill_after: Duration::from_secs(cfg.command_timeout_kill_after_seconds),
            write_confirm: cfg.write_cmd_run_confirm,
            confirm_mode: parse_confirm_mode(&cfg.write_cmd_confirm_mode),
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
        for item in ordered {
            if let Some(result) = item {
                results.push(result);
            }
        }
        Ok(results)
    }

    pub fn clear_interrupt_flag() {
        INTERRUPTED.store(false, Ordering::SeqCst);
    }

    pub fn run(&self, spec: &CommandSpec) -> Result<CommandResult, AppError> {
        let command = spec.command.trim();
        if command.is_empty() {
            return Err(AppError::Command(i18n::command_empty()));
        }

        let lowered = format!(" {} ", command.to_ascii_lowercase());
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
                mode: mode_to_str(spec.mode).to_string(),
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
                mode: mode_to_str(spec.mode).to_string(),
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

        let effective_mode =
            if spec.mode == CommandMode::Write || looks_like_write_command(&lowered) {
                CommandMode::Write
            } else {
                CommandMode::Read
            };

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
            return self.run_inner(spec, &effective_command, effective_mode);
        }
        self.run_inner(spec, command, effective_mode)
    }

    fn run_inner(
        &self,
        spec: &CommandSpec,
        command: &str,
        effective_mode: CommandMode,
    ) -> Result<CommandResult, AppError> {
        logging::info(&format!(
            "command start: label={}, cmd={}",
            spec.label,
            mask_sensitive(command)
        ));
        let started = Instant::now();
        let mut child = shell_command(command)
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
            if elapsed >= self.timeout {
                timed_out = true;
                let _ = child.kill();
                break;
            }

            let wait_slice = std::cmp::min(self.timeout - elapsed, Duration::from_millis(200));
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
        let (stdout, stdout_truncated) = stdout_handle.join().map_err(|_| {
            AppError::Command(format!(
                "failed to join stdout capture thread for command [{}]",
                spec.label
            ))
        })?;
        let (stderr, stderr_truncated) = stderr_handle.join().map_err(|_| {
            AppError::Command(format!(
                "failed to join stderr capture thread for command [{}]",
                spec.label
            ))
        })?;
        let duration_ms = started.elapsed().as_millis();
        let stdout = finalize_captured_output(stdout, stdout_truncated);
        let stderr = finalize_captured_output(stderr, stderr_truncated);
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
    print!("{}\n{}", confirm_title, confirm_question);
    io::stdout()
        .flush()
        .map_err(|err| AppError::Command(format!("failed to flush confirmation prompt: {err}")))?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|err| AppError::Command(format!("failed to read confirmation input: {err}")))?;
    let normalized = input.trim().to_ascii_lowercase();
    if normalized == "e" || normalized == "edit" {
        return Ok(WriteDecision::EditAndApprove);
    }
    if with_session_allow && (normalized == "a" || normalized == "all") {
        return Ok(WriteDecision::ApproveSession);
    }
    if normalized == "y" || normalized == "yes" {
        return Ok(WriteDecision::Approve);
    }
    Ok(WriteDecision::Reject)
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
    io::stdin()
        .read_line(&mut input)
        .map_err(|err| AppError::Command(format!("failed to read edited command: {err}")))?;
    Ok(input.trim().to_string())
}

fn spawn_capture_thread<T>(mut reader: T, cap: usize) -> thread::JoinHandle<(Vec<u8>, bool)>
where
    T: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buf = Vec::<u8>::new();
        let mut tmp = [0_u8; 4096];
        let mut truncated = false;
        loop {
            match reader.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    let remaining = cap.saturating_sub(buf.len());
                    if remaining > 0 {
                        let take = std::cmp::min(remaining, n);
                        buf.extend_from_slice(&tmp[..take]);
                    }
                    if buf.len() >= cap {
                        truncated = true;
                    }
                }
                Err(_) => break,
            }
        }
        (buf, truncated)
    })
}

fn finalize_captured_output(bytes: Vec<u8>, truncated: bool) -> String {
    let mut text = String::from_utf8_lossy(&bytes).to_string();
    if truncated {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("...[output truncated]");
    }
    text
}

fn looks_like_write_command(command_with_padding: &str) -> bool {
    WRITE_HINT_PATTERNS
        .iter()
        .any(|item| command_with_padding.contains(&item.to_ascii_lowercase()))
}

fn mode_to_str(mode: CommandMode) -> &'static str {
    match mode {
        CommandMode::Read => "read",
        CommandMode::Write => "write",
    }
}
