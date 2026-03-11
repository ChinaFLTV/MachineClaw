use std::{
    cell::RefCell,
    collections::HashMap,
    fs,
    io::{self, BufRead, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use chrono::{Local, TimeZone};
use dialoguer::{MultiSelect, theme::ColorfulTheme};
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::{
    ai::{
        AiClient, ChatRoundEvent, ChatStreamEvent, ChatStreamEventKind, ExternalToolDefinition,
        ModelPriceCheckResult, ModelPriceSource, ToolCallRequest, ToolUsePolicy,
        is_transient_ai_error,
    },
    cli::InspectTarget,
    config::{AppConfig, expand_tilde, normalize_chat_model_price_check_mode},
    context::{SessionMessage, SessionStore},
    error::{AppError, ExitCode},
    i18n, logging,
    mask::mask_sensitive,
    mcp::McpManager,
    platform::OsType,
    render::{self, ActionRenderData},
    shell::{CommandMode, CommandResult, CommandSpec, ShellExecutor, note_interactive_input_wait},
};

pub struct ActionServices<'a> {
    pub cfg: &'a AppConfig,
    pub assets_dir: &'a Path,
    pub shell: &'a ShellExecutor,
    pub ai: &'a AiClient,
    pub session: &'a mut SessionStore,
    pub os_type: OsType,
    pub os_name: &'a str,
    pub skills: &'a [String],
    pub mcp_summary: String,
    pub mcp: &'a mut McpManager,
}

pub struct ActionOutcome {
    pub rendered: String,
    pub exit_code: ExitCode,
}

#[derive(Debug, Default)]
struct ChatToolStats {
    tool_calls: usize,
    command_failures: usize,
    blocked_count: usize,
    timeout_count: usize,
    interrupted_count: usize,
    tool_duration_ms: u128,
    cache_hits: usize,
}

#[derive(Debug, Clone)]
struct ToolCallCacheItem {
    created_at_epoch_ms: u128,
    payload: String,
}

pub fn run_prepare(services: &mut ActionServices<'_>) -> Result<ActionOutcome, AppError> {
    let started = Instant::now();
    let group_id = Uuid::new_v4().to_string();
    services
        .session
        .add_user_message("action=prepare".to_string(), Some(group_id.clone()));

    let commands = prepare_commands(services.os_type);
    let results = services.shell.run_many(&commands)?;

    let command_summary = format_prepare_command_summary(&results);
    services.session.add_tool_message(
        format!(
            "prepare command results:\n{}",
            trim_text(&command_summary, 3000)
        ),
        Some(group_id.clone()),
    );
    services.session.persist()?;

    let success = !has_command_failures(&results);
    let status = if success {
        i18n::status_success()
    } else {
        i18n::status_failed()
    };
    let key_metrics = format_prepare_metrics(services, &results);
    let risk_summary = format_prepare_risk_summary(&results);

    let ai_summary = generate_ai_summary(
        services,
        "prepare_user.md",
        "prepare",
        "prepare",
        &key_metrics,
        &results,
    )?;

    services
        .session
        .add_assistant_message(ai_summary.clone(), Some(group_id));
    services.session.persist()?;

    let rendered = render::render_action(
        services.assets_dir,
        "prepare",
        &ActionRenderData {
            action: "prepare".to_string(),
            status: status.to_string(),
            key_metrics,
            risk_summary,
            ai_summary,
            command_summary,
            elapsed: i18n::human_duration_ms(started.elapsed().as_millis()),
        },
        services.cfg.console.colorful,
    )?;

    let exit_code = if success {
        ExitCode::Success
    } else {
        ExitCode::CommandFailure
    };

    Ok(ActionOutcome {
        rendered,
        exit_code,
    })
}

pub fn run_inspect(
    services: &mut ActionServices<'_>,
    target: InspectTarget,
) -> Result<ActionOutcome, AppError> {
    let started = Instant::now();
    let group_id = Uuid::new_v4().to_string();
    services.session.add_user_message(
        format!("action=inspect target={}", target.as_str()),
        Some(group_id.clone()),
    );

    let commands = inspect_commands(services.os_type, target);
    if commands.is_empty() {
        return Err(AppError::Runtime(format!(
            "no inspect commands found for target={} on os={}",
            target.as_str(),
            services.os_name
        )));
    }

    let results = services.shell.run_many(&commands)?;
    let command_summary = format_command_summary(&results);
    services.session.add_tool_message(
        format!(
            "inspect target={} command results:\n{}",
            target.as_str(),
            trim_text(&command_summary, 3000)
        ),
        Some(group_id.clone()),
    );
    services.session.persist()?;

    let success = !has_command_failures(&results);
    let status = if success {
        i18n::status_success()
    } else {
        i18n::status_failed()
    };
    let key_metrics = format_inspect_metrics(services, target, &results);
    let risk_summary = build_risk_summary(&results);

    let ai_summary = generate_ai_summary(
        services,
        "inspect_user.md",
        "inspect",
        target.as_str(),
        &key_metrics,
        &results,
    )?;

    services
        .session
        .add_assistant_message(ai_summary.clone(), Some(group_id));
    services.session.persist()?;

    let rendered = render::render_action(
        services.assets_dir,
        "inspect",
        &ActionRenderData {
            action: format!("inspect {}", target.as_str()),
            status: status.to_string(),
            key_metrics,
            risk_summary,
            ai_summary,
            command_summary,
            elapsed: i18n::human_duration_ms(started.elapsed().as_millis()),
        },
        services.cfg.console.colorful,
    )?;

    let exit_code = if success {
        ExitCode::Success
    } else {
        ExitCode::CommandFailure
    };

    Ok(ActionOutcome {
        rendered,
        exit_code,
    })
}

pub fn run_chat(services: &mut ActionServices<'_>) -> Result<ActionOutcome, AppError> {
    let started = Instant::now();
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(AppError::Command(i18n::chat_requires_interactive_terminal()));
    }

    let base_system_prompt = render::load_prompt_template(services.assets_dir, "chat_system.md")?;
    let system_prompt = build_chat_system_prompt(services, &base_system_prompt);
    maybe_ensure_chat_environment_profile(services, &system_prompt)?;
    maybe_prepare_chat_model_price(services);
    let initial_message_count = services.session.message_count();
    let mut chat_turns: usize = 0;
    let mut tool_stats = ChatToolStats::default();
    let mut last_assistant_reply = String::new();
    let mut pending_message: Option<String> = None;
    let mut tool_call_cache = HashMap::<String, ToolCallCacheItem>::new();
    let command_cache_ttl_ms = services.cfg.ai.chat.command_cache_ttl_seconds as u128 * 1000;
    let external_mcp_tools: Vec<ExternalToolDefinition> = services.mcp.external_tool_definitions();
    let mut autosave_worker = ChatAutosaveWorker::start(services.session, Duration::from_secs(3));
    render_chat_window_header(services, false);

    loop {
        print!(
            "{}",
            render::render_chat_user_prompt(
                i18n::chat_prompt_user(),
                services.cfg.console.colorful
            )
        );
        io::stdout()
            .flush()
            .map_err(|err| AppError::Command(format!("failed to flush chat prompt: {err}")))?;

        let message_raw = if let Some(next_message) = pending_message.take() {
            println!("{next_message}");
            next_message
        } else {
            let mut input = Vec::<u8>::new();
            let wait_started = Instant::now();
            let read_count = {
                let stdin = io::stdin();
                let mut stdin_lock = stdin.lock();
                stdin_lock
                    .read_until(b'\n', &mut input)
                    .map_err(|err| AppError::Command(format!("failed to read chat input: {err}")))?
            };
            note_interactive_input_wait(wait_started);
            if read_count == 0 {
                break;
            }
            while matches!(input.last(), Some(b'\n' | b'\r')) {
                input.pop();
            }
            String::from_utf8_lossy(&input).trim().to_string()
        };
        let message = normalize_builtin_command_alias(normalize_chat_message(&message_raw));
        if message.is_empty() {
            continue;
        }
        if is_noise_message(&message) {
            continue;
        }

        if let Some(command) = parse_builtin_command(&message) {
            match command.name.as_str() {
                "exit" | "quit" => break,
                "help" if command.arg.is_empty() => {
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            i18n::chat_help_text(),
                            services.cfg.console.colorful
                        )
                    );
                }
                "stats" if command.arg.is_empty() => {
                    let archived_counts = services.session.archived_role_counts();
                    let effective_counts = services.session.effective_context_role_counts(true);
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            &i18n::chat_stats(
                                services.session.session_id(),
                                services.session.file_path(),
                                archived_counts.total,
                                effective_counts.total,
                                services.session.summary_len(),
                                services.cfg.session.recent_messages,
                                services.cfg.session.max_messages,
                                chat_turns,
                                services.os_name,
                                &services.cfg.ai.model,
                                services.skills.len(),
                                services.mcp_summary.as_str(),
                                archived_counts.user,
                                archived_counts.assistant,
                                archived_counts.tool,
                                archived_counts.system,
                                effective_counts.user,
                                effective_counts.assistant,
                                effective_counts.tool,
                                effective_counts.system,
                            ),
                            services.cfg.console.colorful
                        )
                    );
                }
                "list" if command.arg.is_empty() => {
                    let sessions = services.session.list_sessions()?;
                    let markdown = format_chat_session_list_markdown(&sessions);
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            &markdown,
                            services.cfg.console.colorful
                        )
                    );
                }
                "change" => {
                    if command.arg.is_empty() {
                        println!(
                            "{}",
                            render::render_chat_notice(
                                i18n::chat_change_usage(),
                                services.cfg.console.colorful
                            )
                        );
                        continue;
                    }
                    match services.session.switch_session_by_query(&command.arg) {
                        Ok(switched) => {
                            pending_message = None;
                            last_assistant_reply.clear();
                            autosave_worker.submit_from_session(services.session);
                            maybe_ensure_chat_environment_profile(services, &system_prompt)?;
                            autosave_worker.submit_from_session(services.session);
                            println!(
                                "{}",
                                render::render_markdown_for_terminal(
                                    &i18n::chat_session_changed(
                                        &switched.session_name,
                                        &switched.session_id,
                                        &switched.file_path
                                    ),
                                    services.cfg.console.colorful
                                )
                            );
                        }
                        Err(err) => {
                            println!(
                                "{}",
                                render::render_chat_notice(
                                    &i18n::localize_error(&err),
                                    services.cfg.console.colorful
                                )
                            );
                        }
                    }
                }
                "name" => {
                    if command.arg.is_empty() {
                        println!(
                            "{}",
                            render::render_chat_notice(
                                i18n::chat_name_usage(),
                                services.cfg.console.colorful
                            )
                        );
                        continue;
                    }
                    services.session.rename_current_session(&command.arg)?;
                    autosave_worker.submit_from_session(services.session);
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            &i18n::chat_session_renamed(
                                services.session.session_name(),
                                services.session.session_id()
                            ),
                            services.cfg.console.colorful
                        )
                    );
                }
                "new" if command.arg.is_empty() => {
                    services.session.start_new_session_with_new_file()?;
                    pending_message = None;
                    last_assistant_reply.clear();
                    autosave_worker.submit_from_session(services.session);
                    maybe_ensure_chat_environment_profile(services, &system_prompt)?;
                    autosave_worker.submit_from_session(services.session);
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            &i18n::chat_session_switched(
                                services.session.session_id(),
                                services.session.file_path()
                            ),
                            services.cfg.console.colorful
                        )
                    );
                }
                "clear" if command.arg.is_empty() => {
                    clear_terminal()?;
                    render_chat_window_header(services, true);
                }
                "history" => {
                    let requested = match parse_chat_history_limit(&command.arg) {
                        Ok(value) => value,
                        Err(text) => {
                            println!(
                                "{}",
                                render::render_chat_notice(text, services.cfg.console.colorful)
                            );
                            continue;
                        }
                    };
                    let history_items = services.session.recent_messages_for_display(requested);
                    let markdown = format_chat_history_markdown(
                        &history_items,
                        requested,
                        services.session.message_count(),
                    );
                    println!(
                        "{}",
                        render::render_markdown_for_terminal(
                            &markdown,
                            services.cfg.console.colorful
                        )
                    );
                }
                other => {
                    println!(
                        "{}",
                        render::render_chat_notice(
                            &i18n::chat_unknown_builtin_command(other),
                            services.cfg.console.colorful,
                        )
                    );
                }
            }
            continue;
        }

        let group_id = Uuid::new_v4().to_string();
        chat_turns += 1;
        services
            .session
            .add_user_message(message.clone(), Some(group_id.clone()));
        autosave_worker.submit_from_session(services.session);
        if services.cfg.ai.chat.show_tips
            && let Some(warn_message) = services.session.context_pressure_warning(
                services.cfg.ai.chat.context_warn_percent,
                services.cfg.ai.chat.context_critical_percent,
            )
        {
            println!(
                "{}",
                render::render_chat_notice(&warn_message, services.cfg.console.colorful)
            );
        }
        maybe_run_ai_context_compression(services, &system_prompt)?;
        services.session.persist()?;
        autosave_worker.submit_from_session(services.session);
        if services.cfg.skills.enabled && !services.skills.is_empty() {
            println!(
                "{}",
                render::render_chat_custom_tag_event(
                    i18n::chat_tag_skill(),
                    &i18n::chat_skill_prepare_started(services.skills.len()),
                    services.cfg.console.colorful
                )
            );
        }

        logging::info("AI chat start");
        let history = services.session.build_chat_history();
        let policy = if should_require_tool_call(&message) {
            ToolUsePolicy::RequireAtLeastOne
        } else {
            ToolUsePolicy::Auto
        };
        let colorful = services.cfg.console.colorful;
        let stream_output = services.cfg.ai.chat.stream_output;
        let show_tips = services.cfg.ai.chat.show_tips;
        let mut printed_round_thinking = false;
        let mut last_round_thinking = String::new();
        let mut last_round_content = String::new();
        let stream_printer = RefCell::new(render::ChatStreamPrinter::new(colorful));
        let ai_wait_spinner = RefCell::new((!services.cfg.ai.debug).then(|| {
            ActivitySpinner::start(i18n::chat_progress_analyzing().to_string(), colorful)
        }));
        let spinner_first_output = Arc::new(AtomicBool::new(false));
        let spinner_first_output_cloned = spinner_first_output.clone();
        let response = match services.ai.chat_with_shell_tool(
            &history,
            &system_prompt,
            &message,
            policy,
            services.cfg.ai.chat.max_tool_rounds,
            services.cfg.ai.chat.max_total_tool_calls,
            stream_output,
            &external_mcp_tools,
            |tool_call| {
                execute_tool_call(
                    services,
                    &group_id,
                    tool_call,
                    &mut tool_stats,
                    &mut tool_call_cache,
                    command_cache_ttl_ms,
                )
            },
            |event: ChatRoundEvent| {
                stop_activity_spinner_once(&ai_wait_spinner, &spinner_first_output_cloned);
                if show_tips && event.has_tool_calls {
                    let _ = stream_printer.borrow_mut().finish();
                    println!(
                        "{}",
                        render::render_chat_notice(
                            &i18n::chat_round_received(event.round, event.tool_call_count),
                            colorful
                        )
                    );
                }
                if let Some(thinking) = event.thinking.as_deref()
                    && !thinking.trim().is_empty()
                {
                    printed_round_thinking = true;
                    last_round_thinking = thinking.trim().to_string();
                    if !event.streamed_thinking {
                        let _ = stream_printer.borrow_mut().finish();
                        println!(
                            "{}",
                            render::render_chat_thinking(
                                i18n::chat_prompt_thinking(),
                                thinking,
                                colorful
                            )
                        );
                    }
                }
                if !event.content.trim().is_empty() {
                    last_round_content = event.content.trim().to_string();
                    if !event.streamed_content {
                        let _ = stream_printer.borrow_mut().finish();
                        println!(
                            "{}",
                            render::render_chat_assistant_reply(
                                i18n::chat_prompt_assistant(),
                                event.content.trim(),
                                colorful
                            )
                        );
                    }
                }
            },
            |event: ChatStreamEvent| {
                stop_activity_spinner_once(&ai_wait_spinner, &spinner_first_output_cloned);
                let result = match event.kind {
                    ChatStreamEventKind::Content => stream_printer.borrow_mut().write(
                        render::ChatStreamBlockKind::Assistant,
                        i18n::chat_prompt_assistant(),
                        &event.text,
                    ),
                    ChatStreamEventKind::Thinking => stream_printer.borrow_mut().write(
                        render::ChatStreamBlockKind::Thinking,
                        i18n::chat_prompt_thinking(),
                        &event.text,
                    ),
                };
                let _ = result;
            },
        ) {
            Ok(response) => response,
            Err(err) => {
                let _ = stream_printer.borrow_mut().finish();
                if let Some(mut spinner) = ai_wait_spinner.into_inner() {
                    spinner.stop();
                }
                if is_transient_ai_error(&err) {
                    let failure_notice = i18n::chat_ai_recoverable_failure(&mask_sensitive(
                        &i18n::localize_error(&err),
                    ));
                    println!("{}", render::render_chat_warning(&failure_notice, colorful));
                    services
                        .session
                        .add_assistant_message(failure_notice.clone(), Some(group_id));
                    services.session.persist()?;
                    autosave_worker.submit_from_session(services.session);
                    last_assistant_reply = failure_notice;
                    continue;
                }
                return Err(err);
            }
        };
        stream_printer.borrow_mut().finish()?;
        if let Some(mut spinner) = ai_wait_spinner.into_inner() {
            spinner.stop();
        }
        logging::info("AI chat finished");

        if let Some(stop_reason) = response.stop_reason {
            println!(
                "{}",
                render::render_chat_warning(
                    &i18n::chat_tool_guard_warning(
                        stop_reason.code(),
                        response.tool_rounds_used,
                        response.total_tool_calls,
                        services.cfg.ai.chat.max_tool_rounds,
                        services.cfg.ai.chat.max_total_tool_calls,
                    ),
                    services.cfg.console.colorful,
                )
            );
        }

        if !printed_round_thinking
            && let Some(thinking) = response.thinking.as_deref()
            && !thinking.trim().is_empty()
            && thinking.trim() != last_round_thinking.trim()
        {
            println!(
                "{}",
                render::render_chat_thinking(
                    i18n::chat_prompt_thinking(),
                    thinking,
                    services.cfg.console.colorful
                )
            );
        }
        let final_content = response.content.trim();
        let final_content_already_printed =
            !last_round_content.trim().is_empty() && last_round_content.trim() == final_content;
        if !final_content_already_printed {
            println!(
                "{}",
                render::render_chat_assistant_reply(
                    i18n::chat_prompt_assistant(),
                    final_content,
                    colorful
                )
            );
        }
        if services.cfg.ai.chat.show_round_metrics && services.cfg.ai.chat.show_tips {
            println!(
                "{}",
                render::render_chat_notice(
                    &i18n::chat_round_metrics(
                        response.metrics.api_rounds,
                        response.metrics.api_duration_ms,
                        response.metrics.prompt_tokens,
                        response.metrics.completion_tokens,
                        response.metrics.total_tokens,
                        response.metrics.estimated_cost_usd,
                        services.cfg.ai.chat.show_token_cost
                    ),
                    services.cfg.console.colorful
                )
            );
        }
        last_assistant_reply = response.archived_content.clone();

        services
            .session
            .add_assistant_message(response.archived_content.clone(), Some(group_id));
        services.session.persist()?;
        autosave_worker.submit_from_session(services.session);
        if let Some(selected) = maybe_pick_next_option(
            &response.content,
            services.cfg.console.colorful,
            services.cfg.ai.chat.show_tips,
        )? {
            pending_message = Some(selected);
        }
    }

    let final_message_count = services.session.message_count();
    let key_metrics = format!(
        "session_id={}\nchat_turns={}\nmessages_before={}\nmessages_after={}\nsummary_chars={}\ntool_calls={}\ntool_cache_hits={}\nmax_messages={}\nrecent_messages={}",
        services.session.session_id(),
        chat_turns,
        i18n::human_count_u128(initial_message_count as u128),
        i18n::human_count_u128(final_message_count as u128),
        i18n::human_count_u128(services.session.summary_len() as u128),
        tool_stats.tool_calls,
        tool_stats.cache_hits,
        services.cfg.session.max_messages,
        services.cfg.session.recent_messages
    );
    let risk_summary = if tool_stats.command_failures == 0 {
        i18n::risk_no_obvious().to_string()
    } else {
        format!(
            "tool_calls_failed={}, blocked={}, timeout={}, interrupted={}",
            tool_stats.command_failures,
            tool_stats.blocked_count,
            tool_stats.timeout_count,
            tool_stats.interrupted_count
        )
    };
    let ai_summary = if last_assistant_reply.trim().is_empty() {
        i18n::chat_goodbye().to_string()
    } else {
        last_assistant_reply
    };
    let command_summary = format!(
        "tool_calls_total={}\nfailures={}\nblocked={}\ntimeouts={}\ninterrupted={}\ncache_hits={}\ntool_duration={}",
        tool_stats.tool_calls,
        tool_stats.command_failures,
        tool_stats.blocked_count,
        tool_stats.timeout_count,
        tool_stats.interrupted_count,
        tool_stats.cache_hits,
        i18n::human_duration_ms(tool_stats.tool_duration_ms)
    );
    let rendered = render::render_action(
        services.assets_dir,
        "chat",
        &ActionRenderData {
            action: "chat".to_string(),
            status: i18n::status_success().to_string(),
            key_metrics,
            risk_summary,
            ai_summary,
            command_summary,
            elapsed: i18n::human_duration_ms(started.elapsed().as_millis()),
        },
        services.cfg.console.colorful,
    )?;
    autosave_worker.submit_from_session(services.session);
    autosave_worker.stop();

    Ok(ActionOutcome {
        rendered,
        exit_code: ExitCode::Success,
    })
}

fn generate_ai_summary(
    services: &mut ActionServices<'_>,
    prompt_name: &str,
    action: &str,
    target: &str,
    key_metrics: &str,
    results: &[CommandResult],
) -> Result<String, AppError> {
    let base_system_prompt = render::load_prompt_template(services.assets_dir, "system.md")?;
    let system_prompt =
        append_env_mode_prompt(&base_system_prompt, services.cfg.app.env_mode.as_str());
    let user_template = render::load_prompt_template(services.assets_dir, prompt_name)?;
    let command_details = format_command_details(results);

    let user_prompt = user_template
        .replace("{{action}}", action)
        .replace("{{target}}", target)
        .replace("{{key_metrics}}", key_metrics)
        .replace("{{command_details}}", &command_details);

    logging::info(&format!(
        "AI summarize start: action={action}, target={target}"
    ));
    let mut spinner = ActivitySpinner::start(
        i18n::progress_ai_summarizing(action, target),
        services.cfg.console.colorful,
    );
    let history = services.session.build_chat_history();
    let summary = services.ai.chat(&history, &system_prompt, &user_prompt)?;
    spinner.stop();
    logging::info(&format!(
        "AI summarize finished: action={action}, target={target}"
    ));
    Ok(summary)
}

#[derive(Debug, Deserialize)]
struct ShellToolArguments {
    #[serde(default)]
    label: Option<String>,
    command: String,
    #[serde(default)]
    mode: Option<String>,
}

fn execute_tool_call(
    services: &mut ActionServices<'_>,
    group_id: &str,
    tool_call: &ToolCallRequest,
    stats: &mut ChatToolStats,
    cache: &mut HashMap<String, ToolCallCacheItem>,
    cache_ttl_ms: u128,
) -> String {
    if tool_call.name != "run_shell_command" {
        if services.mcp.has_ai_tool(&tool_call.name) {
            return execute_mcp_tool_call(services, group_id, tool_call, stats);
        }
        let error = json!({
            "ok": false,
            "error": format!("unsupported tool function: {}", tool_call.name)
        });
        let text = error.to_string();
        services.session.add_tool_message(
            format!("tool_call_error={text}"),
            Some(group_id.to_string()),
        );
        let _ = services.session.persist();
        return text;
    }

    let parsed: ShellToolArguments = match serde_json::from_str(&tool_call.arguments) {
        Ok(value) => value,
        Err(err) => {
            let error = json!({
                "ok": false,
                "error": format!("invalid function arguments: {err}")
            });
            let text = error.to_string();
            services.session.add_tool_message(
                format!("tool_call_error={text}"),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            return text;
        }
    };

    let command = parsed.command.trim().to_string();
    if command.is_empty() {
        let error = json!({
            "ok": false,
            "error": "command is empty"
        });
        let text = error.to_string();
        services.session.add_tool_message(
            format!("tool_call_error={text}"),
            Some(group_id.to_string()),
        );
        let _ = services.session.persist();
        return text;
    }

    let label = parsed
        .label
        .as_deref()
        .unwrap_or("chat_tool")
        .trim()
        .to_string();
    let mode = parse_mode(parsed.mode.as_deref());
    let spec = CommandSpec {
        label: if label.is_empty() {
            "chat_tool".to_string()
        } else {
            label
        },
        command,
        mode,
    };
    if services.cfg.skills.enabled
        && let Some(skill_name) =
            detect_skill_name_from_command(&spec.command, services.cfg.skills.dir.as_str())
    {
        println!(
            "{}",
            render::render_chat_custom_tag_event(
                i18n::chat_tag_skill(),
                &i18n::chat_skill_workflow_started(&skill_name),
                services.cfg.console.colorful
            )
        );
    }
    let mode_text = if matches!(spec.mode, CommandMode::Write) {
        i18n::command_mode_write()
    } else {
        i18n::command_mode_read()
    };
    if services.cfg.ai.chat.show_tool {
        println!(
            "{}",
            render::render_chat_tool_event(
                &format_tool_running_output(
                    services,
                    &spec.label,
                    mode_text,
                    &mask_sensitive(&trim_text(&spec.command, 180))
                ),
                render::ChatToolEventKind::Running,
                services.cfg.console.colorful
            )
        );
    }

    let cache_mode = if spec.mode == CommandMode::Read {
        "read"
    } else {
        "write"
    };
    let cache_key = format!("{cache_mode}::{}", spec.command.trim().to_ascii_lowercase());
    if spec.mode == CommandMode::Read
        && cache_ttl_ms > 0
        && let Some(item) = cache.get(&cache_key)
    {
        let age = now_epoch_ms().saturating_sub(item.created_at_epoch_ms);
        if age <= cache_ttl_ms {
            stats.cache_hits += 1;
            if services.cfg.ai.chat.show_tool {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &format_tool_cache_hit_output(services, &spec.label, age),
                        render::ChatToolEventKind::Running,
                        services.cfg.console.colorful
                    )
                );
            }
            services.session.add_tool_message(
                format!(
                    "tool_call_id={} function={} args={} cache_hit=true",
                    tool_call.id,
                    tool_call.name,
                    trim_text(&tool_call.arguments, 400)
                ),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            return item.payload.clone();
        }
    }

    ShellExecutor::clear_interrupt_flag();
    let call_started = Instant::now();
    let run_timeout = Duration::from_secs(services.cfg.ai.chat.cmd_run_timout);
    match services.shell.run_with_timeout(&spec, run_timeout) {
        Ok(result) => {
            stats.tool_calls += 1;
            stats.tool_duration_ms += call_started.elapsed().as_millis();
            if !result.success {
                stats.command_failures += 1;
            }
            if result.blocked {
                stats.blocked_count += 1;
            }
            if result.timed_out {
                stats.timeout_count += 1;
            }
            if result.interrupted {
                stats.interrupted_count += 1;
            }
            let event_kind = if result.timed_out {
                render::ChatToolEventKind::Timeout
            } else if result.success {
                render::ChatToolEventKind::Success
            } else {
                render::ChatToolEventKind::Error
            };
            if should_show_tool_event(services, event_kind) {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &format_tool_finished_output(services, &result),
                        event_kind,
                        services.cfg.console.colorful
                    )
                );
            }
            if services.cfg.ai.chat.show_tool
                && let Some(preview) = build_command_output_preview(&result.stdout, &result.stderr)
            {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &format_tool_preview_output(services, &mask_sensitive(&preview)),
                        render::ChatToolEventKind::Running,
                        services.cfg.console.colorful
                    )
                );
            }
            let tool_result = json!({
                "ok": result.success,
                "label": result.label,
                "mode": result.mode,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "interrupted": result.interrupted,
                "blocked": result.blocked,
                "block_reason": mask_sensitive(&result.block_reason),
                "timeout_hint": result_timeout_hint(&spec.command, &result.stdout, &result.stderr),
                "stdout": mask_sensitive(&trim_text(result.stdout.trim(), 3000)),
                "stderr": mask_sensitive(&trim_text(result.stderr.trim(), 2000))
            });
            let text = tool_result.to_string();
            if result.timed_out
                && services.cfg.ai.chat.show_tool_timeout
                && let Some(hint) =
                    result_timeout_hint(&spec.command, &result.stdout, &result.stderr)
            {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &hint,
                        render::ChatToolEventKind::Timeout,
                        services.cfg.console.colorful
                    )
                );
            }
            if spec.mode == CommandMode::Read && result.success && cache_ttl_ms > 0 {
                cache.insert(
                    cache_key,
                    ToolCallCacheItem {
                        created_at_epoch_ms: now_epoch_ms(),
                        payload: text.clone(),
                    },
                );
            }
            services.session.add_tool_message(
                format!(
                    "tool_call_id={} function={} args={} result={}",
                    tool_call.id,
                    tool_call.name,
                    trim_text(&tool_call.arguments, 400),
                    trim_text(&text, 2400)
                ),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            text
        }
        Err(err) => {
            stats.tool_calls += 1;
            stats.command_failures += 1;
            stats.tool_duration_ms += call_started.elapsed().as_millis();
            if services.cfg.ai.chat.show_tool_err {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &i18n::chat_tool_finished(&spec.label, false, None, 0, false, false, false),
                        render::ChatToolEventKind::Error,
                        services.cfg.console.colorful
                    )
                );
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &format!("error={}", mask_sensitive(&err.to_string())),
                        render::ChatToolEventKind::Error,
                        services.cfg.console.colorful
                    )
                );
            }
            let tool_result = json!({
                "ok": false,
                "error": err.to_string()
            });
            let text = tool_result.to_string();
            services.session.add_tool_message(
                format!(
                    "tool_call_id={} function={} args={} result={}",
                    tool_call.id,
                    tool_call.name,
                    trim_text(&tool_call.arguments, 400),
                    trim_text(&text, 2400)
                ),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            text
        }
    }
}

fn execute_mcp_tool_call(
    services: &mut ActionServices<'_>,
    group_id: &str,
    tool_call: &ToolCallRequest,
    stats: &mut ChatToolStats,
) -> String {
    let started = Instant::now();
    let (server_name, remote_tool_name) = services
        .mcp
        .resolve_ai_tool_target(&tool_call.name)
        .unwrap_or_else(|| ("unknown".to_string(), tool_call.name.clone()));
    let started_text = if services.cfg.ai.chat.output_multilines {
        format!(
            "type=mcp_service_request\nserver={}\ntool={}\nai_tool={}",
            server_name, remote_tool_name, tool_call.name
        )
    } else {
        i18n::chat_mcp_service_request_started(&server_name, &remote_tool_name)
    };
    println!(
        "{}",
        render::render_chat_custom_tag_event(
            i18n::chat_tag_mcp(),
            &started_text,
            services.cfg.console.colorful
        )
    );
    match services
        .mcp
        .call_ai_tool(&tool_call.name, &tool_call.arguments)
    {
        Ok(content) => {
            let elapsed_ms = started.elapsed().as_millis();
            stats.tool_calls += 1;
            stats.tool_duration_ms += elapsed_ms;
            if services.cfg.ai.chat.show_tool_ok {
                let ok_text = if services.cfg.ai.chat.output_multilines {
                    format!(
                        "type=mcp_tool_result\ntool={}\nstatus={}\nduration={}\nduration_ms={}",
                        tool_call.name,
                        i18n::status_success(),
                        i18n::human_duration_ms(elapsed_ms),
                        i18n::human_count_u128(elapsed_ms)
                    )
                } else {
                    i18n::chat_tool_finished(
                        &tool_call.name,
                        true,
                        Some(0),
                        elapsed_ms,
                        false,
                        false,
                        false,
                    )
                };
                println!(
                    "{}",
                    render::render_chat_custom_tag_event(
                        i18n::chat_tag_mcp(),
                        &ok_text,
                        services.cfg.console.colorful
                    )
                );
            }
            let payload = json!({
                "ok": true,
                "tool": tool_call.name,
                "content": trim_text(&mask_sensitive(&content), 3000)
            })
            .to_string();
            services.session.add_tool_message(
                format!(
                    "tool_call_id={} function={} args={} result={}",
                    tool_call.id,
                    tool_call.name,
                    trim_text(&tool_call.arguments, 400),
                    trim_text(&payload, 2400)
                ),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            payload
        }
        Err(err) => {
            let elapsed_ms = started.elapsed().as_millis();
            stats.tool_calls += 1;
            stats.command_failures += 1;
            stats.tool_duration_ms += elapsed_ms;
            if services.cfg.ai.chat.show_tool_err {
                let err_text = if services.cfg.ai.chat.output_multilines {
                    format!(
                        "type=mcp_tool_result\ntool={}\nstatus={}\nduration={}\nduration_ms={}\nerror={}",
                        tool_call.name,
                        i18n::status_failed(),
                        i18n::human_duration_ms(elapsed_ms),
                        i18n::human_count_u128(elapsed_ms),
                        mask_sensitive(&err.to_string())
                    )
                } else {
                    format!("mcp_error={}", mask_sensitive(&err.to_string()))
                };
                println!(
                    "{}",
                    render::render_chat_custom_tag_event(
                        i18n::chat_tag_mcp(),
                        &err_text,
                        services.cfg.console.colorful
                    )
                );
            }
            let payload = json!({
                "ok": false,
                "tool": tool_call.name,
                "error": err.to_string()
            })
            .to_string();
            services.session.add_tool_message(
                format!(
                    "tool_call_id={} function={} args={} result={}",
                    tool_call.id,
                    tool_call.name,
                    trim_text(&tool_call.arguments, 400),
                    trim_text(&payload, 2400)
                ),
                Some(group_id.to_string()),
            );
            let _ = services.session.persist();
            payload
        }
    }
}

fn parse_mode(mode: Option<&str>) -> CommandMode {
    let normalized = mode.unwrap_or("read").trim().to_ascii_lowercase();
    if normalized == "write" {
        return CommandMode::Write;
    }
    CommandMode::Read
}

fn format_tool_running_output(
    services: &ActionServices<'_>,
    label: &str,
    mode_text: &str,
    command: &str,
) -> String {
    if !services.cfg.ai.chat.output_multilines {
        return i18n::chat_tool_running(label, mode_text, command);
    }
    format!(
        "type={}\nlabel={}\nmode={}\ncommand={}",
        i18n::chat_tool_type_shell_command(),
        label,
        mode_text,
        command
    )
}

fn format_tool_cache_hit_output(
    services: &ActionServices<'_>,
    label: &str,
    age_ms: u128,
) -> String {
    if !services.cfg.ai.chat.output_multilines {
        return i18n::chat_tool_cache_hit(label, age_ms);
    }
    format!(
        "type={}\nlabel={}\nage={}\nage_ms={}",
        "command_cache_hit",
        label,
        i18n::human_duration_ms(age_ms),
        i18n::human_count_u128(age_ms)
    )
}

fn format_tool_finished_output(services: &ActionServices<'_>, result: &CommandResult) -> String {
    if !services.cfg.ai.chat.output_multilines {
        return i18n::chat_tool_finished(
            &result.label,
            result.success,
            result.exit_code,
            result.duration_ms,
            result.timed_out,
            result.interrupted,
            result.blocked,
        );
    }
    let status = if result.success {
        i18n::status_success()
    } else {
        i18n::status_failed()
    };
    format!(
        "type={}\nlabel={}\nstatus={}\nexit={:?}\nduration={}\nduration_ms={}\ntimeout={}\ninterrupted={}\nblocked={}",
        i18n::chat_tool_type_shell_result(),
        result.label,
        status,
        result.exit_code,
        i18n::human_duration_ms(result.duration_ms),
        i18n::human_count_u128(result.duration_ms),
        result.timed_out,
        result.interrupted,
        result.blocked
    )
}

fn format_tool_preview_output(services: &ActionServices<'_>, preview: &str) -> String {
    if !services.cfg.ai.chat.output_multilines {
        return i18n::chat_tool_output_preview(preview);
    }
    format!(
        "type={}\noutput={}",
        i18n::chat_tool_type_output_preview(),
        preview
    )
}

fn detect_skill_name_from_command(command: &str, configured_dir: &str) -> Option<String> {
    let configured = configured_dir
        .trim()
        .trim_end_matches('/')
        .trim_end_matches('\\');
    let mut prefixes = Vec::<String>::new();
    if !configured.is_empty() {
        prefixes.push(configured.to_string());
        let expanded = expand_tilde(configured).to_string_lossy().to_string();
        if !expanded.is_empty() && expanded != configured {
            prefixes.push(expanded);
        }
    }
    prefixes.push("~/.codex/skills".to_string());
    if let Some(home) = dirs::home_dir() {
        prefixes.push(
            home.join(".codex")
                .join("skills")
                .to_string_lossy()
                .to_string(),
        );
    }
    prefixes.sort();
    prefixes.dedup();
    for prefix in prefixes {
        if let Some(name) = extract_skill_name_with_prefix(command, &prefix) {
            return Some(name);
        }
    }
    None
}

fn extract_skill_name_with_prefix(command: &str, prefix: &str) -> Option<String> {
    let normalized_prefix = prefix.trim().replace('\\', "/");
    if normalized_prefix.is_empty() {
        return None;
    }
    let normalized_command = command.replace('\\', "/");
    let marker = format!("{normalized_prefix}/");
    let start = normalized_command.find(&marker)?;
    let rest = &normalized_command[start + marker.len()..];
    let skill_name = rest
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-' || *ch == '.')
        .collect::<String>();
    if skill_name.is_empty() || skill_name == "." || skill_name == ".." {
        return None;
    }
    Some(skill_name)
}

struct BuiltinCommand {
    name: String,
    arg: String,
}

fn parse_builtin_command(message: &str) -> Option<BuiltinCommand> {
    let trimmed = message.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let mut end = 1usize;
    for (idx, ch) in trimmed.char_indices().skip(1) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            end = idx + ch.len_utf8();
            continue;
        }
        break;
    }
    if end <= 1 {
        return None;
    }
    let name = trimmed.get(1..end)?.to_ascii_lowercase();
    let known = [
        "help", "stats", "list", "change", "name", "new", "clear", "history", "exit", "quit",
    ];
    if !known.contains(&name.as_str()) {
        return None;
    }
    let suffix = trimmed.get(end..)?.trim_start();
    if suffix.starts_with('/') {
        return None;
    }
    Some(BuiltinCommand {
        name,
        arg: suffix.to_string(),
    })
}

fn normalize_chat_message(raw: &str) -> String {
    let stripped = strip_terminal_control_sequences(raw);
    let mut normalized = stripped
        .trim()
        .chars()
        .filter(|ch| {
            !matches!(
                ch,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}'
            ) && !ch.is_control()
        })
        .collect::<String>();
    if let Some(first) = normalized.chars().next()
        && matches!(first, '／' | '⁄' | '∕' | '⧸' | '╱')
    {
        normalized.replace_range(0..first.len_utf8(), "/");
    }
    normalized.trim().to_string()
}

fn normalize_builtin_command_alias(message: String) -> String {
    let prompt = i18n::chat_prompt_user().trim();
    let mut trimmed = message.trim();
    if let Some(rest) = trimmed.strip_prefix(prompt) {
        trimmed = rest.trim_start();
    }
    if trimmed.is_empty() || trimmed.starts_with('/') {
        return trimmed.to_string();
    }
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let first = parts.next().unwrap_or_default();
    let rest = parts.next().unwrap_or_default().trim();
    let first_lower = first.to_ascii_lowercase();
    let alias_noarg = ["help", "stats", "list", "new", "clear", "exit", "quit"];
    if alias_noarg.contains(&first_lower.as_str()) && rest.is_empty() {
        return format!("/{first_lower}");
    }
    if (first_lower == "change" || first_lower == "name") && !rest.is_empty() {
        return format!("/{first_lower} {rest}");
    }
    if first_lower == "history" {
        if rest.is_empty() {
            return "/history".to_string();
        }
        return format!("/history {rest}");
    }
    trimmed.to_string()
}

fn strip_terminal_control_sequences(input: &str) -> String {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Normal,
        Esc,
        Csi,
        Osc,
    }
    let mut state = State::Normal;
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match state {
            State::Normal => {
                if ch == '\u{1B}' {
                    state = State::Esc;
                } else {
                    out.push(ch);
                }
            }
            State::Esc => {
                if ch == '[' {
                    state = State::Csi;
                } else if ch == ']' {
                    state = State::Osc;
                } else {
                    state = State::Normal;
                }
            }
            State::Csi => {
                if ('\u{40}'..='\u{7E}').contains(&ch) {
                    state = State::Normal;
                }
            }
            State::Osc => {
                if ch == '\u{07}' {
                    state = State::Normal;
                }
            }
        }
    }
    out
}

fn format_chat_session_list_markdown(sessions: &[crate::context::SessionOverview]) -> String {
    if sessions.is_empty() {
        return i18n::chat_session_list_empty().to_string();
    }
    let mut lines = Vec::<String>::new();
    lines.push(format!(
        "### {}",
        i18n::chat_session_list_title(sessions.len())
    ));
    lines.push(String::new());
    lines.push(format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |",
        i18n::chat_session_list_header_active(),
        i18n::chat_session_list_header_name(),
        i18n::chat_session_list_header_id(),
        i18n::chat_session_list_header_messages(),
        i18n::chat_session_list_header_summary(),
        i18n::chat_session_list_header_created(),
        i18n::chat_session_list_header_updated(),
        i18n::chat_session_list_header_file(),
    ));
    lines.push("|---|---|---|---|---|---|---|---|".to_string());
    for item in sessions {
        let active = if item.active {
            i18n::chat_session_list_active_yes()
        } else {
            i18n::chat_session_list_active_no()
        };
        let counts = format!(
            "{} (u:{} a:{} t:{} s:{})",
            i18n::human_count_u128(item.message_count as u128),
            i18n::human_count_u128(item.user_count as u128),
            i18n::human_count_u128(item.assistant_count as u128),
            i18n::human_count_u128(item.tool_count as u128),
            i18n::human_count_u128(item.system_count as u128),
        );
        let summary = i18n::human_count_u128(item.summary_len as u128);
        let created = format_local_datetime(item.created_at_epoch_ms);
        let updated = format_local_datetime(item.last_updated_epoch_ms);
        let short_id = trim_text(&item.session_id, 16);
        lines.push(format!(
            "| {} | {} | `{}` | {} | {} | {} | {} | `{}` |",
            active,
            item.session_name.replace('|', "\\|"),
            short_id,
            counts,
            summary,
            created,
            updated,
            item.file_path.display()
        ));
    }
    lines.join("\n")
}

fn format_local_datetime(epoch_ms: u128) -> String {
    if epoch_ms == 0 || epoch_ms > i64::MAX as u128 {
        return "-".to_string();
    }
    let Some(ts) = Local.timestamp_millis_opt(epoch_ms as i64).single() else {
        return "-".to_string();
    };
    ts.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn parse_chat_history_limit(arg: &str) -> Result<usize, &'static str> {
    const DEFAULT_LIMIT: usize = 10;
    const MAX_LIMIT: usize = 200;
    let trimmed = arg.trim();
    if trimmed.is_empty() {
        return Ok(DEFAULT_LIMIT);
    }
    let parsed = trimmed.parse::<usize>().ok();
    let Some(value) = parsed else {
        return Err(i18n::chat_history_usage());
    };
    if value == 0 {
        return Err(i18n::chat_history_usage());
    }
    if value > MAX_LIMIT {
        return Ok(MAX_LIMIT);
    }
    Ok(value)
}

fn format_chat_history_markdown(
    messages: &[SessionMessage],
    requested_limit: usize,
    total_messages: usize,
) -> String {
    if messages.is_empty() {
        return i18n::chat_history_empty().to_string();
    }
    let mut lines = Vec::<String>::new();
    lines.push(format!(
        "### {}",
        i18n::chat_history_title(messages.len(), requested_limit, total_messages)
    ));
    lines.push(String::new());
    for (idx, message) in messages.iter().enumerate() {
        let role = normalize_history_role(message.role.as_str());
        let safe_content = normalize_history_content(message.content.as_str());
        lines.push(format!(
            "{}. [{}] `{}` {}",
            idx + 1,
            format_local_datetime(message.created_at_epoch_ms),
            role,
            safe_content
        ));
    }
    lines.join("\n")
}

fn normalize_history_role(role: &str) -> &'static str {
    let normalized = role.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "assistant" => "assistant",
        "tool" => "tool",
        "system" => "system",
        "user" => "user",
        _ => "unknown",
    }
}

fn normalize_history_content(raw: &str) -> String {
    let stripped = strip_terminal_control_sequences(raw);
    let filtered = stripped
        .chars()
        .filter(|ch| {
            !matches!(
                ch,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{00AD}'
            ) && (*ch == '\n' || *ch == '\r' || *ch == '\t' || !ch.is_control())
        })
        .collect::<String>();
    let collapsed = trim_text(&mask_sensitive(filtered.trim()), 400)
        .replace('\n', "\\n")
        .replace('\r', "");
    if collapsed.is_empty() {
        return i18n::chat_history_empty_content().to_string();
    }
    collapsed
}

fn result_timeout_hint(command: &str, stdout: &str, stderr: &str) -> Option<String> {
    let command_lower = command.to_ascii_lowercase();
    let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let has_install_prompt = combined.contains("need to install the following packages")
        || combined.contains("ok to proceed")
        || combined.contains("proceed?")
        || combined.contains("y/n");
    if command_lower.contains("npx") && !command_lower.contains("--yes") && has_install_prompt {
        return Some(match i18n::current_language() {
            i18n::Language::ZhCn => {
                "检测到 npx 安装确认交互导致超时，建议改用 `npx --yes ...` 或先手动安装依赖。"
                    .to_string()
            }
            i18n::Language::ZhTw => {
                "偵測到 npx 安裝確認互動導致逾時，建議改用 `npx --yes ...` 或先手動安裝依賴。"
                    .to_string()
            }
            i18n::Language::Fr => {
                "Un prompt d'installation npx a provoqué le timeout; utilisez `npx --yes ...` ou installez la dépendance avant."
                    .to_string()
            }
            i18n::Language::De => {
                "Ein interaktiver npx-Installationsprompt hat das Timeout ausgelöst; verwenden Sie `npx --yes ...` oder installieren Sie Abhängigkeiten vorab."
                    .to_string()
            }
            i18n::Language::Ja => {
                "npx のインストール確認プロンプトでタイムアウトしました。`npx --yes ...` を使うか、依存関係を先に手動インストールしてください。"
                    .to_string()
            }
            i18n::Language::En => {
                "An interactive npx install prompt caused timeout; use `npx --yes ...` or preinstall dependencies."
                    .to_string()
            }
        });
    }
    None
}

fn clear_terminal() -> Result<(), AppError> {
    print!("\x1B[2J\x1B[H");
    io::stdout()
        .flush()
        .map_err(|err| AppError::Command(format!("failed to clear terminal: {err}")))
}

fn maybe_prepare_chat_model_price(services: &mut ActionServices<'_>) {
    let colorful = services.cfg.console.colorful;
    println!(
        "{}",
        render::render_chat_custom_tag_event(
            i18n::chat_tag_model_price(),
            &i18n::chat_model_price_check_started(&services.cfg.ai.model),
            colorful,
        )
    );
    if services.cfg.ai.chat.skip_model_price_check
        || normalize_chat_model_price_check_mode(
            services.cfg.ai.chat.model_price_check_mode.as_str(),
        ) != "async"
    {
        let result = services
            .ai
            .prepare_model_pricing(services.cfg.ai.chat.skip_model_price_check);
        println!(
            "{}",
            render::render_chat_custom_tag_event(
                i18n::chat_tag_model_price(),
                &format_model_price_check_message(result),
                colorful
            )
        );
        return;
    }

    let ai = services.ai.clone();
    let model = services.cfg.ai.model.clone();
    thread::spawn(move || {
        let result = ai.prepare_model_pricing(false);
        let message = format_model_price_check_message(result);
        println!(
            "{}",
            render::render_chat_custom_tag_event(i18n::chat_tag_model_price(), &message, colorful)
        );
        logging::info(&format!(
            "async model price check finished, model={}, message={}",
            model, message
        ));
    });
}

fn format_model_price_check_message(result: ModelPriceCheckResult) -> String {
    match (result.source, result.prices, result.probe_skipped) {
        (ModelPriceSource::Configured, Some((input, output)), _) => {
            i18n::chat_model_price_check_configured(input, output)
        }
        (ModelPriceSource::LocalCache, Some((input, output)), _) => {
            i18n::chat_model_price_check_cached(input, output)
        }
        (ModelPriceSource::RuntimeProbe, Some((input, output)), _) => {
            i18n::chat_model_price_check_probed(input, output)
        }
        (_, Some((input, output)), true) => {
            format!(
                "{}；{}",
                i18n::chat_model_price_check_skipped(),
                i18n::chat_model_price_check_builtin(input, output)
            )
        }
        (ModelPriceSource::Builtin, Some((input, output)), _) => {
            i18n::chat_model_price_check_builtin(input, output)
        }
        (_, None, true) => {
            format!(
                "{}；{}",
                i18n::chat_model_price_check_skipped(),
                i18n::chat_model_price_check_unavailable()
            )
        }
        _ => i18n::chat_model_price_check_unavailable().to_string(),
    }
}

fn render_chat_window_header(services: &ActionServices<'_>, after_clear: bool) {
    if after_clear {
        println!(
            "{}",
            render::render_markdown_for_terminal(
                i18n::chat_cleared(),
                services.cfg.console.colorful
            )
        );
    }
    if services.cfg.ai.chat.show_tips || after_clear {
        println!(
            "{}",
            render::render_chat_notice(
                &i18n::chat_welcome(
                    services.session.session_id(),
                    services.session.file_path(),
                    services.session.message_count(),
                    services.session.summary_len(),
                    services.cfg.session.recent_messages,
                    services.cfg.session.max_messages,
                    services.os_name,
                    &services.cfg.ai.model,
                    services.skills.len(),
                    services.mcp_summary.as_str(),
                    &services.cfg.ai.chat
                ),
                services.cfg.console.colorful
            )
        );
        println!(
            "{}",
            render::render_chat_notice(i18n::chat_hint(), services.cfg.console.colorful)
        );
        if let Some(warn_message) = services.session.context_pressure_warning(
            services.cfg.ai.chat.context_warn_percent,
            services.cfg.ai.chat.context_critical_percent,
        ) {
            println!(
                "{}",
                render::render_chat_notice(&warn_message, services.cfg.console.colorful)
            );
        }
    }
    if services.cfg.skills.enabled {
        println!(
            "{}",
            render::render_chat_custom_tag_event(
                i18n::chat_tag_skill(),
                &i18n::chat_skill_enabled(services.skills.len()),
                services.cfg.console.colorful
            )
        );
    }
}

fn build_chat_system_prompt(services: &ActionServices<'_>, base: &str) -> String {
    let mut prompt = append_env_mode_prompt(base, services.cfg.app.env_mode.as_str());
    let skills_list = if services.skills.is_empty() {
        "none".to_string()
    } else {
        services.skills.join(", ")
    };
    let mcp_tools = services
        .mcp
        .external_tool_definitions()
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
    let mcp_tools_list = if mcp_tools.is_empty() {
        "none".to_string()
    } else {
        mcp_tools.join(", ")
    };
    let skills_enabled = services.cfg.skills.enabled;
    let skills_dir = services.cfg.skills.dir.trim();
    prompt.push_str("\n\n[Capability Selection]\n- Shell execution, Skills, and MCP are peer capabilities. Choose the smallest sufficient one; do not systematically bias toward any single category.\n- Prefer Shell for deterministic local inspection, code edits, build/test, log/file/process analysis, and other machine-local operations.\n- Prefer Skills when a detected skill clearly matches the task and provides a reusable workflow or domain guardrails.\n- Prefer MCP when the task needs an integrated external service or dedicated remote capability that Shell/Skill cannot provide cleanly.\n- Avoid both extremes: do not stay text-only when evidence is missing, and do not chain tools endlessly when the evidence is already sufficient.\n- After each tool result, reassess whether you should answer in text now or call another tool.\n- If the model returns user-visible text in any round, preserve it and surface it; do not drop, overwrite, or silently ignore earlier assistant text.\n");
    prompt.push_str("\n[Skill Workflow]\n- Before complex tasks, scan available skills first.\n- If a matching skill exists, follow its SKILL.md workflow.\n- In responses, explicitly mention which skill is used.\n");
    prompt.push_str(&format!(
        "\n[Runtime Capability Context]\n- shell_tool=run_shell_command\n- skills_enabled={skills_enabled}\n- skills_dir={skills_dir}\n- detected_skills={skills_list}\n- detected_mcp_tools={mcp_tools_list}\n"
    ));
    prompt
}

fn append_env_mode_prompt(base: &str, env_mode_raw: &str) -> String {
    let env_mode = env_mode_raw.trim().to_ascii_lowercase();
    let policy = match env_mode.as_str() {
        "dev" => {
            "\n[Environment Mode]\n- env_mode=dev\n- You can be more exploratory for diagnostics and performance experiments.\n- Keep strong safety: never run destructive commands (rm -rf, mkfs, fdisk, shutdown, reboot) without explicit user confirmation.\n- For write actions, prefer reversible operations and explain rollback steps.\n- For potentially disruptive commands, scope them narrowly and cap output/time."
        }
        "test" => {
            "\n[Environment Mode]\n- env_mode=test\n- Balanced behavior: validate hypotheses with enough evidence while controlling execution cost.\n- Avoid long full-disk/full-process scans unless required; prefer sampling and staged checks.\n- Any write or service-affecting action must describe impact and require explicit confirmation.\n- Never execute destructive commands by default."
        }
        _ => {
            "\n[Environment Mode]\n- env_mode=prod\n- Conservative-first policy.\n- Prefer low-impact, low-overhead read commands.\n- Avoid high CPU/IO commands by default (e.g. full recursive scans over large paths) unless strictly necessary.\n- Never run destructive commands (especially rm/rm -rf) or service-stop operations without explicit user confirmation and clear safety notes.\n- If risk is high, provide safer alternatives first."
        }
    };
    format!("{}{}", base.trim(), policy)
}

fn ensure_chat_environment_profile(
    services: &mut ActionServices<'_>,
    system_prompt: &str,
) -> Result<(), AppError> {
    if services.session.has_chat_profile() {
        return Ok(());
    }
    println!(
        "{}",
        render::render_chat_custom_tag_event(
            i18n::chat_tag_profile(),
            i18n::chat_profile_started(),
            services.cfg.console.colorful
        )
    );
    let profile_commands = chat_profile_commands(services.os_type);
    let mut shell_spinner = ActivitySpinner::start(
        i18n::chat_profile_collecting().to_string(),
        services.cfg.console.colorful,
    );
    let profile_results = match services.shell.run_many(&profile_commands) {
        Ok(results) => results,
        Err(err) => {
            shell_spinner.stop();
            println!(
                "{}",
                render::render_chat_custom_tag_event(
                    i18n::chat_tag_profile(),
                    &i18n::chat_profile_failed(&mask_sensitive(&err.to_string())),
                    services.cfg.console.colorful
                )
            );
            return Ok(());
        }
    };
    shell_spinner.stop();
    let profile_details = format_command_details(&profile_results);
    let profile_prompt = format!(
        "请基于以下本机探测信息生成一份环境画像，要求包含：系统版本、硬件概况、资源概况、网络概况、开发工具概况、风险提示。\n\n输出要求：\n1. 先给结论\n2. 列表化关键事实\n3. 风险等级与原因\n4. 后续诊断建议（最多3条）\n\n# command_details\n{}",
        trim_text(&profile_details, 8000)
    );
    let mut ai_spinner = ActivitySpinner::start(
        i18n::chat_profile_analyzing().to_string(),
        services.cfg.console.colorful,
    );
    let profile_summary_result = services.ai.chat(&[], system_prompt, &profile_prompt);
    ai_spinner.stop();
    match profile_summary_result {
        Ok(summary) => {
            services
                .session
                .add_system_message(SessionStore::wrap_chat_profile(&summary), None);
            services.session.add_tool_message(
                format!(
                    "chat_profile_command_details:\n{}",
                    trim_text(&profile_details, 3000)
                ),
                None,
            );
            services.session.persist()?;
            println!(
                "{}",
                render::render_chat_custom_tag_event(
                    i18n::chat_tag_profile(),
                    i18n::chat_profile_completed(),
                    services.cfg.console.colorful
                )
            );
            Ok(())
        }
        Err(err) => {
            println!(
                "{}",
                render::render_chat_custom_tag_event(
                    i18n::chat_tag_profile(),
                    &i18n::chat_profile_failed(&mask_sensitive(&err.to_string())),
                    services.cfg.console.colorful
                )
            );
            Ok(())
        }
    }
}

fn maybe_ensure_chat_environment_profile(
    services: &mut ActionServices<'_>,
    system_prompt: &str,
) -> Result<(), AppError> {
    if services.cfg.ai.chat.skip_env_profile {
        return Ok(());
    }
    ensure_chat_environment_profile(services, system_prompt)
}

fn maybe_run_ai_context_compression(
    services: &mut ActionServices<'_>,
    system_prompt: &str,
) -> Result<(), AppError> {
    let Some(plan) = services.session.build_ai_compression_plan() else {
        return Ok(());
    };
    println!(
        "{}",
        render::render_chat_custom_tag_event(
            i18n::chat_tag_compress(),
            &i18n::chat_compression_started(plan.candidate_messages),
            services.cfg.console.colorful
        )
    );
    let previous = if plan.previous_summaries.is_empty() {
        "none".to_string()
    } else {
        plan.previous_summaries
            .iter()
            .map(|item| trim_text(item, 500))
            .collect::<Vec<_>>()
            .join("\n---\n")
    };
    let keep_recent =
        compression_keep_recent_messages(services.cfg.ai.chat.compression.max_history_messages);
    let prompt = format!(
        "你需要压缩历史对话片段，保持事实与结论可追溯。\n\n要求：\n1. 保留目标、关键证据、执行结果、风险、未完成事项。\n2. 删除寒暄与重复内容。\n3. 输出结构：目标 / 事实 / 结论 / 风险 / 待办。\n4. 内容应可直接作为后续上下文继续推理。\n5. 最近 {keep_recent} 条消息会被保留，不在压缩范围内。\n\n# previous_compression_summaries\n{}\n\n# new_candidate_messages\n{}",
        trim_text(&previous, 2500),
        trim_text(&plan.transcript, 9000)
    );
    let mut ai_spinner = ActivitySpinner::start(
        i18n::chat_compression_running().to_string(),
        services.cfg.console.colorful,
    );
    let summary_result = services.ai.chat(&[], system_prompt, &prompt);
    ai_spinner.stop();
    let summary = match summary_result {
        Ok(content) => content,
        Err(err) => {
            println!(
                "{}",
                render::render_chat_custom_tag_event(
                    i18n::chat_tag_compress(),
                    &i18n::chat_compression_failed(&mask_sensitive(&err.to_string())),
                    services.cfg.console.colorful
                )
            );
            return Ok(());
        }
    };
    if let Some(result) = services.session.apply_ai_compression_summary(&summary) {
        services.session.persist()?;
        println!(
            "{}",
            render::render_chat_custom_tag_event(
                i18n::chat_tag_compress(),
                &i18n::chat_compression_completed(result.removed_messages, result.total_messages),
                services.cfg.console.colorful
            )
        );
    }
    Ok(())
}

fn maybe_pick_next_option(
    reply: &str,
    colorful: bool,
    show_tips: bool,
) -> Result<Option<String>, AppError> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Ok(None);
    }
    if !looks_like_option_prompt(reply) {
        return Ok(None);
    }
    let options = extract_reply_options(reply);
    if options.len() < 2 || options.len() > 8 {
        return Ok(None);
    }
    if show_tips {
        println!(
            "{}",
            render::render_chat_notice(i18n::chat_option_detected(), colorful)
        );
    }
    let selections = MultiSelect::with_theme(&ColorfulTheme::default())
        .with_prompt(i18n::chat_option_prompt())
        .items(&options)
        .interact()
        .map_err(|err| AppError::Command(format!("failed to read option selection: {err}")))?;
    if let Some(index) = selections.first()
        && let Some(option) = options.get(*index)
    {
        if show_tips {
            println!(
                "{}",
                render::render_chat_notice(&i18n::chat_option_selected(option), colorful)
            );
        }
        return Ok(Some(option.clone()));
    }
    Ok(None)
}

fn looks_like_option_prompt(reply: &str) -> bool {
    const HINTS: [&str; 14] = [
        "请选择",
        "请明确",
        "选择以下",
        "你希望执行",
        "可选操作",
        "pick one",
        "choose one",
        "choose an option",
        "which option",
        "select one",
        "option:",
        "option：",
        "sélectionnez",
        "wählen",
    ];
    let lowered = reply.to_ascii_lowercase();
    HINTS
        .iter()
        .any(|hint| lowered.contains(&hint.to_ascii_lowercase()))
}

fn extract_reply_options(reply: &str) -> Vec<String> {
    let ordered_re = Regex::new(r"^\d+\.\s+(.*)$").expect("valid ordered list regex");
    let mut options = Vec::<String>::new();
    for line in reply.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut candidate = if let Some(rest) = trimmed.strip_prefix("- ") {
            Some(rest.trim())
        } else if let Some(rest) = trimmed.strip_prefix("* ") {
            Some(rest.trim())
        } else if let Some(captures) = ordered_re.captures(trimmed) {
            captures.get(1).map(|m| m.as_str().trim())
        } else {
            None
        };
        let Some(value) = candidate.take() else {
            continue;
        };
        let mut option = value
            .trim_matches('"')
            .trim_matches('\'')
            .trim_matches('“')
            .trim_matches('”')
            .trim_matches('`')
            .trim()
            .to_string();
        if option.is_empty() {
            continue;
        }
        if let Some(idx) = option.find("（") {
            option = option[..idx].trim().to_string();
        } else if let Some(idx) = option.find(" (") {
            option = option[..idx].trim().to_string();
        }
        if option.len() < 2 || option.len() > 96 {
            continue;
        }
        if options.contains(&option) {
            continue;
        }
        options.push(option);
    }
    options
}

fn should_require_tool_call(message: &str) -> bool {
    const FORCE_TOOL_KEYWORDS: [&str; 28] = [
        "执行",
        "自动执行",
        "帮我执行",
        "你来执行",
        "检查",
        "排查",
        "查看",
        "运行命令",
        "命令",
        "inspect",
        "check",
        "run",
        "execute",
        "memory",
        "cpu",
        "disk",
        "network",
        "process",
        "filesystem",
        "hardware",
        "logs",
        "os",
        "内存",
        "CPU",
        "磁盘",
        "网络",
        "进程",
        "文件系统",
    ];
    let lowered = message.to_ascii_lowercase();
    FORCE_TOOL_KEYWORDS
        .iter()
        .any(|keyword| lowered.contains(&keyword.to_ascii_lowercase()))
}

fn should_show_tool_event(services: &ActionServices<'_>, kind: render::ChatToolEventKind) -> bool {
    match kind {
        render::ChatToolEventKind::Running => services.cfg.ai.chat.show_tool,
        render::ChatToolEventKind::Success => services.cfg.ai.chat.show_tool_ok,
        render::ChatToolEventKind::Error => services.cfg.ai.chat.show_tool_err,
        render::ChatToolEventKind::Timeout => services.cfg.ai.chat.show_tool_timeout,
    }
}

struct ChatAutosaveWorker {
    latest_snapshot: Arc<Mutex<Option<ChatAutosaveSnapshot>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct ChatAutosaveSnapshot {
    path: PathBuf,
    content: String,
}

impl ChatAutosaveWorker {
    fn start(session: &SessionStore, interval: Duration) -> Self {
        let latest_snapshot = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let latest_snapshot_cloned = latest_snapshot.clone();
        let stop_cloned = stop.clone();
        let handle = thread::spawn(move || {
            loop {
                if stop_cloned.load(Ordering::SeqCst) {
                    flush_chat_autosave_snapshot(&latest_snapshot_cloned);
                    break;
                }
                flush_chat_autosave_snapshot(&latest_snapshot_cloned);
                thread::sleep(interval);
            }
        });
        let worker = Self {
            latest_snapshot,
            stop,
            handle: Some(handle),
        };
        worker.submit_from_session(session);
        logging::info("chat autosave worker started");
        worker
    }

    fn submit_from_session(&self, session: &SessionStore) {
        match session.serialized_state_pretty() {
            Ok(raw) => self.submit_raw(ChatAutosaveSnapshot {
                path: session.autosave_file_path(),
                content: raw,
            }),
            Err(err) => logging::warn(&format!(
                "chat autosave serialize failed: {}",
                mask_sensitive(&err.to_string())
            )),
        }
    }

    fn submit_raw(&self, raw: ChatAutosaveSnapshot) {
        match self.latest_snapshot.lock() {
            Ok(mut slot) => {
                *slot = Some(raw);
            }
            Err(_) => {
                logging::warn("chat autosave snapshot lock poisoned");
            }
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ChatAutosaveWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn flush_chat_autosave_snapshot(snapshot: &Arc<Mutex<Option<ChatAutosaveSnapshot>>>) {
    let raw = match snapshot.lock() {
        Ok(mut slot) => slot.take(),
        Err(_) => {
            logging::warn("chat autosave snapshot lock poisoned");
            None
        }
    };
    let Some(snapshot_data) = raw else {
        return;
    };
    if let Err(err) = write_string_atomically_local(&snapshot_data.path, &snapshot_data.content) {
        logging::warn(&format!(
            "chat autosave persist failed path={}: {}",
            snapshot_data.path.display(),
            mask_sensitive(&err.to_string())
        ));
        if let Ok(mut slot) = snapshot.lock()
            && slot.is_none()
        {
            *slot = Some(snapshot_data);
        }
    }
}

fn write_string_atomically_local(path: &Path, content: &str) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("session.json");
    let tmp_name = format!(".{file_name}.{}.tmp", Uuid::new_v4());
    let tmp_path = path.with_file_name(tmp_name);
    fs::write(&tmp_path, content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write temp session file {}: {err}",
            tmp_path.display()
        ))
    })?;
    if let Err(err) = fs::rename(&tmp_path, path) {
        if path.exists() {
            let _ = fs::remove_file(path);
            fs::rename(&tmp_path, path).map_err(|rename_err| {
                AppError::Runtime(format!(
                    "failed to replace session file {} after rename error {}: {}",
                    path.display(),
                    err,
                    rename_err
                ))
            })?;
        } else {
            let _ = fs::remove_file(&tmp_path);
            return Err(AppError::Runtime(format!(
                "failed to move temporary file into place {}: {err}",
                path.display()
            )));
        }
    }
    Ok(())
}

struct ActivitySpinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl ActivitySpinner {
    fn start(message: String, colorful: bool) -> Self {
        if !io::stdout().is_terminal() {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_cloned = stop.clone();
        let handle = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut idx = 0usize;
            while !stop_cloned.load(Ordering::SeqCst) {
                let text = format!("{message} {}", frames[idx % frames.len()]);
                print!("\r{}", render::render_chat_notice(&text, colorful));
                let _ = io::stdout().flush();
                idx = idx.wrapping_add(1);
                thread::sleep(Duration::from_millis(120));
            }
            clear_spinner_line();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ActivitySpinner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn clear_spinner_line() {
    if !io::stdout().is_terminal() {
        return;
    }
    print!("\r{: <160}\r", "");
    let _ = io::stdout().flush();
}

fn stop_activity_spinner_once(
    spinner: &RefCell<Option<ActivitySpinner>>,
    first_output: &Arc<AtomicBool>,
) {
    if first_output.swap(true, Ordering::SeqCst) {
        return;
    }
    if let Some(mut active_spinner) = spinner.borrow_mut().take() {
        active_spinner.stop();
    }
}

fn compression_keep_recent_messages(max_history_messages: usize) -> usize {
    let compress_recent = max_history_messages / 2;
    max_history_messages.saturating_sub(compress_recent).max(1)
}

fn is_noise_message(message: &str) -> bool {
    message
        .chars()
        .all(|ch| !ch.is_alphanumeric() && !('\u{4e00}'..='\u{9fff}').contains(&ch))
}

fn has_command_failures(results: &[CommandResult]) -> bool {
    results.iter().any(|item| !item.success)
}

fn format_prepare_metrics(services: &ActionServices<'_>, results: &[CommandResult]) -> String {
    let total = results.len();
    let success = results.iter().filter(|item| item.success).count();
    let failed = total.saturating_sub(success);
    i18n::prepare_metrics_overview(
        services.os_name,
        total,
        success,
        failed,
        services.skills.len(),
        services.mcp_summary.as_str(),
        services.cfg.session.recent_messages,
        services.cfg.session.max_messages,
    )
}

fn format_inspect_metrics(
    services: &ActionServices<'_>,
    target: InspectTarget,
    results: &[CommandResult],
) -> String {
    let total = results.len();
    let success = results.iter().filter(|item| item.success).count();
    let failed = total.saturating_sub(success);
    let duration_ms: u128 = results.iter().map(|item| item.duration_ms).sum();
    format!(
        "os={}\ntarget={}\ncommands_total={}\ncommands_success={}\ncommands_failed={}\ncommands_total_duration={}\nskills_count={}\nmcp={}",
        services.os_name,
        target.as_str(),
        total,
        success,
        failed,
        i18n::human_duration_ms(duration_ms),
        services.skills.len(),
        services.mcp_summary.as_str()
    )
}

fn build_risk_summary(results: &[CommandResult]) -> String {
    let mut items = Vec::new();
    for item in results.iter().filter(|item| !item.success).take(5) {
        if item.blocked {
            items.push(format!("{} blocked: {}", item.label, item.block_reason));
            continue;
        }
        if item.timed_out {
            items.push(format!("{} timeout", item.label));
            continue;
        }
        if item.interrupted {
            items.push(format!("{} interrupted", item.label));
            continue;
        }
        let stderr = trim_text(item.stderr.trim(), 120);
        items.push(format!(
            "{} failed, exit_code={:?}, stderr={}",
            item.label,
            item.exit_code,
            mask_sensitive(&stderr)
        ));
    }

    if items.is_empty() {
        return i18n::risk_no_obvious().to_string();
    }
    items.join(" | ")
}

fn format_prepare_risk_summary(results: &[CommandResult]) -> String {
    let mut lines = Vec::new();
    for item in results.iter().filter(|item| !item.success).take(6) {
        let reason = if item.blocked {
            item.block_reason.clone()
        } else if item.timed_out {
            i18n::prepare_risk_timeout().to_string()
        } else if item.interrupted {
            i18n::prepare_risk_interrupted().to_string()
        } else if !item.stderr.trim().is_empty() {
            mask_sensitive(&trim_text(item.stderr.trim(), 120))
        } else {
            format!("exit_code={:?}", item.exit_code)
        };
        lines.push(format!("- {}: {}", item.label, reason));
    }
    if lines.is_empty() {
        return i18n::risk_no_obvious().to_string();
    }
    lines.join("\n")
}

fn format_prepare_command_summary(results: &[CommandResult]) -> String {
    let mut lines = Vec::new();
    for (idx, item) in results.iter().enumerate() {
        let mut line = format!(
            "{}. {} -> {}, {}",
            idx + 1,
            item.label,
            if item.success {
                i18n::status_success()
            } else {
                i18n::status_failed()
            },
            i18n::human_duration_ms(item.duration_ms)
        );
        if item.blocked {
            line.push_str(&format!(", blocked={}", item.block_reason));
        } else if item.timed_out {
            line.push_str(", timed_out");
        } else if item.interrupted {
            line.push_str(", interrupted");
        } else if !item.success {
            line.push_str(&format!(", exit={:?}", item.exit_code));
        }
        lines.push(line);
    }
    lines.join("\n")
}

fn format_command_summary(results: &[CommandResult]) -> String {
    let mut lines = Vec::new();
    for item in results {
        let marker = if item.success {
            i18n::status_ok()
        } else {
            i18n::status_fail_short()
        };
        let mut line = format!(
            "[{marker}] {} mode={} exit={:?} duration={}",
            item.label,
            item.mode,
            item.exit_code,
            i18n::human_duration_ms(item.duration_ms)
        );
        if item.blocked {
            line.push_str(&format!(" blocked_reason={}", item.block_reason));
        }
        if item.timed_out {
            line.push_str(" timed_out=true");
        }
        if item.interrupted {
            line.push_str(" interrupted=true");
        }
        lines.push(line);
    }
    lines.join("\n")
}

fn format_command_details(results: &[CommandResult]) -> String {
    let mut lines = Vec::new();
    for item in results {
        lines.push(format!(
            "label={}\ncommand={}\nmode={}\nexit={:?}\nduration_ms={}\nstdout={}\nstderr={}\n---",
            item.label,
            mask_sensitive(&item.command),
            item.mode,
            item.exit_code,
            item.duration_ms,
            mask_sensitive(&trim_text(item.stdout.trim(), 800)),
            mask_sensitive(&trim_text(item.stderr.trim(), 400))
        ));
    }
    trim_text(&lines.join("\n"), 10_000)
}

fn trim_text(input: &str, max_len: usize) -> String {
    if input.chars().count() <= max_len {
        return input.to_string();
    }
    let mut end = 0usize;
    for (count, (idx, ch)) in input.char_indices().enumerate() {
        if count >= max_len {
            break;
        }
        end = idx + ch.len_utf8();
    }
    let mut s = input[..end].to_string();
    s.push_str("...");
    s
}

fn build_command_output_preview(stdout: &str, stderr: &str) -> Option<String> {
    let source = if !stdout.trim().is_empty() {
        stdout
    } else if !stderr.trim().is_empty() {
        stderr
    } else {
        return None;
    };
    let mut preview_lines = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(2)
        .map(|line| trim_text(line, 120))
        .collect::<Vec<_>>();
    if preview_lines.is_empty() {
        return None;
    }
    if source.lines().skip(2).any(|line| !line.trim().is_empty()) {
        preview_lines.push("...".to_string());
    }
    Some(preview_lines.join(" | "))
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn chat_profile_commands(os: OsType) -> Vec<CommandSpec> {
    match os {
        OsType::Windows => vec![
            read_cmd("profile_identity", "whoami"),
            read_cmd(
                "profile_os",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,LastBootUpTime | Format-List\"",
            ),
            read_cmd(
                "profile_hardware",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | Format-List\"",
            ),
            read_cmd(
                "profile_disk",
                "powershell -NoProfile -Command \"Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | Format-Table -AutoSize\"",
            ),
            read_cmd(
                "profile_network",
                "powershell -NoProfile -Command \"Get-NetIPConfiguration | Select-Object InterfaceAlias,IPv4Address | Format-Table -AutoSize\"",
            ),
            read_cmd(
                "profile_process_top_mem",
                "powershell -NoProfile -Command \"Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,Id,CPU,WS | Format-Table -AutoSize\"",
            ),
        ],
        OsType::MacOS => vec![
            read_cmd("profile_identity", "whoami"),
            read_cmd("profile_os", "sw_vers && uname -a"),
            read_cmd("profile_uptime", "uptime"),
            read_cmd("profile_disk", "df -h"),
            read_cmd("profile_memory", "vm_stat | head -20"),
            read_cmd(
                "profile_network",
                "ifconfig | grep -E '^[a-z]|inet ' | head -80",
            ),
            read_cmd(
                "profile_process_top_mem",
                "ps aux | sort -nr -k 4 | head -8",
            ),
        ],
        OsType::Linux | OsType::Other => vec![
            read_cmd("profile_identity", "whoami"),
            read_cmd("profile_os", "uname -a && cat /etc/os-release"),
            read_cmd("profile_uptime", "uptime"),
            read_cmd("profile_disk", "df -h"),
            read_cmd("profile_memory", "free -h || cat /proc/meminfo | head -20"),
            read_cmd(
                "profile_network",
                "ip -brief address || ifconfig -a | head -80",
            ),
            read_cmd(
                "profile_process_top_mem",
                "ps aux | sort -nr -k 4 | head -8",
            ),
        ],
    }
}

fn prepare_commands(os: OsType) -> Vec<CommandSpec> {
    match os {
        OsType::Windows => vec![
            read_cmd("identity", "whoami"),
            read_cmd(
                "os_version",
                "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version | Format-List\"",
            ),
            read_cmd(
                "uptime",
                "powershell -NoProfile -Command \"(Get-Date) - (gcim Win32_OperatingSystem).LastBootUpTime\"",
            ),
        ],
        _ => vec![
            read_cmd("identity", "whoami"),
            read_cmd("os_version", "uname -a"),
            read_cmd("uptime", "uptime"),
        ],
    }
}

fn inspect_commands(os: OsType, target: InspectTarget) -> Vec<CommandSpec> {
    if matches!(target, InspectTarget::All) {
        let mut list = Vec::new();
        for item in [
            InspectTarget::Cpu,
            InspectTarget::Memory,
            InspectTarget::Disk,
            InspectTarget::Os,
            InspectTarget::Process,
            InspectTarget::Filesystem,
            InspectTarget::Hardware,
            InspectTarget::Logs,
            InspectTarget::Network,
        ] {
            list.extend(inspect_commands(os, item));
        }
        return list;
    }

    match os {
        OsType::Windows => inspect_windows_commands(target),
        _ => inspect_unix_commands(target),
    }
}

fn inspect_unix_commands(target: InspectTarget) -> Vec<CommandSpec> {
    match target {
        InspectTarget::Cpu => vec![read_cmd(
            "cpu_info",
            "sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | head -n 20",
        )],
        InspectTarget::Memory => vec![read_cmd(
            "memory_info",
            "vm_stat 2>/dev/null || free -h 2>/dev/null",
        )],
        InspectTarget::Disk => vec![read_cmd("disk_usage", "df -h")],
        InspectTarget::Os => vec![read_cmd("os_info", "uname -a")],
        InspectTarget::Process => vec![read_cmd("process_top", "ps aux | head -n 30")],
        InspectTarget::Filesystem => vec![read_cmd("filesystem_mounts", "mount | head -n 40")],
        InspectTarget::Hardware => vec![read_cmd(
            "hardware_info",
            "system_profiler SPHardwareDataType 2>/dev/null | head -n 40 || lshw -short 2>/dev/null | head -n 40",
        )],
        InspectTarget::Logs => vec![read_cmd(
            "logs_snapshot",
            "ls -lah /var/log 2>/dev/null | head -n 30 || journalctl -n 30 --no-pager 2>/dev/null",
        )],
        InspectTarget::Network => vec![read_cmd(
            "network_info",
            "ifconfig 2>/dev/null | head -n 60 || ip addr 2>/dev/null | head -n 60",
        )],
        InspectTarget::All => Vec::new(),
    }
}

fn inspect_windows_commands(target: InspectTarget) -> Vec<CommandSpec> {
    match target {
        InspectTarget::Cpu => vec![read_cmd(
            "cpu_info",
            "powershell -NoProfile -Command \"Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List\"",
        )],
        InspectTarget::Memory => vec![read_cmd(
            "memory_info",
            "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | Format-List\"",
        )],
        InspectTarget::Disk => vec![read_cmd(
            "disk_usage",
            "powershell -NoProfile -Command \"Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | Format-Table -AutoSize\"",
        )],
        InspectTarget::Os => vec![read_cmd(
            "os_info",
            "powershell -NoProfile -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | Format-List\"",
        )],
        InspectTarget::Process => vec![read_cmd(
            "process_top",
            "powershell -NoProfile -Command \"Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Name,Id,CPU,PM | Format-Table -AutoSize\"",
        )],
        InspectTarget::Filesystem => vec![read_cmd(
            "filesystem_info",
            "powershell -NoProfile -Command \"Get-Volume | Select-Object DriveLetter,FileSystemLabel,FileSystem,SizeRemaining,Size | Format-Table -AutoSize\"",
        )],
        InspectTarget::Hardware => vec![read_cmd(
            "hardware_info",
            "powershell -NoProfile -Command \"Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | Format-List\"",
        )],
        InspectTarget::Logs => vec![read_cmd(
            "logs_snapshot",
            "powershell -NoProfile -Command \"Get-WinEvent -LogName System -MaxEvents 20 | Select-Object TimeCreated,Id,LevelDisplayName,Message | Format-Table -Wrap\"",
        )],
        InspectTarget::Network => vec![read_cmd(
            "network_info",
            "powershell -NoProfile -Command \"Get-NetIPConfiguration | Format-List\"",
        )],
        InspectTarget::All => Vec::new(),
    }
}

fn read_cmd(label: &str, command: &str) -> CommandSpec {
    CommandSpec {
        label: label.to_string(),
        command: command.to_string(),
        mode: CommandMode::Read,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChatAutosaveSnapshot, flush_chat_autosave_snapshot, normalize_builtin_command_alias,
        normalize_history_content, parse_builtin_command, parse_chat_history_limit,
    };
    use std::fs;
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    #[test]
    fn parse_builtin_command_supports_history_with_optional_limit() {
        let cmd = parse_builtin_command("/history 20").expect("history should parse");
        assert_eq!(cmd.name, "history");
        assert_eq!(cmd.arg, "20");
        let cmd_noarg = parse_builtin_command("/history").expect("history no-arg should parse");
        assert_eq!(cmd_noarg.name, "history");
        assert!(cmd_noarg.arg.is_empty());
    }

    #[test]
    fn normalize_builtin_alias_supports_history() {
        assert_eq!(
            normalize_builtin_command_alias("history".to_string()),
            "/history"
        );
        assert_eq!(
            normalize_builtin_command_alias("history 25".to_string()),
            "/history 25"
        );
    }

    #[test]
    fn parse_chat_history_limit_defaults_and_caps() {
        assert_eq!(parse_chat_history_limit("").expect("default"), 10);
        assert_eq!(parse_chat_history_limit("5").expect("value"), 5);
        assert_eq!(parse_chat_history_limit("999").expect("capped"), 200);
        assert!(parse_chat_history_limit("0").is_err());
        assert!(parse_chat_history_limit("abc").is_err());
    }

    #[test]
    fn normalize_history_content_sanitizes_control_and_empty() {
        let cleaned = normalize_history_content("\u{1B}[31mhello\u{1B}[0m\nworld");
        assert_eq!(cleaned, "hello\\nworld");
        let empty = normalize_history_content("\u{200B}\u{0007}");
        assert_eq!(empty, crate::i18n::chat_history_empty_content());
    }

    #[test]
    fn flush_chat_autosave_snapshot_writes_to_snapshot_path() {
        let temp_dir = std::env::temp_dir().join(format!("machineclaw-actions-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let target_path = temp_dir.join("session-a.json.autosave");
        let snapshot = Arc::new(Mutex::new(Some(ChatAutosaveSnapshot {
            path: target_path.clone(),
            content: "{\"ok\":true}".to_string(),
        })));
        flush_chat_autosave_snapshot(&snapshot);
        let written = fs::read_to_string(&target_path).expect("autosave file should be written");
        assert_eq!(written, "{\"ok\":true}");
        let _ = fs::remove_dir_all(temp_dir);
    }
}
