use std::{
    collections::HashMap,
    io::{self, BufRead, IsTerminal, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use dialoguer::{MultiSelect, theme::ColorfulTheme};
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::{
    ai::{AiClient, ExternalToolDefinition, ToolCallRequest, ToolUsePolicy},
    cli::InspectTarget,
    config::AppConfig,
    context::SessionStore,
    error::{AppError, ExitCode},
    i18n, logging,
    mask::mask_sensitive,
    mcp::McpManager,
    platform::OsType,
    render::{self, ActionRenderData},
    shell::{CommandMode, CommandResult, CommandSpec, ShellExecutor},
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
            elapsed: format!("{} ms", started.elapsed().as_millis()),
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
            elapsed: format!("{} ms", started.elapsed().as_millis()),
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

    let system_prompt = render::load_prompt_template(services.assets_dir, "chat_system.md")?;
    let initial_message_count = services.session.message_count();
    let mut chat_turns: usize = 0;
    let mut tool_stats = ChatToolStats::default();
    let mut last_assistant_reply = String::new();
    let mut pending_message: Option<String> = None;
    let mut tool_call_cache = HashMap::<String, ToolCallCacheItem>::new();
    let command_cache_ttl_ms = services.cfg.ai.chat.command_cache_ttl_seconds as u128 * 1000;
    let external_mcp_tools: Vec<ExternalToolDefinition> = services.mcp.external_tool_definitions();
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

        let message = if let Some(next_message) = pending_message.take() {
            println!("{next_message}");
            next_message
        } else {
            let mut input = Vec::<u8>::new();
            let read_count = {
                let stdin = io::stdin();
                let mut stdin_lock = stdin.lock();
                stdin_lock
                    .read_until(b'\n', &mut input)
                    .map_err(|err| AppError::Command(format!("failed to read chat input: {err}")))?
            };
            if read_count == 0 {
                break;
            }
            while matches!(input.last(), Some(b'\n' | b'\r')) {
                input.pop();
            }
            String::from_utf8_lossy(&input).trim().to_string()
        };
        if message.is_empty() {
            continue;
        }
        if is_noise_message(&message) {
            continue;
        }

        if message.eq_ignore_ascii_case("/exit") || message.eq_ignore_ascii_case("/quit") {
            break;
        }
        if message.eq_ignore_ascii_case("/help") {
            println!("{}", i18n::chat_help_text());
            continue;
        }
        if message.eq_ignore_ascii_case("/stats") {
            println!(
                "{}",
                i18n::chat_stats(
                    services.session.session_id(),
                    services.session.file_path(),
                    services.session.message_count(),
                    services.session.summary_len(),
                    services.cfg.session.recent_messages,
                    services.cfg.session.max_messages,
                    chat_turns,
                    services.os_name,
                    &services.cfg.ai.model,
                    services.skills.len(),
                    services.mcp_summary.as_str(),
                    services.session.count_by_role("user"),
                    services.session.count_by_role("assistant"),
                    services.session.count_by_role("tool"),
                    services.session.count_by_role("system")
                )
            );
            continue;
        }
        if message.eq_ignore_ascii_case("/new") {
            services.session.start_new_session_with_new_file()?;
            pending_message = None;
            last_assistant_reply.clear();
            println!(
                "{}",
                i18n::chat_session_switched(
                    services.session.session_id(),
                    services.session.file_path()
                )
            );
            continue;
        }
        if message.eq_ignore_ascii_case("/clear") {
            clear_terminal()?;
            println!("{}", i18n::chat_cleared());
            println!("{}", i18n::chat_hint());
            continue;
        }

        let group_id = Uuid::new_v4().to_string();
        chat_turns += 1;
        services
            .session
            .add_user_message(message.clone(), Some(group_id.clone()));
        if let Some(warn_message) = services.session.context_pressure_warning(
            services.cfg.ai.chat.context_warn_percent,
            services.cfg.ai.chat.context_critical_percent,
        ) {
            println!(
                "{}",
                render::render_chat_notice(&warn_message, services.cfg.console.colorful)
            );
        }
        services.session.persist()?;

        logging::info("AI chat start");
        let history = services.session.build_chat_history();
        let policy = if should_require_tool_call(&message) {
            ToolUsePolicy::RequireAtLeastOne
        } else {
            ToolUsePolicy::Auto
        };
        println!(
            "{}",
            render::render_chat_notice(
                i18n::chat_progress_analyzing(),
                services.cfg.console.colorful
            )
        );
        let response = services.ai.chat_with_shell_tool(
            &history,
            &system_prompt,
            &message,
            policy,
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
        )?;
        logging::info("AI chat finished");

        if let Some(thinking) = response.thinking.as_deref()
            && !thinking.trim().is_empty()
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
        println!(
            "{}",
            render::render_chat_assistant_reply(
                i18n::chat_prompt_assistant(),
                response.content.trim(),
                services.cfg.console.colorful
            )
        );
        if services.cfg.ai.chat.show_round_metrics {
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
        last_assistant_reply = response.content.clone();

        services
            .session
            .add_assistant_message(response.content.clone(), Some(group_id));
        services.session.persist()?;
        if let Some(selected) =
            maybe_pick_next_option(&response.content, services.cfg.console.colorful)?
        {
            pending_message = Some(selected);
        }
    }

    let final_message_count = services.session.message_count();
    let key_metrics = format!(
        "session_id={}\nchat_turns={}\nmessages_before={}\nmessages_after={}\nsummary_chars={}\ntool_calls={}\ntool_cache_hits={}\nmax_messages={}\nrecent_messages={}",
        services.session.session_id(),
        chat_turns,
        initial_message_count,
        final_message_count,
        services.session.summary_len(),
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
        "tool_calls_total={}\nfailures={}\nblocked={}\ntimeouts={}\ninterrupted={}\ncache_hits={}\ntool_duration_ms={}",
        tool_stats.tool_calls,
        tool_stats.command_failures,
        tool_stats.blocked_count,
        tool_stats.timeout_count,
        tool_stats.interrupted_count,
        tool_stats.cache_hits,
        tool_stats.tool_duration_ms
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
            elapsed: format!("{} ms", started.elapsed().as_millis()),
        },
        services.cfg.console.colorful,
    )?;

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
    let system_prompt = render::load_prompt_template(services.assets_dir, "system.md")?;
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
    let history = services.session.build_chat_history();
    let summary = services.ai.chat(&history, &system_prompt, &user_prompt)?;
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
    let mode_text = if matches!(spec.mode, CommandMode::Write) {
        i18n::command_mode_write()
    } else {
        i18n::command_mode_read()
    };
    if services.cfg.ai.chat.show_tool {
        println!(
            "{}",
            render::render_chat_tool_event(
                &i18n::chat_tool_running(
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
                        &i18n::chat_tool_cache_hit(&spec.label, age),
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
    match services.shell.run(&spec) {
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
                        &i18n::chat_tool_finished(
                            &result.label,
                            result.success,
                            result.exit_code,
                            result.duration_ms,
                            result.timed_out,
                            result.interrupted,
                            result.blocked
                        ),
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
                        &i18n::chat_tool_output_preview(&mask_sensitive(&preview)),
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
                "stdout": mask_sensitive(&trim_text(result.stdout.trim(), 3000)),
                "stderr": mask_sensitive(&trim_text(result.stderr.trim(), 2000))
            });
            let text = tool_result.to_string();
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
    match services
        .mcp
        .call_ai_tool(&tool_call.name, &tool_call.arguments)
    {
        Ok(content) => {
            stats.tool_calls += 1;
            stats.tool_duration_ms += started.elapsed().as_millis();
            if services.cfg.ai.chat.show_tool_ok {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &i18n::chat_tool_finished(
                            &tool_call.name,
                            true,
                            Some(0),
                            started.elapsed().as_millis(),
                            false,
                            false,
                            false
                        ),
                        render::ChatToolEventKind::Success,
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
            stats.tool_calls += 1;
            stats.command_failures += 1;
            stats.tool_duration_ms += started.elapsed().as_millis();
            if services.cfg.ai.chat.show_tool_err {
                println!(
                    "{}",
                    render::render_chat_tool_event(
                        &format!("mcp_error={}", mask_sensitive(&err.to_string())),
                        render::ChatToolEventKind::Error,
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

fn clear_terminal() -> Result<(), AppError> {
    print!("\x1B[2J\x1B[H");
    io::stdout()
        .flush()
        .map_err(|err| AppError::Command(format!("failed to clear terminal: {err}")))
}

fn maybe_pick_next_option(reply: &str, colorful: bool) -> Result<Option<String>, AppError> {
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
    println!(
        "{}",
        render::render_chat_notice(i18n::chat_option_detected(), colorful)
    );
    let selections = MultiSelect::with_theme(&ColorfulTheme::default())
        .with_prompt(i18n::chat_option_prompt())
        .items(&options)
        .interact()
        .map_err(|err| AppError::Command(format!("failed to read option selection: {err}")))?;
    if let Some(index) = selections.first()
        && let Some(option) = options.get(*index)
    {
        println!(
            "{}",
            render::render_chat_notice(&i18n::chat_option_selected(option), colorful)
        );
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
        "os={}\ntarget={}\ncommands_total={}\ncommands_success={}\ncommands_failed={}\ncommands_total_duration_ms={}\nskills_count={}\nmcp={}",
        services.os_name,
        target.as_str(),
        total,
        success,
        failed,
        duration_ms,
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
            item.label, item.exit_code, stderr
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
            trim_text(item.stderr.trim(), 120)
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
            "{}. {} -> {}, {}ms",
            idx + 1,
            item.label,
            if item.success {
                i18n::status_success()
            } else {
                i18n::status_failed()
            },
            item.duration_ms
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
            "[{marker}] {} mode={} exit={:?} duration={}ms",
            item.label, item.mode, item.exit_code, item.duration_ms
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
            item.command,
            item.mode,
            item.exit_code,
            item.duration_ms,
            trim_text(item.stdout.trim(), 800),
            trim_text(item.stderr.trim(), 400)
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
