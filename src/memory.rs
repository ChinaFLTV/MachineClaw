use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::{AiMemoryConfig, expand_tilde},
    error::AppError,
};

const USER_MEMORY_KIND: &str = "user";
const MAX_PROMPT_MEMORY_ITEMS: usize = 12;
const MAX_PROMPT_MEMORY_CHARS: usize = 4_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserMemoryRecord {
    pub id: String,
    pub created_at: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl UserMemoryRecord {
    pub fn new(content: String, tags: Vec<String>) -> Result<Self, AppError> {
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            created_at: now_rfc3339(),
            kind: USER_MEMORY_KIND.to_string(),
            content: normalize_memory_content(&content)?,
            tags: normalize_memory_tags(tags),
        })
    }
}

#[derive(Debug, Clone)]
pub struct UserMemoryManager {
    enabled: bool,
    file_path: PathBuf,
    records: Vec<UserMemoryRecord>,
}

impl UserMemoryManager {
    pub fn load(cfg: &AiMemoryConfig, executable_dir: &Path) -> Result<Self, AppError> {
        let file_path = resolve_user_memory_file_path(&cfg.user_memory_file, executable_dir)?;
        let records = load_records_from_file(file_path.as_path())?;
        Ok(Self {
            enabled: cfg.enabled,
            file_path,
            records,
        })
    }

    pub fn reconfigure(
        &mut self,
        cfg: &AiMemoryConfig,
        executable_dir: &Path,
    ) -> Result<(), AppError> {
        let file_path = resolve_user_memory_file_path(&cfg.user_memory_file, executable_dir)?;
        let records = load_records_from_file(file_path.as_path())?;
        self.enabled = cfg.enabled;
        self.file_path = file_path;
        self.records = records;
        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    pub fn records(&self) -> &[UserMemoryRecord] {
        &self.records
    }

    pub fn reload(&mut self) -> Result<(), AppError> {
        self.records = load_records_from_file(self.file_path.as_path())?;
        Ok(())
    }

    pub fn add(
        &mut self,
        content: String,
        tags: Vec<String>,
    ) -> Result<UserMemoryRecord, AppError> {
        let record = UserMemoryRecord::new(content, tags)?;
        self.records.push(record.clone());
        sort_records(&mut self.records);
        self.persist()?;
        Ok(record)
    }

    pub fn update(
        &mut self,
        id: &str,
        content: String,
        tags: Vec<String>,
    ) -> Result<UserMemoryRecord, AppError> {
        let target = self
            .records
            .iter_mut()
            .find(|item| item.id == id)
            .ok_or_else(|| AppError::Runtime(format!("user memory not found: {id}")))?;
        target.kind = USER_MEMORY_KIND.to_string();
        target.content = normalize_memory_content(&content)?;
        target.tags = normalize_memory_tags(tags);
        let updated = target.clone();
        sort_records(&mut self.records);
        self.persist()?;
        Ok(updated)
    }

    pub fn delete(&mut self, id: &str) -> Result<Option<UserMemoryRecord>, AppError> {
        let Some(idx) = self.records.iter().position(|item| item.id == id) else {
            return Ok(None);
        };
        let removed = self.records.remove(idx);
        self.persist()?;
        Ok(Some(removed))
    }

    pub fn render_prompt_section(&self) -> String {
        if !self.enabled {
            return "[User Memory]\n- status=disabled\n- Memory is turned off for this chat. Do not assume any stored user memory.\n".to_string();
        }
        if self.records.is_empty() {
            return "[User Memory]\n- status=enabled\n- none\n".to_string();
        }

        let selected = select_prompt_records(&self.records);
        let omitted = self.records.len().saturating_sub(selected.len());
        let mut lines = vec![
            "[User Memory]".to_string(),
            "- status=enabled".to_string(),
            "- These are explicit user-managed memory entries. Use them only when relevant.".to_string(),
            "- If the current user request conflicts with a memory item, follow the current request and treat the memory as stale.".to_string(),
            "- Ignore unrelated memories instead of forcing them into the answer.".to_string(),
        ];
        for (idx, item) in selected.iter().enumerate() {
            let tags = if item.tags.is_empty() {
                "[]".to_string()
            } else {
                format!("[{}]", item.tags.join(", "))
            };
            lines.push(format!(
                "{}. created_at={} type={} tags={}",
                idx + 1,
                item.created_at,
                item.kind,
                tags
            ));
            lines.push(format!(
                "   content={}",
                collapse_memory_text(item.content.as_str())
            ));
        }
        if omitted > 0 {
            lines.push(format!("- omitted={} older entries not injected", omitted));
        }
        let mut rendered = lines.join("\n");
        if rendered.chars().count() > MAX_PROMPT_MEMORY_CHARS {
            rendered = trim_chars(&rendered, MAX_PROMPT_MEMORY_CHARS);
        }
        rendered.push('\n');
        rendered
    }

    fn persist(&self) -> Result<(), AppError> {
        let raw = serde_json::to_string_pretty(&self.records)
            .map_err(|err| AppError::Runtime(format!("failed to serialize user memory: {err}")))?;
        write_string_atomically(self.file_path.as_path(), raw.as_str())
    }
}

pub fn resolve_user_memory_file_path(
    raw: &str,
    executable_dir: &Path,
) -> Result<PathBuf, AppError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(AppError::Config(
            "ai.memory.user-memory-file must not be empty".to_string(),
        ));
    }
    let expanded = expand_tilde(trimmed);
    if expanded.is_absolute() {
        return Ok(expanded);
    }
    Ok(executable_dir.join(expanded))
}

fn load_records_from_file(path: &Path) -> Result<Vec<UserMemoryRecord>, AppError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read user memory file {}: {err}",
            path.display()
        ))
    })?;
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let parsed = serde_json::from_str::<Vec<UserMemoryRecord>>(&raw).map_err(|err| {
        AppError::Runtime(format!(
            "failed to parse user memory file {}: {err}",
            path.display()
        ))
    })?;
    normalize_records(parsed)
}

fn normalize_records(records: Vec<UserMemoryRecord>) -> Result<Vec<UserMemoryRecord>, AppError> {
    let mut normalized = Vec::with_capacity(records.len());
    let mut ids = HashSet::<String>::new();
    for mut item in records {
        item.id = item.id.trim().to_string();
        if item.id.is_empty() {
            return Err(AppError::Runtime(
                "user memory entry id must not be empty".to_string(),
            ));
        }
        if !ids.insert(item.id.clone()) {
            return Err(AppError::Runtime(format!(
                "duplicated user memory id detected: {}",
                item.id
            )));
        }
        if item.created_at.trim().is_empty() {
            item.created_at = now_rfc3339();
        }
        item.kind = USER_MEMORY_KIND.to_string();
        item.content = normalize_memory_content(&item.content)?;
        item.tags = normalize_memory_tags(item.tags);
        normalized.push(item);
    }
    sort_records(&mut normalized);
    Ok(normalized)
}

fn normalize_memory_content(raw: &str) -> Result<String, AppError> {
    let normalized = raw
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_string();
    if normalized.is_empty() {
        return Err(AppError::Runtime(
            "user memory content must not be empty".to_string(),
        ));
    }
    Ok(normalized)
}

fn normalize_memory_tags(tags: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::<String>::new();
    let mut normalized = Vec::<String>::new();
    for item in tags {
        let tag = item.trim();
        if tag.is_empty() {
            continue;
        }
        let lowered = tag.to_ascii_lowercase();
        if seen.insert(lowered) {
            normalized.push(tag.to_string());
        }
    }
    normalized.sort_by(|left, right| {
        left.to_ascii_lowercase()
            .cmp(&right.to_ascii_lowercase())
            .then(left.cmp(right))
    });
    normalized
}

fn select_prompt_records(records: &[UserMemoryRecord]) -> Vec<UserMemoryRecord> {
    let mut selected = records
        .iter()
        .rev()
        .take(MAX_PROMPT_MEMORY_ITEMS)
        .cloned()
        .collect::<Vec<_>>();
    selected.reverse();
    selected
}

fn sort_records(records: &mut [UserMemoryRecord]) {
    records.sort_by(|left, right| {
        left.created_at
            .cmp(&right.created_at)
            .then(left.id.cmp(&right.id))
    });
}

fn collapse_memory_text(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn trim_chars(raw: &str, max_chars: usize) -> String {
    raw.chars().take(max_chars).collect()
}

fn write_string_atomically(path: &Path, content: &str) -> Result<(), AppError> {
    let parent = path.parent().ok_or_else(|| {
        AppError::Runtime(format!(
            "failed to resolve parent directory for user memory {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create user memory directory {}: {err}",
            parent.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("user-memory.json");
    let temp_path = parent.join(format!(".{}.{}.tmp", file_name, Uuid::new_v4()));
    fs::write(&temp_path, content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write temporary user memory file {}: {err}",
            temp_path.display()
        ))
    })?;
    if let Err(rename_err) = fs::rename(&temp_path, path) {
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return Err(AppError::Runtime(format!(
                "failed to replace user memory file {} (rename: {}; copy fallback: {})",
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
    use std::{fs, path::PathBuf};

    use crate::config::AiMemoryConfig;

    use super::{UserMemoryManager, resolve_user_memory_file_path};

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("machineclaw-memory-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.as_path()).expect("create temp dir");
        dir
    }

    #[test]
    fn resolve_user_memory_file_path_uses_executable_dir_for_relative_path() {
        let root = temp_dir();
        let resolved =
            resolve_user_memory_file_path(".machineclaw/memory/user-memory.json", root.as_path())
                .expect("resolve path");
        assert_eq!(
            resolved,
            root.join(".machineclaw")
                .join("memory")
                .join("user-memory.json")
        );
    }

    #[test]
    fn user_memory_manager_add_update_delete_persists() {
        let root = temp_dir();
        let cfg = AiMemoryConfig {
            enabled: true,
            user_memory_file: ".machineclaw/memory/user-memory.json".to_string(),
        };
        let mut manager = UserMemoryManager::load(&cfg, root.as_path()).expect("load manager");
        let created = manager
            .add(
                "Prefers concise Rust answers".to_string(),
                vec!["rust".to_string(), "style".to_string()],
            )
            .expect("add memory");
        assert_eq!(manager.records().len(), 1);
        let updated = manager
            .update(
                created.id.as_str(),
                "Prefers concise Rust and TUI answers".to_string(),
                vec!["rust".to_string(), "tui".to_string()],
            )
            .expect("update memory");
        assert!(updated.content.contains("TUI"));
        let deleted = manager
            .delete(created.id.as_str())
            .expect("delete memory")
            .expect("deleted item should exist");
        assert_eq!(deleted.id, created.id);
        let reloaded = UserMemoryManager::load(&cfg, root.as_path()).expect("reload manager");
        assert!(reloaded.records().is_empty());
    }

    #[test]
    fn render_prompt_section_reports_disabled_and_none() {
        let root = temp_dir();
        let cfg = AiMemoryConfig {
            enabled: false,
            user_memory_file: ".machineclaw/memory/user-memory.json".to_string(),
        };
        let manager = UserMemoryManager::load(&cfg, root.as_path()).expect("load manager");
        assert!(manager.render_prompt_section().contains("status=disabled"));
        let enabled_cfg = AiMemoryConfig {
            enabled: true,
            user_memory_file: ".machineclaw/memory/user-memory.json".to_string(),
        };
        let empty_manager =
            UserMemoryManager::load(&enabled_cfg, root.as_path()).expect("load enabled manager");
        assert!(empty_manager.render_prompt_section().contains("- none"));
    }

    #[test]
    fn render_prompt_section_keeps_records_sorted_and_normalized() {
        let root = temp_dir();
        let path = root
            .join(".machineclaw")
            .join("memory")
            .join("user-memory.json");
        fs::create_dir_all(path.parent().expect("memory dir")).expect("create memory dir");
        fs::write(
            &path,
            serde_json::to_string_pretty(&vec![
                serde_json::json!({
                    "id": "b-entry",
                    "created_at": "2026-03-22T01:14:13Z",
                    "type": "assistant",
                    "content": "  likes tui workflows  ",
                    "tags": ["TUI", " tui ", "workflow"]
                }),
                serde_json::json!({
                    "id": "a-entry",
                    "created_at": "2026-03-22T01:14:11Z",
                    "type": "user",
                    "content": "prefers concise rust answers",
                    "tags": ["rust", "style"]
                }),
            ])
            .expect("serialize records"),
        )
        .expect("write memory file");
        let cfg = AiMemoryConfig {
            enabled: true,
            user_memory_file: ".machineclaw/memory/user-memory.json".to_string(),
        };
        let manager = UserMemoryManager::load(&cfg, root.as_path()).expect("load manager");
        assert_eq!(manager.records()[0].id, "a-entry");
        assert_eq!(manager.records()[1].id, "b-entry");
        assert_eq!(manager.records()[1].kind, "user");
        assert_eq!(manager.records()[1].tags, vec!["TUI", "workflow"]);
        let rendered = manager.render_prompt_section();
        let first_idx = rendered
            .find("1. created_at=2026-03-22T01:14:11Z")
            .expect("first record should be present");
        let second_idx = rendered
            .find("2. created_at=2026-03-22T01:14:13Z")
            .expect("second record should be present");
        assert!(first_idx < second_idx);
        assert!(rendered.contains("type=user"));
        assert!(rendered.contains("tags=[rust, style]"));
        assert!(rendered.contains("tags=[TUI, workflow]"));
        assert!(rendered.contains("content=likes tui workflows"));
    }
}
