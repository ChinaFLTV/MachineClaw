# MachineClaw

![Rust](https://img.shields.io/badge/Rust-Edition%202024-orange)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![UI](https://img.shields.io/badge/UI-CLI-5C4EE5)
![Status](https://img.shields.io/badge/Status-Active-success)
![Language](https://img.shields.io/badge/i18n-zh--CN%20%7C%20zh--TW%20%7C%20en%20%7C%20fr%20%7C%20de%20%7C%20ja-informational)
[![Version](https://img.shields.io/github/v/release/ChinaFLTV/MachineClaw?display_name=tag)](https://github.com/ChinaFLTV/MachineClaw/releases)

MachineClaw 是一个跨平台命令行工具（CLI），用于机器预检、状态巡检与交互式诊断。  
它通过 AI + 本地命令执行协同工作，输出结构化结果，适合日常排查与运维巡检场景。

## 功能概览

- `prepare`：执行运行前检查（配置、权限、AI 连通性等）。
- `inspect <target>`：按目标检查机器状态。
- `chat`：进入交互式诊断模式，支持工具调用与会话管理。
- `snapshot`：将当前配置快照安全内嵌到二进制，生成可直接运行的分发包。
- `show-config`：展示当前生效配置快照（已脱敏）。
- `config get/set`：读取或写入配置项。
- `test config`：校验配置文件语法与字段有效性。
- 多语言输出（简中/繁中/英文/法语/德语/日语）。
- 可选彩色终端渲染，自动降级纯文本。

## 环境要求

- Rust 工具链（建议 stable）。
- Linux/macOS/Windows。
- 可访问 AI 接口（OpenAI 兼容 API）。
- 运行 action 时需要高权限：
    - Linux/macOS：`root`
    - Windows：管理员权限

## 快速开始

### 1) 构建

```bash
cargo build --release
```

二进制默认位置：

```text
./target/release/MachineClaw
```

### 2) 生成配置模板

```bash
./target/release/MachineClaw --show-config-template > ./target/release/claw.toml
```

默认读取 `<可执行文件目录>/claw.toml`。  
若你更希望把示例文件放在仓库目录，可保存为 `sample/claw-sample.toml`，并通过 `--conf` 显式指定。

### 3) 运行示例

```bash
# 运行前检查
sudo ./target/release/MachineClaw prepare --conf=./sample/claw-sample.toml

# 状态巡检
sudo ./target/release/MachineClaw inspect all --conf=./sample/claw-sample.toml

# 进入 chat
sudo ./target/release/MachineClaw chat --conf=./sample/claw-sample.toml

# 查看当前生效配置快照（脱敏）
./target/release/MachineClaw show-config --conf=./sample/claw-sample.toml

# 生成内嵌配置快照的新二进制
./target/release/MachineClaw snapshot --conf=./sample/claw-sample.toml -o ./MachineClaw-prod
```

## 跨平台构建可执行文件

### 方式一：原生构建（同平台）

```bash
# 当前平台直接构建
cargo build --release
```

### 方式二：Rust target 交叉构建（推荐）

先安装目标平台：

```bash
rustup target add x86_64-unknown-linux-musl
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-pc-windows-msvc
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
```

按目标构建：

```bash
# Linux (musl) - 仓库已内置 zig 交叉编译脚本
cargo build --release --target x86_64-unknown-linux-musl

# Linux (gnu)
cargo build --release --target x86_64-unknown-linux-gnu

# macOS Intel
cargo build --release --target x86_64-apple-darwin

# macOS Apple Silicon
cargo build --release --target aarch64-apple-darwin
```

Windows 额外说明：

- 在 Windows 主机本地构建 MSVC：
```bash
cargo build --release --target x86_64-pc-windows-msvc
```
- 在 macOS/Linux 直接执行 `cargo build --target x86_64-pc-windows-msvc` 常见失败原因是缺少 MSVC SDK/`link.exe`。
- 若必须在 macOS/Linux 产出 `x86_64-pc-windows-msvc`，请使用 `cargo-xwin`：
```bash
cargo install cargo-xwin
cargo xwin build --release --target x86_64-pc-windows-msvc
```
- macOS 上若出现 `failed to find tool "llvm-lib"`，请先安装 LLVM 并补充 PATH：
```bash
brew install llvm
export PATH="$(brew --prefix llvm)/bin:$PATH"
```
- `cargo-xwin` 首次执行会下载 MSVC CRT/SDK，耗时通常较长。

产物路径格式：

```text
target/<target-triple>/release/MachineClaw
target/<target-triple>/release/MachineClaw.exe   # Windows
```

### 方式三：使用 cross（依赖 Docker）

```bash
cargo install cross --git https://github.com/cross-rs/cross
cross build --release --target x86_64-unknown-linux-musl
```

注意：

- 需要 Docker Desktop / Docker Engine 正常运行。
- 如果出现 `failed to connect to the docker API`，先启动 Docker 再重试。

### 方式四：使用 cargo-zigbuild（不依赖 Docker）

```bash
cargo install cargo-zigbuild
rustup target add x86_64-unknown-linux-musl
cargo zigbuild --release --target x86_64-unknown-linux-musl
```

注意：

- 若报 `can't find crate for core`，通常是目标未安装，执行 `rustup target add <target>`。

## 配置文件说明

- 文件格式：TOML
- 默认配置路径：`<可执行文件目录>/claw.toml`
- 也可使用全局参数覆盖：`-c, --conf <path>`

主要配置段：

- `[app]`：语言、环境模式（`prod/test/dev`）
- `[ai]`：API 地址、Token、模型、重试
- `[ai.chat]`：chat 行为、工具显示、压缩、超时、轮次上限
- `[cmd]`：命令超时、写命令确认、allow/deny 命令规则
- `[skills]`：skills 目录与开关
- `[mcp]`：MCP 开关与服务配置（含 `mcp.servers.<name>`）
- `[console]`：是否彩色输出
- `[log]`：日志目录、滚动策略、保留时长
- `[session]`：上下文窗口相关配置

### MCP 配置建议

- HTTP 连接建议使用 `server-url = ".../mcp"`（优先 `/mcp`；若服务仅暴露旧 `/sse`，建议在服务侧开启 `/mcp` 兼容入口）。
- `transport` 支持 `http` / `stdio`；不配置时会按 `command` 或 `endpoint/server-url` 自动推断。
- 鉴权可用 `auth-type` + `auth-token`，或在 `headers` 中显式配置 `Authorization`。

```toml
[mcp]
enabled = true

[mcp.servers.deepwiki]
transport = "http"
server-url = "https://mcp.deepwiki.com/mcp"
auth-type = "bearer"
auth-token = "<token>"
[mcp.servers.deepwiki.headers]
X-Trace-Id = "machineclaw"
```

## 命令说明

```bash
MachineClaw [OPTIONS] <COMMAND>
```

全局参数：

- `-c, --conf <path>`：指定配置文件路径
- `--show-config-template`：输出完整配置模板
- `-h, --help`：查看帮助

子命令：

- `prepare`
- `inspect <target>`，`target` 支持：
    - `cpu`
    - `memory`
    - `disk`
    - `os`
    - `process`
    - `filesystem`
    - `hardware`
    - `logs`
    - `network`
    - `all`
- `test config`
- `chat`
- `snapshot`
- `show-config`
- `config get <key>`
- `config set <key> <value>`

## Chat 内置指令

进入 `chat` 后可用：

- `/help`：帮助
- `/stats`：会话统计
- `/list`：列出会话
- `/change <id|name>`：切换会话
- `/name <new-name>`：重命名当前会话
- `/new`：新建会话
- `/clear`：清屏（不清历史）
- `/exit`：退出 chat

## 退出码

- `0`：成功
- `1`：一般运行失败
- `2`：权限不足
- `3`：配置错误
- `4`：AI 接口调用失败
- `5`：命令执行失败

## 日志与会话文件

- 日志目录默认：`<可执行文件目录>/logs/`（不存在会自动创建）
- 会话目录默认：`<运行目录>/.machineclaw/`

## 安全说明

- 写命令默认需要确认（可在配置中调整策略）。
- 支持命令 allow/deny 规则（正则）进行执行前拦截。
- 输出会做敏感信息脱敏（如 token、cookie、密码等）。
- 非交互环境下若写命令需要确认，会直接失败退出，不会阻塞等待输入。
