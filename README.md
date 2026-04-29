# agentprism

**A universal MCP server that exposes coding agents as background subagents behind a single, unified tool interface.**

agentprism lets one AI agent orchestrate other AI agents. Drop it into your MCP client (Claude Code, Cursor, Continue, …) and gain nine tools — `agent_run`, `agent_spawn`, `agent_send`, `agent_wait`, `agent_status`, `agent_list`, `agent_kill`, `agent_models`, `agent_providers` — that drive any supported coding agent through its native CLI. Run several in parallel, hand off tasks between them, or use a cheaper/local model as a worker while a more expensive planner directs it.

**8 providers out of the box:** GitHub Copilot, Claude Code, Codex, Gemini CLI, Ollama (local), OpenCode, Aider.

## Installation

**Recommended — no install required ([uvx](https://docs.astral.sh/uv/)):**

```bash
uvx agentprism
```

**Or install permanently:**

```bash
pip install agentprism
# or: uv tool install agentprism
```

**Or from source:**

```bash
git clone https://github.com/StefanMaron/agentprism
cd agentprism
pip install -e .
```

You need at least one supported CLI installed and authenticated:

| Provider     | CLI install                                                   | Auth command                  |
|--------------|---------------------------------------------------------------|-------------------------------|
| Copilot      | [GitHub Copilot CLI][copilot-cli]                            | `copilot login`               |
| Claude Code  | [claude.ai/code][claude-cli]                                  | `claude` → `/login`           |
| Codex        | [github.com/openai/codex][codex-cli]                         | `codex login`                 |
| Gemini       | `npm install -g @google/gemini-cli`                           | `gemini auth login`           |
| Ollama       | [ollama.com][ollama]                                          | pull a model: `ollama pull qwen2.5-coder:14b` |
| OpenCode     | `npm install -g opencode-ai`                                  | `opencode providers login`    |
| Aider        | `uv tool install --python 3.12 aider-chat`                   | set `ANTHROPIC_API_KEY` or use Ollama |

[copilot-cli]:  https://docs.github.com/en/copilot/github-copilot-in-the-cli
[claude-cli]:   https://docs.anthropic.com/en/docs/claude-code
[codex-cli]:    https://github.com/openai/codex
[ollama]:       https://ollama.com

## Usage with Claude Code

Register globally (recommended):

```bash
claude mcp add agentprism --scope user -- agentprism
# or via uvx (no install needed):
claude mcp add agentprism --scope user -- uvx agentprism
```

Or add manually to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "agentprism": {
      "command": "agentprism",
      "type": "stdio"
    }
  }
}
```

Restart Claude Code. The nine `agent_*` tools appear. Example prompt:

> Call `agent_providers` to see what's available, then use `agent_run` to let Copilot write a Python script that prints prime numbers up to 100 in `/tmp/playground`.

## Dashboard

Every running agentprism instance auto-starts an HTTP API on a random free port and registers itself in `~/.agentprism/{pid}.json`. A standalone dashboard discovers and aggregates all running instances:

```bash
agentprism dashboard           # default port 7070
agentprism dashboard --port 8080
```

Open `http://localhost:7070` to see every active session across every project, grouped by working directory, with live streaming output per session.

## Tool reference

| Tool               | Args                                              | Returns                                          |
|--------------------|---------------------------------------------------|--------------------------------------------------|
| `agent_providers`  | —                                                 | which providers are installed + authenticated    |
| `agent_models`     | `provider?`                                       | model ids + cost multipliers                     |
| `agent_run`        | `task`, `cwd`, `provider?`, `model?`, `timeout?`  | output — one-shot, blocks until done, auto-cleans |
| `agent_spawn`      | `task`, `cwd`, `provider?`, `model?`, `mode?`     | `session_id` — non-blocking, persistent          |
| `agent_send`       | `session_id`, `message`                           | non-blocking — use `agent_wait` to observe       |
| `agent_status`     | `session_id`                                      | status + git delta + activity info               |
| `agent_wait`       | `session_id`, `timeout_seconds?`                  | accumulated output + git delta (blocks)          |
| `agent_list`       | —                                                 | all active sessions                              |
| `agent_kill`       | `session_id`                                      | terminates the subprocess                        |

**`agent_status`** returns `process_alive`, `last_activity_seconds_ago`, `uptime_seconds`, `new_commits`, and `working_tree_changes` — enough to decide whether a session is stuck without calling `agent_wait`.

**`agent_wait` / `agent_run`** include `new_commits` and `working_tree_changes` in the result — no need to run `git log` or `git status` separately.

**`mode`** values (Copilot / Claude Code): `agent` (default), `plan`, `autopilot`

> **Note:** Background agents always run in **yolo mode** — they auto-approve all file edits and shell commands without prompting. This is intentional: subagents are unattended by design. Review their output (via `agent_wait` or the dashboard) before merging or running produced code.

## Provider guide

### Free — no quota

| Provider  | `provider=` | Model example                    | Notes                                   |
|-----------|-------------|----------------------------------|-----------------------------------------|
| Copilot   | `copilot`   | `gpt-4.1`, `gpt-5-mini`          | 0x quota, requires Copilot subscription |
| Ollama    | `ollama`    | `qwen2.5-coder:14b-8k`           | Local GPU — best for Q&A, single-file   |
| OpenCode  | `opencode`  | `opencode/big-pickle`            | Free bundled models, no API key         |

### Free — quota-limited

| Provider | `provider=` | Model                   | Quota     |
|----------|-------------|-------------------------|-----------|
| Gemini   | `gemini`    | `gemini-2.5-flash`      | ~1000/day |
| Gemini   | `gemini`    | `gemini-2.0-flash`      | Higher    |

### Paid

| Provider    | `provider=` | Model                   | Cost   | Notes                         |
|-------------|-------------|-------------------------|--------|-------------------------------|
| Copilot     | `copilot`   | `claude-haiku-4.5`      | 0.33x  | Cheapest Claude               |
| Copilot     | `copilot`   | `gpt-5.4-mini`          | 0.33x  | Cheapest GPT                  |
| Copilot     | `copilot`   | `claude-sonnet-4.6`     | 1x     | **Default** — most tasks      |
| Copilot     | `copilot`   | `claude-opus-4.7`       | 7.5x   | Deep reasoning                |
| Claude Code | `claude`    | any Claude model        | varies | Direct Anthropic billing      |
| Codex       | `codex`     | any OpenAI model        | varies | Direct OpenAI billing         |
| Aider       | `aider`     | `qwen2.5-coder:14b-8k`  | 0x     | Local Ollama — git repo req'd |

### Local LLM notes

Ollama and Aider use your local GPU. A 16 GB GPU (e.g. RTX 4080) runs `qwen2.5-coder:14b` well for Q&A and single-file edits (~84 tok/s at 8K context). For reliable multi-file editing, 32 GB+ VRAM (RTX 5090 or workstation GPU) is recommended so 70B quantized models fit.

Create a custom 8K-context modelfile if needed:
```
FROM qwen2.5-coder:14b
PARAMETER num_ctx 8192
```
```bash
ollama create qwen2.5-coder:14b-8k -f Modelfile
```

For OpenCode + Ollama, add to `~/.config/opencode/config.json`:
```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": {
        "qwen2.5-coder:14b-8k": { "name": "Qwen2.5 Coder 14B (8K)" }
      }
    }
  }
}
```

## Quota handling

When a provider hits its quota limit, `agent_run` / `agent_wait` return a structured error instead of a crash:

```json
{
  "error": "quota_exceeded",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "suggestion": "Try provider='ollama', model='qwen2.5-coder:14b-8k' for free local inference..."
}
```

## Helping your agent know when to delegate

Add a snippet to your project's `CLAUDE.md` or `AGENTS.md`:

```markdown
## Delegation with agentprism

You have access to the agentprism MCP server. Delegate coding tasks to external
agents instead of doing the work yourself to preserve context window.

Trigger conditions:
- User says "let Copilot handle", "delegate", "offload to an agent"
- Task is large/mechanical and would burn significant context
- Multiple tasks can run in parallel

Quick patterns:
- One-shot:          agent_run(task, cwd)
- Parallel workers:  multiple agent_spawn calls, then agent_wait each
- With corrections:  agent_spawn → agent_wait → agent_send → agent_wait → agent_kill
- Free brainstorm:   agent_run(task, cwd, provider="ollama", model="qwen2.5:14b")
- Free Gemini:       agent_run(task, cwd, provider="gemini", model="gemini-2.5-flash")

Default provider is copilot (claude-sonnet-4.6, 1x cost). Call agent_providers
to check what's available on this machine.
```

## Push notifications

When a worker finishes, agentprism proactively notifies the orchestrating MCP client.

If the client advertised the `sampling` capability (Claude Code does), agentprism sends a `sampling/createMessage` wake-up with the session summary so the orchestrator can immediately act. Falls back to `notifications/message` for clients that don't support sampling.

## Architecture

```
Claude session A          Claude session B          agentprism dashboard
(project X)               (project Y)               (standalone, any terminal)
     │                         │                              │
     ▼                         ▼                              │
agentprism (stdio)        agentprism (stdio)                  │
SessionRegistry           SessionRegistry                     │
HTTP API :auto ──────────────────────────────────────────────►│ fan-out
     │  writes                 │  writes              reads   │
     ▼                         ▼                      all     ▼
~/.agentprism/{pidA}.json  ~/.agentprism/{pidB}.json ──► grouped by project
     │                         │                         http://localhost:7070
     ▼                         ▼
CopilotAdapter           ClaudeCodeAdapter
GeminiAdapter            OllamaAdapter
AiderAdapter             OpenCodeAdapter
     │                         │
     ▼                         ▼
copilot -p --yolo        claude --output-format stream-json
(subprocess)             (subprocess)
```

Sessions are fully isolated per Claude session — no cross-session interference. The standalone dashboard is read-only and discovers instances via `~/.agentprism/{pid}.json` lockfiles.

## Configuration

Environment variables:

| Variable                   | Default      | Purpose                                        |
|----------------------------|--------------|------------------------------------------------|
| `AGENTPRISM_LOG_LEVEL`     | `INFO`       | Python logging level (logs go to stderr)       |
| `AGENTPRISM_DEFAULT_PROVIDER` | `copilot` | Provider used when `provider` arg is omitted   |
| `AGENTPRISM_COPILOT_BIN`   | `copilot`    | Path to the `copilot` binary                   |
| `AGENTPRISM_CLAUDE_BIN`    | `claude`     | Path to the `claude` binary                    |
| `AGENTPRISM_CODEX_BIN`     | `codex`      | Path to the `codex` binary                     |
| `AGENTPRISM_GEMINI_BIN`    | `gemini`     | Path to the `gemini` binary                    |
| `AGENTPRISM_OPENCODE_BIN`  | `opencode`   | Path to the `opencode` binary                  |
| `AGENTPRISM_AIDER_BIN`     | `aider`      | Path to the `aider` binary                     |
| `OLLAMA_API_BASE`          | `http://localhost:11434` | Ollama endpoint                  |

## Development

```bash
git clone https://github.com/StefanMaron/agentprism
cd agentprism
pip install -e ".[dev]"
ruff check .
pytest
```

See [CLAUDE.md](CLAUDE.md) for architecture notes and contributor guidance.

## License

MIT
