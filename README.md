# agentprism

**A universal MCP server that exposes coding agents — GitHub Copilot, Claude Code, Codex — as background subagents behind a single, unified tool interface.**

agentprism lets one AI agent orchestrate other AI agents. Drop it into your MCP client (Claude Code, Cursor, Continue, …) and you gain nine tools — `agent_run`, `agent_spawn`, `agent_send`, `agent_wait`, `agent_status`, `agent_list`, `agent_kill`, `agent_models`, `agent_providers` — that drive any supported coding agent through its native protocol. Run several in parallel, hand off tasks between them, or use a cheaper model as a worker for a more expensive planner. agentprism speaks each provider's wire protocol natively (ACP JSON-RPC for Copilot, stream-JSON for Claude Code, exec-resume for Codex) — no fragile screen-scraping.

## Installation

**Recommended — no install required ([uvx](https://docs.astral.sh/uv/)):**

```bash
# uvx runs agentprism directly from PyPI, no pip install needed
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

You also need at least one supported coding-agent CLI installed and authenticated:

| Provider     | CLI                                  | Auth                    |
|--------------|--------------------------------------|-------------------------|
| Copilot      | `copilot` ([install][copilot-cli])   | `copilot login`         |
| Claude Code  | `claude` ([install][claude-cli])     | `claude` then `/login`  |
| Codex        | `codex` ([install][codex-cli])       | `codex login`           |

[copilot-cli]: https://docs.github.com/en/copilot/github-copilot-in-the-cli
[claude-cli]:  https://docs.anthropic.com/en/docs/claude-code
[codex-cli]:   https://github.com/openai/codex

## Usage with Claude Code

Add to `~/.claude/mcp.json` (create if it doesn't exist):

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

No flags needed — every running agentprism instance auto-starts an HTTP API
on a random free port and registers itself in `~/.agentprism/{pid}.json`. The
global dashboard discovers and aggregates all running instances:

```bash
agentprism dashboard           # default port 7070
agentprism dashboard --port 8080
```

Then open `http://localhost:7070` to see every active session across every
project, grouped by working directory. The legacy `--dashboard PORT` flag
still works for a single per-instance dashboard if you need it.

If you'd rather run via uvx, use `"command": "uvx", "args": ["agentprism"]`.

Restart Claude Code. The nine `agent_*` tools will appear. Try:

> Call `agent_providers` to see what's available, then use `agent_spawn` to start a Copilot session in `/tmp/playground` with the task "write a Python script that prints prime numbers up to 100", then `agent_wait` for it to finish.

## Usage with other MCP clients

Any MCP client that supports stdio servers works. The config shape is the same — point `command` at `agentprism` (or `uvx` + `args: ["agentprism"]`).

## Helping your agent know when to delegate

agentprism's tool descriptions include trigger conditions, but for reliable delegation add a short snippet to your project's `AGENTS.md` or `CLAUDE.md`:

```markdown
## Delegation with agentprism

You have access to the agentprism MCP server. Use it to delegate coding tasks
to external agents (Copilot, Claude Code, Codex) instead of doing the work yourself.

Trigger conditions — reach for agentprism when:
- The user says "let Copilot handle", "delegate to Copilot", "offload to an agent", or similar
- A task is large/mechanical and offloading would preserve your context window
- You want to run multiple tasks in parallel

Quick patterns:
- One-shot (no corrections): `agent_run(task, cwd)`
- Parallel workers: multiple `agent_spawn` calls, then `agent_wait` each
- With corrections: `agent_spawn` → `agent_wait` → `agent_send` → `agent_wait` → `agent_kill`

Default provider is Copilot (1x cost). Use `agent_providers` to check what's available.
```


## Tool reference

| Tool               | Args                                              | Returns                                    |
|--------------------|---------------------------------------------------|--------------------------------------------|
| `agent_providers`  | —                                                 | which providers are installed + authenticated |
| `agent_models`     | `provider?`                                       | model ids + cost multipliers per provider  |
| `agent_run`        | `task`, `cwd`, `provider?`, `model?`, `timeout?`  | output — one-shot, blocks, auto-cleans up  |
| `agent_spawn`      | `task`, `cwd`, `provider?`, `model?`, `mode?`     | `session_id` — non-blocking, persistent    |
| `agent_send`       | `session_id`, `message`                           | agent reply (blocks until response)        |
| `agent_status`     | `session_id`                                      | `working` \| `idle` \| `done` \| `error`  |
| `agent_wait`       | `session_id`, `timeout_seconds?`                  | accumulated output (blocks until done)     |
| `agent_list`       | —                                                 | all active sessions                        |
| `agent_kill`       | `session_id`                                      | terminates the subprocess                  |

**`provider`** values: `copilot`, `claude`, `codex` — omit to use `AGENTPRISM_DEFAULT_PROVIDER` (default: `copilot`)

**`mode`** values (Copilot / Claude Code): `agent` (default), `plan`, `autopilot`

## Push notifications

When a worker finishes, agentprism proactively notifies the orchestrating MCP client — no polling required.

If the client advertised the `sampling` capability (Claude Code does), agentprism sends a `sampling/createMessage` request: the LLM receives a structured wake-up message with the session summary and can immediately act on the results. Falls back to a `notifications/message` log event for clients that don't support sampling.

## Provider support

| Provider       | Status | Protocol                              |
|----------------|--------|---------------------------------------|
| GitHub Copilot | ✓      | ACP JSON-RPC over stdio               |
| Claude Code    | ✓      | stream-JSON bidirectional stdio       |
| Codex          | ✓      | `codex exec` / `codex exec resume`    |

## Model cost multipliers

Use `agent_models(provider="copilot")` at runtime to get the current list. Examples:

| Model (Copilot)       | Multiplier | Notes                  |
|-----------------------|-----------|------------------------|
| `auto` / `claude-sonnet-4.6` | 1x | default               |
| `claude-haiku-4.5`    | 0.33x     | cheapest Claude        |
| `gpt-5-mini`          | 0x        | free                   |
| `gpt-4.1`             | 0x        | free                   |
| `claude-opus-4.7`     | 7.5x      | deep reasoning only    |
| `gpt-5.5`             | 7.5x      | GPT flagship           |

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
CodexAdapter             CopilotAdapter
     │                         │
     ▼                         ▼
copilot --acp           claude stream-json
(subprocess)            (subprocess)
```

Sessions are fully isolated per Claude session — no cross-session interference. The standalone dashboard is read-only and discovers instances via `~/.agentprism/{pid}.json` lockfiles.

## Configuration

Environment variables:

| Variable               | Default    | Purpose                                   |
|------------------------|------------|-------------------------------------------|
| `AGENTPRISM_LOG_LEVEL`   | `INFO`     | Python logging level (logs go to stderr)  |
| `AGENTPRISM_COPILOT_BIN` | `copilot`  | Path to the `copilot` binary              |
| `AGENTPRISM_CLAUDE_BIN`  | `claude`   | Path to the `claude` binary               |
| `AGENTPRISM_CODEX_BIN`   | `codex`    | Path to the `codex` binary                |

## Development

```bash
git clone https://github.com/StefanMaron/agentprism
cd agentprism
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT
