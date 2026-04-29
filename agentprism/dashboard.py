"""Lightweight HTTP dashboard for monitoring agentprism sessions.

Starts a minimal asyncio HTTP server alongside the MCP stdio server.
No extra dependencies — pure stdlib.

Usage:
    agentprism --dashboard 7070
    # then open http://localhost:7070
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentprism.session import SessionRegistry

log = logging.getLogger("agentprism.dashboard")

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentprism</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: ui-monospace, monospace; background: #0d1117; color: #e6edf3; padding: 24px; }
  h1 { font-size: 1.1rem; color: #58a6ff; margin-bottom: 4px; }
  .subtitle { font-size: 0.75rem; color: #8b949e; margin-bottom: 24px; }
  .empty { color: #8b949e; font-size: 0.85rem; padding: 16px 0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 500; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: top; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  tr:hover td { background: #161b22; }
  .status-working { color: #f0883e; }
  .status-idle    { color: #3fb950; }
  .status-done    { color: #8b949e; }
  .status-error   { color: #f85149; }
  .provider-copilot { color: #58a6ff; }
  .provider-claude  { color: #d2a8ff; }
  .provider-codex   { color: #79c0ff; }
  .task-cell { max-width: 360px; }
  .kill-btn { background: none; border: 1px solid #f85149; color: #f85149; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-family: inherit; }
  .kill-btn:hover { background: #f8514920; }
  .watch-btn { background: none; border: 1px solid #58a6ff; color: #58a6ff; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-family: inherit; }
  .watch-btn.active { background: #58a6ff22; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
  .dot-working { background: #f0883e; animation: pulse 1.2s infinite; }
  .dot-idle    { background: #3fb950; }
  .dot-done    { background: #8b949e; }
  .dot-error   { background: #f85149; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .footer { margin-top: 20px; font-size: 0.72rem; color: #484f58; }

  /* Terminal overlay */
  #terminal-overlay {
    display: none;
    position: fixed; inset: 0;
    background: #00000088;
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  #terminal-overlay.open { display: flex; }
  #terminal-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    width: 90vw; max-width: 900px;
    height: 70vh;
    display: flex; flex-direction: column;
    overflow: hidden;
    box-shadow: 0 24px 64px #000a;
  }
  #terminal-header {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    font-size: 0.78rem; color: #8b949e;
  }
  #terminal-header .close-btn {
    margin-left: auto;
    background: none; border: none; color: #8b949e;
    cursor: pointer; font-size: 1rem; font-family: inherit;
    padding: 0 4px;
  }
  #terminal-header .close-btn:hover { color: #e6edf3; }
  #terminal-title { color: #e6edf3; font-weight: 600; }
  #terminal-output {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
    font-size: 0.8rem;
    line-height: 1.5;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
  }
  #terminal-output .chunk-tool   { color: #79c0ff; }
  #terminal-output .chunk-think  { color: #8b949e; font-style: italic; }
  #terminal-output .chunk-text   { color: #c9d1d9; }
  #terminal-output .chunk-error  { color: #f85149; }
  #terminal-output .chunk-done   { color: #3fb950; }
  #terminal-status {
    padding: 6px 14px;
    font-size: 0.72rem; color: #484f58;
    border-top: 1px solid #21262d;
  }
  .cursor { display: inline-block; width: 8px; height: 13px; background: #f0883e; animation: blink .8s step-end infinite; vertical-align: text-bottom; margin-left: 2px; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<h1>agentprism</h1>
<p class="subtitle">active sessions &nbsp;·&nbsp; auto-refreshes every 2s</p>
<div id="root"><p class="empty">Loading…</p></div>
<p class="footer" id="ts"></p>

<!-- Live terminal overlay -->
<div id="terminal-overlay">
  <div id="terminal-box">
    <div id="terminal-header">
      <span class="dot dot-working" id="terminal-dot"></span>
      <span id="terminal-title">session</span>
      <span id="terminal-provider"></span>
      <button class="close-btn" onclick="closeTerminal()">✕</button>
    </div>
    <div id="terminal-output"></div>
    <div id="terminal-status">connecting…</div>
  </div>
</div>

<script>
let activeEs = null;
let activeId = null;

function elapsed(iso) {
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function kill(sid) {
  if (!confirm('Kill session ' + sid.slice(0,8) + '?')) return;
  await fetch('/api/sessions/' + sid, {method:'DELETE'});
  if (activeId === sid) closeTerminal();
}

function openTerminal(sid, provider, task) {
  if (activeEs) { activeEs.close(); activeEs = null; }
  activeId = sid;

  document.getElementById('terminal-title').textContent = sid.slice(0,8) + '… — ' + task.slice(0,60);
  document.getElementById('terminal-provider').textContent = provider;
  const dot = document.getElementById('terminal-dot');
  dot.className = 'dot dot-working';

  const out = document.getElementById('terminal-output');
  out.innerHTML = '';
  document.getElementById('terminal-status').innerHTML = 'connecting…';
  document.getElementById('terminal-overlay').classList.add('open');

  // Prevent background scroll
  document.body.style.overflow = 'hidden';

  const es = new EventSource('/api/sessions/' + sid + '/stream');
  activeEs = es;

  es.addEventListener('chunk', e => {
    const d = JSON.parse(e.data);
    const span = document.createElement('span');
    span.className = 'chunk-' + (d.kind || 'text');
    span.textContent = d.text;
    out.appendChild(span);
    out.scrollTop = out.scrollHeight;
    document.getElementById('terminal-status').innerHTML =
      'live · ' + new Date().toLocaleTimeString() + ' <span class="cursor"></span>';
  });

  es.addEventListener('history', e => {
    const text = JSON.parse(e.data).text;
    if (text) {
      const span = document.createElement('span');
      span.className = 'chunk-text';
      span.textContent = text;
      out.appendChild(span);
      out.scrollTop = out.scrollHeight;
    }
  });

  es.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    dot.className = 'dot dot-' + d.status;
    document.getElementById('terminal-status').textContent =
      d.status === 'working'
        ? 'live · ' + new Date().toLocaleTimeString()
        : d.status + ' · ' + new Date().toLocaleTimeString();
  });

  es.addEventListener('done', e => {
    dot.className = 'dot dot-done';
    document.getElementById('terminal-status').textContent = 'session done';
    const span = document.createElement('span');
    span.className = 'chunk-done';
    span.textContent = '\\n\\n[session complete]';
    out.appendChild(span);
    out.scrollTop = out.scrollHeight;
    es.close();
    activeEs = null;
  });

  es.onerror = () => {
    document.getElementById('terminal-status').textContent = 'stream closed';
    es.close();
    activeEs = null;
  };
}

function closeTerminal() {
  if (activeEs) { activeEs.close(); activeEs = null; }
  activeId = null;
  document.getElementById('terminal-overlay').classList.remove('open');
  document.body.style.overflow = '';
}

// Close on overlay click
document.getElementById('terminal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('terminal-overlay')) closeTerminal();
});

// Close on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeTerminal(); });

async function refresh() {
  const res = await fetch('/api/sessions');
  const { sessions } = await res.json();
  const root = document.getElementById('root');
  document.getElementById('ts').textContent = 'last update ' + new Date().toLocaleTimeString();

  if (!sessions.length) {
    root.innerHTML = '<p class="empty">No active sessions.</p>';
    return;
  }

  let html = '<table><thead><tr><th>id</th><th>provider</th><th>model</th><th>status</th><th>elapsed</th><th class="task-cell">task</th><th></th></tr></thead><tbody>';
  for (const s of sessions) {
    const short = s.session_id.slice(0,8);
    const isActive = activeId === s.session_id;
    const taskEsc = s.initial_task.replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'");
    html += `<tr>
      <td title="${s.session_id}">${short}…</td>
      <td class="provider-${s.provider}">${s.provider}</td>
      <td>${s.model || 'auto'}</td>
      <td class="status-${s.status}"><span class="dot dot-${s.status}"></span>${s.status}</td>
      <td>${elapsed(s.created_at)}</td>
      <td class="task-cell" title="${esc(s.initial_task)}">${esc(s.initial_task.slice(0,80))}${s.initial_task.length>80?'…':''}</td>
      <td style="white-space:nowrap">
        <button class="watch-btn${isActive?' active':''}" onclick="openTerminal('${s.session_id}','${s.provider}','${taskEsc}')">▶ watch</button>
        &nbsp;
        <button class="kill-btn" onclick="kill('${s.session_id}')">kill</button>
      </td>
    </tr>`;
  }
  html += '</tbody></table>';
  root.innerHTML = html;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


async def _sse_stream(
    writer: asyncio.StreamWriter,
    registry: SessionRegistry,
    sid: str,
) -> None:
    """Stream session output as SSE events."""
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Access-Control-Allow-Origin: *\r\n"
        b"X-Accel-Buffering: no\r\n"
        b"\r\n"
    )
    await writer.drain()

    def send(event: str, data: dict) -> bytes:
        payload = json.dumps(data)
        return f"event: {event}\ndata: {payload}\n\n".encode()

    try:
        session = registry.get(sid)
    except ValueError:
        writer.write(send("done", {"error": "session not found"}))
        await writer.drain()
        return

    # Replay existing chunks as history first (typed chunks with kind)
    all_chunks: list[dict] = list(getattr(session.adapter, "_all_chunks", []))
    for chunk in all_chunks:
        writer.write(send("chunk", chunk))
    if all_chunks:
        await writer.drain()

    last_len = len(all_chunks)

    # Stream new chunks as they arrive
    while True:
        try:
            status = await session.adapter.status(sid)
        except Exception:
            writer.write(send("done", {}))
            await writer.drain()
            return

        writer.write(send("status", {"status": status}))

        # Flush any new typed chunks
        current_chunks: list[dict] = list(getattr(session.adapter, "_all_chunks", []))
        if len(current_chunks) > last_len:
            for chunk in current_chunks[last_len:]:
                writer.write(send("chunk", chunk))
            last_len = len(current_chunks)

        await writer.drain()

        if status in ("done", "error"):
            writer.write(send("done", {}))
            await writer.drain()
            return

        await asyncio.sleep(0.15)


async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    registry: SessionRegistry,
) -> None:
    try:
        raw = await asyncio.wait_for(reader.read(4096), timeout=5)
        req = raw.decode(errors="replace")
        first = req.split("\r\n")[0]
        method, path, *_ = (first + " HTTP/1.1").split()

        if path == "/" and method == "GET":
            body = _HTML.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()

        elif path == "/api/sessions" and method == "GET":
            sessions = []
            for s in registry.list():
                try:
                    status = await s.adapter.status(s.session_id)
                    output = "".join(getattr(s.adapter, "_output_buffer", []))[-2000:]
                except Exception:
                    status = "error"
                    output = ""
                sessions.append({
                    **s.summary(),
                    "status": status,
                    "initial_task": s.initial_task,
                    "output": output,
                })
            body = json.dumps({"sessions": sessions}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()

        elif path.startswith("/api/sessions/") and path.endswith("/stream") and method == "GET":
            sid = path.removeprefix("/api/sessions/").removesuffix("/stream")
            await _sse_stream(writer, registry, sid)
            return  # writer already closed inside _sse_stream path

        elif path.startswith("/api/sessions/") and method == "DELETE":
            sid = path.removeprefix("/api/sessions/")
            try:
                await registry.kill(sid)
                body = b'{"ok":true}'
            except ValueError:
                body = b'{"ok":false,"error":"not found"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()

        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()

    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def start_dashboard(port: int, registry: SessionRegistry) -> None:
    """Start the dashboard HTTP server on the given port (non-blocking)."""
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, registry),
        host="127.0.0.1",
        port=port,
    )
    addr = server.sockets[0].getsockname()
    log.info("Dashboard running at http://%s:%d", addr[0], addr[1])
    asyncio.create_task(server.serve_forever())  # noqa: RUF006
