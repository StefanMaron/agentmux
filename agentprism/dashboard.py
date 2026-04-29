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
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: top; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  tr:hover td { background: #161b22; }
  .status-working { color: #f0883e; }
  .status-idle    { color: #3fb950; }
  .status-done    { color: #8b949e; }
  .status-error   { color: #f85149; }
  .provider-copilot { color: #58a6ff; }
  .provider-claude  { color: #d2a8ff; }
  .provider-codex   { color: #79c0ff; }
  .task-cell { max-width: 360px; }
  .output-row td { background: #161b22; padding: 0; }
  .output-pre { padding: 10px 12px; font-size: 0.78rem; color: #8b949e; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; border-left: 3px solid #30363d; }
  .kill-btn { background: none; border: 1px solid #f85149; color: #f85149; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-family: inherit; }
  .kill-btn:hover { background: #f8514920; }
  .expand-btn { background: none; border: none; color: #58a6ff; cursor: pointer; font-size: 0.75rem; font-family: inherit; padding: 0; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
  .dot-working { background: #f0883e; animation: pulse 1.2s infinite; }
  .dot-idle    { background: #3fb950; }
  .dot-done    { background: #8b949e; }
  .dot-error   { background: #f85149; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .footer { margin-top: 20px; font-size: 0.72rem; color: #484f58; }
</style>
</head>
<body>
<h1>agentprism</h1>
<p class="subtitle">active sessions &nbsp;·&nbsp; auto-refreshes every 2s</p>
<div id="root"><p class="empty">Loading…</p></div>
<p class="footer" id="ts"></p>
<script>
const expanded = new Set();

function elapsed(iso) {
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

async function kill(sid) {
  if (!confirm('Kill session ' + sid.slice(0,8) + '?')) return;
  await fetch('/api/sessions/' + sid, {method:'DELETE'});
  refresh();
}

function toggle(sid) {
  if (expanded.has(sid)) expanded.delete(sid); else expanded.add(sid);
  refresh();
}

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
    const isExp = expanded.has(s.session_id);
    html += `<tr>
      <td title="${s.session_id}">${short}…</td>
      <td class="provider-${s.provider}">${s.provider}</td>
      <td>${s.model || 'auto'}</td>
      <td class="status-${s.status}"><span class="dot dot-${s.status}"></span>${s.status}</td>
      <td>${elapsed(s.created_at)}</td>
      <td class="task-cell" title="${s.initial_task.replace(/"/g,'&quot;')}">${s.initial_task.slice(0,80)}${s.initial_task.length>80?'…':''}</td>
      <td style="white-space:nowrap">
        <button class="expand-btn" onclick="toggle('${s.session_id}')">${isExp?'▲ hide':'▼ output'}</button>
        &nbsp;
        <button class="kill-btn" onclick="kill('${s.session_id}')">kill</button>
      </td>
    </tr>`;
    if (isExp) {
      html += `<tr class="output-row"><td colspan="7"><pre class="output-pre">${s.output ? s.output.replace(/&/g,'&amp;').replace(/</g,'&lt;') : '(no output yet)'}</pre></td></tr>`;
    }
  }
  html += '</tbody></table>';
  root.innerHTML = html;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


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

        elif path == "/api/sessions" and method == "GET":
            sessions = []
            for s in registry.list():
                try:
                    status = await s.adapter.status(s.session_id)
                    output = " ".join(getattr(s.adapter, "_output_buffer", []))[-2000:]
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

        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")

        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


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
