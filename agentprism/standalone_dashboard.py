"""Standalone aggregator dashboard for all running agentprism instances.

Discovers instances via ``~/.agentprism/*.json`` lockfiles (see
:mod:`agentprism.lockfile`), fans out HTTP calls to each instance's auto-API,
and serves a unified HTML/JSON view grouped by project (cwd basename).

Pure stdlib — uses ``asyncio`` for the server and ``urllib.request`` for the
fan-out HTTP client (run in a worker thread to keep the loop responsive).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from agentprism.lockfile import discover

log = logging.getLogger("agentprism.standalone_dashboard")

_HTTP_TIMEOUT = 2.0


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentprism — global</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: ui-monospace, monospace; background: #0d1117; color: #e6edf3; padding: 24px; }
  h1 { font-size: 1.1rem; color: #58a6ff; margin-bottom: 4px; }
  .subtitle { font-size: 0.75rem; color: #8b949e; margin-bottom: 24px; }
  .empty { color: #8b949e; font-size: 0.85rem; padding: 16px 0; }
  .group { margin-bottom: 28px; border: 1px solid #21262d; border-radius: 6px; overflow: hidden; }
  .group-header {
    background: #161b22; padding: 10px 14px;
    display: flex; align-items: center; gap: 14px;
    border-bottom: 1px solid #21262d;
    font-size: 0.82rem;
  }
  .group-header .project { color: #d2a8ff; font-weight: 600; }
  .group-header .cwd { color: #8b949e; font-size: 0.72rem; }
  .group-header .pid { color: #79c0ff; font-size: 0.72rem; margin-left: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 500; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: top; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  tr:hover td { background: #161b22; }
  tr:last-child td { border-bottom: none; }
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
<h1>agentprism — global dashboard</h1>
<p class="subtitle">all running instances &nbsp;·&nbsp; auto-refreshes every 2s</p>
<div id="root"><p class="empty">Loading…</p></div>
<p class="footer" id="ts"></p>

<div id="terminal-overlay">
  <div id="terminal-box">
    <div id="terminal-header">
      <span class="dot dot-working" id="terminal-dot"></span>
      <span id="terminal-title">session</span>
      <span id="terminal-provider"></span>
      <button class="close-btn" onclick="closeTerminal()">x</button>
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

document.getElementById('terminal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('terminal-overlay')) closeTerminal();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeTerminal(); });

async function refresh() {
  let payload;
  try {
    const res = await fetch('/api/sessions');
    payload = await res.json();
  } catch (e) {
    document.getElementById('root').innerHTML = '<p class="empty">discovery failed.</p>';
    return;
  }
  const groups = payload.groups || [];
  const root = document.getElementById('root');
  document.getElementById('ts').textContent = 'last update ' + new Date().toLocaleTimeString();

  if (!groups.length) {
    root.innerHTML = '<p class="empty">No running agentprism instances found.</p>';
    return;
  }

  let html = '';
  for (const g of groups) {
    html += `<div class="group">
      <div class="group-header">
        <span class="project">${esc(g.project || '(unknown)')}</span>
        <span class="cwd" title="${esc(g.cwd)}">${esc(g.cwd)}</span>
        <span class="pid">pid ${g.pid} · port ${g.port}</span>
      </div>`;
    if (!g.sessions.length) {
      html += '<p class="empty" style="padding:10px 14px">No sessions.</p>';
    } else {
      html += '<table><thead><tr><th>id</th><th>provider</th><th>model</th><th>status</th><th>elapsed</th><th class="task-cell">task</th><th></th></tr></thead><tbody>';
      for (const s of g.sessions) {
        const short = s.session_id.slice(0,8);
        const isActive = activeId === s.session_id;
        const taskEsc = (s.initial_task || '').replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'");
        html += `<tr>
          <td title="${s.session_id}">${short}…</td>
          <td class="provider-${s.provider}">${s.provider}</td>
          <td>${s.model || 'auto'}</td>
          <td class="status-${s.status}"><span class="dot dot-${s.status}"></span>${s.status}</td>
          <td>${elapsed(s.created_at)}</td>
          <td class="task-cell" title="${esc(s.initial_task || '')}">${esc((s.initial_task||'').slice(0,80))}${(s.initial_task||'').length>80?'…':''}</td>
          <td style="white-space:nowrap">
            <button class="watch-btn${isActive?' active':''}" onclick="openTerminal('${s.session_id}','${s.provider}','${taskEsc}')">watch</button>
            &nbsp;
            <button class="kill-btn" onclick="kill('${s.session_id}')">kill</button>
          </td>
        </tr>`;
      }
      html += '</tbody></table>';
    }
    html += '</div>';
  }
  root.innerHTML = html;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


def _http_request_sync(
    method: str,
    url: str,
    timeout: float = _HTTP_TIMEOUT,
) -> tuple[int, bytes]:
    """Synchronous HTTP request; returns (status, body)."""
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except Exception:
        return 0, b""


async def _http_request(method: str, url: str) -> tuple[int, bytes]:
    return await asyncio.to_thread(_http_request_sync, method, url)


async def _fetch_instance_sessions(inst: dict) -> list[dict]:
    """GET /api/sessions from one instance, tagging each session with `project`."""
    url = f"http://127.0.0.1:{inst['port']}/api/sessions"
    status, body = await _http_request("GET", url)
    if status != 200 or not body:
        return []
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return []
    sessions = data.get("sessions", []) or []
    project = os.path.basename(str(inst["cwd"]).rstrip("/\\")) or str(inst["cwd"])
    for s in sessions:
        s["project"] = project
        s["instance_pid"] = inst["pid"]
        s["instance_port"] = inst["port"]
        s["instance_cwd"] = inst["cwd"]
    return sessions


async def _aggregate() -> dict[str, Any]:
    """Discover and aggregate sessions across all live instances."""
    instances = discover()
    # Fan out concurrently.
    results = await asyncio.gather(
        *(_fetch_instance_sessions(i) for i in instances),
        return_exceptions=True,
    )
    groups = []
    for inst, sessions in zip(instances, results, strict=False):
        if isinstance(sessions, Exception):
            sessions = []
        project = os.path.basename(str(inst["cwd"]).rstrip("/\\")) or str(inst["cwd"])
        groups.append(
            {
                "pid": inst["pid"],
                "port": inst["port"],
                "cwd": inst["cwd"],
                "project": project,
                "sessions": sessions,
            }
        )
    # Stable sort: by project name, then pid.
    groups.sort(key=lambda g: (g["project"].lower(), g["pid"]))
    return {"groups": groups}


async def _find_owner(sid: str) -> dict | None:
    """Find the instance that owns ``sid`` by querying each one's session list."""
    instances = discover()
    results = await asyncio.gather(
        *(_fetch_instance_sessions(i) for i in instances),
        return_exceptions=True,
    )
    for inst, sessions in zip(instances, results, strict=False):
        if isinstance(sessions, Exception):
            continue
        for s in sessions:
            if s.get("session_id") == sid:
                return inst
    return None


async def _proxy_sse(
    writer: asyncio.StreamWriter,
    inst: dict,
    sid: str,
) -> None:
    """Proxy an SSE stream from the upstream instance to the client.

    Uses a thread to read from urllib (which doesn't natively integrate with
    asyncio) and forwards bytes as they arrive. Closes when the upstream
    response ends or the client disconnects.
    """
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Access-Control-Allow-Origin: *\r\n"
        b"X-Accel-Buffering: no\r\n"
        b"\r\n"
    )
    await writer.drain()

    url = f"http://127.0.0.1:{inst['port']}/api/sessions/{sid}/stream"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def _reader_thread() -> None:
        try:
            with urllib.request.urlopen(url, timeout=None) as resp:
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        except Exception as e:
            log.debug("upstream SSE read failed: %s", e)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    import threading

    t = threading.Thread(target=_reader_thread, daemon=True)
    t.start()

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            try:
                writer.write(chunk)
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _serve_html(writer: asyncio.StreamWriter) -> None:
    body = _HTML.encode()
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    await writer.drain()


async def _serve_json(writer: asyncio.StreamWriter, payload: dict | list) -> None:
    body = json.dumps(payload).encode()
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Access-Control-Allow-Origin: *\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    await writer.drain()


async def _serve_404(writer: asyncio.StreamWriter) -> None:
    writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
    await writer.drain()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        raw = await asyncio.wait_for(reader.read(4096), timeout=5)
        req = raw.decode(errors="replace")
        first = req.split("\r\n")[0]
        method, path, *_ = (first + " HTTP/1.1").split()

        if path == "/" and method == "GET":
            await _serve_html(writer)
        elif path == "/api/sessions" and method == "GET":
            await _serve_json(writer, await _aggregate())
        elif path == "/api/instances" and method == "GET":
            await _serve_json(writer, {"instances": discover()})
        elif path.startswith("/api/sessions/") and path.endswith("/stream") and method == "GET":
            sid = path.removeprefix("/api/sessions/").removesuffix("/stream")
            inst = await _find_owner(sid)
            if inst is None:
                await _serve_404(writer)
                return
            await _proxy_sse(writer, inst, sid)
            return  # writer closed inside proxy
        elif path.startswith("/api/sessions/") and method == "DELETE":
            sid = path.removeprefix("/api/sessions/")
            inst = await _find_owner(sid)
            if inst is None:
                await _serve_json(writer, {"ok": False, "error": "not found"})
                return
            url = f"http://127.0.0.1:{inst['port']}/api/sessions/{sid}"
            status, body = await _http_request("DELETE", url)
            if status == 200:
                try:
                    await _serve_json(writer, json.loads(body.decode("utf-8")))
                except Exception:
                    await _serve_json(writer, {"ok": True})
            else:
                await _serve_json(writer, {"ok": False, "error": "upstream failed"})
        else:
            await _serve_404(writer)
    except Exception as e:
        log.debug("standalone handler error: %s", e)
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def start_standalone_dashboard(port: int = 7070) -> int:
    """Start the standalone aggregator dashboard. Returns the bound port."""
    server = await asyncio.start_server(_handle, host="127.0.0.1", port=port)
    addr = server.sockets[0].getsockname()
    bound = int(addr[1])
    log.info("standalone dashboard at http://%s:%d", addr[0], bound)
    asyncio.create_task(server.serve_forever())  # noqa: RUF006
    return bound
