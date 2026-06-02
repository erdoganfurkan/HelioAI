/* HelioAI Web UI — vanilla JS, no framework */

marked.setOptions({ breaks: false, gfm: true });

let sessionId = crypto.randomUUID();
let isStreaming = false;
let abortController = null;

const chatArea = document.getElementById('chat-area');
const input    = document.getElementById('input');
const btnSend  = document.getElementById('btn-send');
const btnCancel = document.getElementById('btn-cancel');
const btnNew   = document.getElementById('btn-new');
const provSel  = document.getElementById('provider-select');
const sessList = document.getElementById('session-list');

// ── Helpers ────────────────────────────────────────────────────────────────

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function scrollBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function setStreaming(on) {
  isStreaming = on;
  btnSend.style.display = on ? 'none' : '';
  btnCancel.style.display = on ? '' : 'none';
  input.disabled = on;
}

function cancelStreaming() {
  if (abortController) abortController.abort();
  setStreaming(false);
}

function argsStr(args) {
  if (!args || typeof args !== 'object') return '';
  return Object.entries(args).map(([k, v]) => {
    const s = typeof v === 'string' ? v : JSON.stringify(v);
    return `${k}=${s.length > 50 ? s.slice(0, 47) + '…' : s}`;
  }).join(', ');
}

// ── Welcome screen ──────────────────────────────────────────────────────────

const SUGGESTED_PROMPTS = [
  'Solar wind speed and density from ACE, 2005-01-17 to 2005-01-18, with a plot',
  'Find an interplanetary shock in WIND data around 2004-11-07 and compute θ_Bn',
  'Compare the IMF Bz between ACE and Cluster for 2003-10-29',
  'Compute the plasma beta in the magnetosheath: B=20 nT, n=15 cm⁻³, T=200 eV',
];

function renderWelcome() {
  const wrap = el('div', 'welcome');
  const title = el('div', 'welcome-title', 'What do you want to explore?');
  const sub = el('div', 'welcome-sub', '70+ missions · 83k parameters · sandboxed Python analysis');
  const grid = el('div', 'suggested-prompts');
  SUGGESTED_PROMPTS.forEach(prompt => {
    const btn = el('button', 'suggested-prompt', prompt);
    btn.addEventListener('click', () => {
      input.value = prompt;
      sendMessage();
    });
    grid.append(btn);
  });
  wrap.append(title, sub, grid);
  chatArea.append(wrap);
}

// ── Event rendering ─────────────────────────────────────────────────────────

function appendTlEvent(iconText, text, extraClass) {
  const row = el('div', 'tl-event ' + (extraClass || ''));
  const icon = el('span', 'tl-icon', iconText);
  const span = el('span', 'tl-text', text);
  row.append(icon, span);
  chatArea.append(row);
  scrollBottom();
  return row;
}

function renderEvent(ev) {
  const { event, data } = ev;
  const nested = !!data.sub_agent_ctx;
  const nestCls = nested ? ' tl-nested' : '';

  if (event === 'tool_call') {
    const args = argsStr(data.arguments);
    appendTlEvent('→', `${data.name}(${args})`, 'tl-tool-call' + nestCls);

  } else if (event === 'tool_result') {
    appendTlEvent('←', `${data.name}: ${data.summary || ''}`, 'tl-tool-result' + nestCls);

  } else if (event === 'sub_agent_start') {
    appendTlEvent('⚡', `spawning ${data.role}…`, 'tl-subagent');

  } else if (event === 'sub_agent_end') {
    const summary = (data.summary || '').slice(0, 100);
    const icon = data.error ? '✗' : '✓';
    appendTlEvent(icon, `${data.role}: ${data.error || summary}`, 'tl-subagent');

  } else if (event === 'skill_loaded') {
    appendTlEvent('📖', `skill: ${data.name}`, 'tl-skill' + nestCls);

  } else if (event === 'artifact') {
    renderArtifact(data);

  } else if (event === 'reply') {
    const bubble = el('div', 'msg-ai');
    bubble.innerHTML = marked.parse(data.text || '');
    chatArea.append(bubble);
    scrollBottom();

  } else if (event === 'done') {
    const badge = el('div', 'done-badge', `${data.n_iterations} iteration(s)`);
    chatArea.append(badge);
    scrollBottom();
    loadHistory();

  } else if (event === 'error') {
    const banner = el('div', 'error-banner', `Error: ${data.message}`);
    chatArea.append(banner);
    scrollBottom();
  }
}

function renderArtifact(data) {
  console.log('[HelioAI] artifact event:', data);
  if (data.kind === 'image' && data.figure_paths && data.figure_paths.length > 0) {
    data.figure_paths.forEach(path => {
      const url = `/figure?path=${encodeURIComponent(path)}`;
      const wrap = el('div', 'artifact-image');
      const img = document.createElement('img');
      img.src = url;
      img.alt = 'Figure';
      img.onerror = () => {
        const fallback = el('div', 'figure-fallback');
        fallback.innerHTML = `⚠ Figure non accessible dans le navigateur.<br>`
          + `<a href="${url}" target="_blank" rel="noopener">Ouvrir directement</a>`
          + ` · <code>${path}</code>`;
        wrap.replaceChildren(fallback);
      };
      wrap.append(img);
      chatArea.append(wrap);
    });
    scrollBottom();
  } else if (data.kind === 'parameter_card') {
    const card = el('div', 'parameter-card');

    const header = document.createElement('div');
    const idSpan = el('span', 'pc-id', data.param_id || '');
    header.append(idSpan);
    if (data.name) {
      const nameSpan = el('span', 'pc-name', data.name);
      header.append(nameSpan);
    }
    card.append(header);

    const chips = el('div', 'pc-chips');
    const chipDefs = [
      { label: 'Mission', value: data.mission },
      { label: 'Instrument', value: data.instrument },
      { label: 'Units', value: data.units },
      { label: 'Cadence', value: data.cadence },
      { label: 'Frame', value: data.coord_sys || null },
      { label: 'Components', value: (data.components || []).join(', ') || null },
      { label: 'Points', value: data.n_points != null ? String(data.n_points) : null },
    ];
    chipDefs.forEach(({ label, value }) => {
      if (!value) return;
      const chip = document.createElement('span');
      chip.className = 'param-chip';
      chip.innerHTML = `<span class="chip-label">${label}</span>${value}`;
      chips.append(chip);
    });
    if (chips.children.length) card.append(chips);

    if (data.start && data.stop) {
      const period = el('div', 'pc-period', `${data.start}  →  ${data.stop}`);
      card.append(period);
    }

    chatArea.append(card);
    scrollBottom();
  } else if (data.kind === 'code' && data.code_path) {
    const chip = el('div', 'artifact-code');
    const lines = data.n_lines != null ? ` · ${data.n_lines} lines` : '';
    chip.innerHTML = `<span class="ac-icon">📄</span>${data.name || 'code.py'}${lines}`;
    chip.addEventListener('click', () => openCodePanel(data.code_path, data.name));
    chatArea.append(chip);
    scrollBottom();

  } else if (data.kind === 'data_preview' && data.preview) {
    const pre = el('div', 'artifact-preview',
      `${data.param_id} — ${data.n_points} pts\n${data.preview}`);
    chatArea.append(pre);
    scrollBottom();
  }
}

// ── Code panel ───────────────────────────────────────────────────────────────

async function openCodePanel(path, name) {
  const content = document.getElementById('code-content');
  document.querySelector('.cp-title').textContent = name || 'Generated code';
  content.removeAttribute('data-highlighted');
  content.textContent = 'Loading…';
  document.getElementById('code-panel').classList.add('open');
  try {
    const r = await fetch(`/code?path=${encodeURIComponent(path)}`);
    content.textContent = r.ok ? await r.text() : `⚠ Code non accessible (${r.status})`;
   if (r.ok) {
      content.className = 'language-python';
      Prism.highlightElement(content);
    }
  } catch (e) {
    content.textContent = `⚠ ${e.message}`;
  }
}

// ── SSE streaming ───────────────────────────────────────────────────────────

async function sendMessage() {
  const text = input.value.trim();
  if (!text || isStreaming) return;

  document.querySelector('.welcome')?.remove();
  const userBubble = el('div', 'msg-user', text);
  chatArea.append(userBubble);
  input.value = '';
  input.style.height = 'auto';
  scrollBottom();
  setStreaming(true);

  try {
    abortController = new AbortController();
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId, provider: provSel.value }),
      signal: abortController.signal,
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            renderEvent(JSON.parse(line.slice(6)));
          } catch { /* ignore parse errors */ }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      const banner = el('div', 'error-banner', `Connection error: ${err.message}`);
      chatArea.append(banner);
      scrollBottom();
    }
  } finally {
    abortController = null;
    setStreaming(false);
  }
}

// ── Session management ───────────────────────────────────────────────────────

function closeCodePanel() {
  document.getElementById('code-panel').classList.remove('open');
}

function newSession() {
  sessionId = crypto.randomUUID();
  chatArea.innerHTML = '';
  closeCodePanel();
  document.querySelectorAll('.session-item').forEach(i => i.classList.remove('active'));
  renderWelcome();
}

async function loadHistory() {
  try {
    const resp = await fetch('/api/sessions');
    const sessions = await resp.json();
    sessList.innerHTML = '';
    if (!sessions.length) {
      sessList.append(el('span', 'empty', 'No sessions yet'));
      return;
    }
    sessions.forEach(s => {
      const item = el('div', 'session-item');
      if (s.session_id === sessionId) item.classList.add('active');
      const preview = el('div', 's-preview', s.first_message || '(empty)');
      const meta = el('div', 's-meta', `${s.updated_at.slice(0, 16).replace('T', ' ')} · ${s.n_messages} msgs`);
      const btnExport = el('button', 'btn-export', '↓');
      btnExport.title = 'Export as reproducible notebook (.ipynb)';
      btnExport.addEventListener('click', e => {
        e.stopPropagation();
        exportSession(s.session_id);
      });
      const btnDel = el('button', 'btn-delete', '×');
      btnDel.title = 'Delete session';
      btnDel.addEventListener('click', async e => {
        e.stopPropagation();
        await deleteSession(s.session_id);
      });
      item.append(preview, meta, btnExport, btnDel);
      item.addEventListener('click', () => resumeSession(s.session_id, item));
      sessList.append(item);
    });
  } catch { /* sidebar is non-critical */ }
}

function exportSession(sid) {
  const a = document.createElement('a');
  a.href = `/api/export?session_id=${encodeURIComponent(sid)}`;
  a.download = '';
  document.body.append(a);
  a.click();
  a.remove();
}

async function deleteSession(sid) {
  try {
    await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
    if (sid === sessionId) newSession();
    await loadHistory();
  } catch { /* non-critical */ }
}

async function resumeSession(sid, itemEl) {
  sessionId = sid;
  chatArea.innerHTML = '';
  document.querySelector('.welcome')?.remove();
  closeCodePanel();
  document.querySelectorAll('.session-item').forEach(i => i.classList.remove('active'));
  itemEl.classList.add('active');

  try {
    const resp = await fetch(`/api/sessions/${sid}/messages`);
    const data = await resp.json();
    const messages = data.messages || data;
    messages.forEach(m => {
      if (m.role === 'user') {
        chatArea.append(el('div', 'msg-user', m.content));
      } else if (m.role === 'assistant' && m.content) {
        (m.cards || []).forEach(c => renderArtifact(c));
        (m.code || []).forEach(c => renderArtifact(c));
        if (m.figures && m.figures.length > 0) {
          renderArtifact({ kind: 'image', figure_paths: m.figures });
        }
        const div = el('div', 'msg-ai');
        div.innerHTML = marked.parse(m.content);
        chatArea.append(div);
      }
    });
    scrollBottom();
  } catch { /* non-critical */ }
}

// ── Event wiring ─────────────────────────────────────────────────────────────

btnSend.addEventListener('click', sendMessage);
btnCancel.addEventListener('click', cancelStreaming);
btnNew.addEventListener('click', newSession);
document.getElementById('cp-close').addEventListener('click',
  () => document.getElementById('code-panel').classList.remove('open'));

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
});

// Init
loadHistory();
renderWelcome();
