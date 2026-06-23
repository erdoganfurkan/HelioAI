/* HelioAI Web UI — vanilla JS, no framework */

marked.setOptions({ breaks: false, gfm: true });

let sessionId = crypto.randomUUID();
let isStreaming = false;
let abortController = null;

const chatArea      = document.getElementById('chat-area');
const input         = document.getElementById('input');
const btnSend       = document.getElementById('btn-send');
const btnCancel     = document.getElementById('btn-cancel');
const btnNew        = document.getElementById('btn-new');
const provSel       = document.getElementById('provider-select');
const sessList      = document.getElementById('session-list');
const devTokenInput = document.getElementById('dev-token-input');
const devIndicator  = document.getElementById('dev-indicator');
const activityDock  = document.getElementById('activity-dock');
const adBody        = document.getElementById('ad-body');
const adSummary     = document.getElementById('ad-summary');

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

function scrollDock() {
  adBody.scrollTop = adBody.scrollHeight;
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

// ── Dev token ────────────────────────────────────────────────────────────────

const DEV_TOKEN_KEY = 'helioai_dev_token';

function getDevToken() {
  return localStorage.getItem(DEV_TOKEN_KEY) || '';
}

// Inject the stored token as X-Helio-Token on every same-origin request so the
// nominative-auth backend (HELIOAI_USERS) accepts it. No-op when unset (local dev).
const _origFetch = window.fetch.bind(window);
window.fetch = (url, opts = {}) => {
  const token = getDevToken();
  if (token) opts.headers = { ...(opts.headers || {}), 'X-Helio-Token': token };
  return _origFetch(url, opts);
};

function updateDevIndicator() {
  const token = devTokenInput.value.trim();
  if (token) {
    devIndicator.classList.add('unlocked');
    devIndicator.title = 'Dev mode — scope guardrail bypassed';
  } else {
    devIndicator.classList.remove('unlocked');
    devIndicator.title = 'Dev mode off (restricted)';
  }
}

devTokenInput.value = getDevToken();
updateDevIndicator();
devTokenInput.addEventListener('input', () => {
  localStorage.setItem(DEV_TOKEN_KEY, devTokenInput.value.trim());
  updateDevIndicator();
});

// ── Activity dock ────────────────────────────────────────────────────────────

let _dockSteps = 0;
let _dockTools = 0;
let _dockSubagents = 0;

function resetDock() {
  adBody.innerHTML = '';
  adSummary.textContent = '';
  _dockSteps = 0;
  _dockTools = 0;
  _dockSubagents = 0;
  activityDock.classList.add('collapsed');
}

function openDock() {
  activityDock.classList.remove('collapsed');
}

function closeDock(summary) {
  adSummary.textContent = summary || '';
  activityDock.classList.add('collapsed');
}

document.getElementById('ad-header').addEventListener('click', () => {
  activityDock.classList.toggle('collapsed');
});

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
  adBody.append(row);
  scrollDock();
  _dockSteps++;
  return row;
}

function renderEvent(ev) {
  const { event, data } = ev;
  const nested = !!data.sub_agent_ctx;
  const nestCls = nested ? ' tl-nested' : '';

  if (event === 'tool_call') {
    _dockTools++;
    const args = argsStr(data.arguments);
    appendTlEvent('→', `${data.name}(${args})`, 'tl-tool-call' + nestCls);

  } else if (event === 'tool_result') {
    appendTlEvent('←', `${data.name}: ${data.summary || ''}`, 'tl-tool-result' + nestCls);

  } else if (event === 'sub_agent_start') {
    _dockSubagents++;
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
    const parts = [`✓ ${data.n_iterations} iter`];
    if (_dockTools > 0) parts.push(`${_dockTools} tools`);
    if (_dockSubagents > 0) parts.push(`${_dockSubagents} sub-agents`);
    closeDock(parts.join(' · '));
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
      img.addEventListener('click', () => openLightbox(url));
      img.onerror = () => {
        const fallback = el('div', 'figure-fallback');
        fallback.innerHTML = `⚠ Figure non accessible dans le navigateur.<br>`
          + `<a href="${url}" target="_blank" rel="noopener">Ouvrir directement</a>`
          + ` · <code>${path}</code>`;
        wrap.replaceChildren(fallback);
      };
      const fname = path.split('/').pop() || 'figure.png';
      const dlBtn = document.createElement('a');
      dlBtn.className = 'img-dl';
      dlBtn.href = url;
      dlBtn.download = fname;
      dlBtn.textContent = '↓ PNG';

      const pdfPath = path.replace(/\.png$/, '.pdf');
      const pdfUrl = `/figure?path=${encodeURIComponent(pdfPath)}`;
      const pdfBtn = document.createElement('a');
      pdfBtn.className = 'img-dl';
      pdfBtn.href = pdfUrl;
      pdfBtn.download = fname.replace(/\.png$/, '.pdf');
      pdfBtn.textContent = '↓ PDF';

      wrap.append(img, dlBtn, pdfBtn);
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

  } else if (data.kind === 'recipe_used') {
    const chip = el('div', 'artifact-recipe');
    const ref = data.reference ? ` — ${data.reference}` : '';
    chip.innerHTML = `<span class="ar-icon">📐</span><span class="ar-name">${data.name}</span>${ref}`;
    if (data.description) chip.title = data.description;
    chatArea.append(chip);
    scrollBottom();

  } else if (data.kind === 'catalog_preview') {
    const card = el('div', 'catalog-card');

    const header = el('div', 'cc-header');
    const nameSpan = el('span', 'cc-name', data.name || data.catalog_id || '');
    const typeSpan = el('span', 'cc-type', data.type || 'catalog');
    header.append(nameSpan, typeSpan);
    card.append(header);

    const chips = el('div', 'cc-chips');
    const chipDefs = [
      { label: 'Events', value: data.nb_events_total != null ? String(data.nb_events_total) : null },
      { label: 'From', value: data.survey_start || null },
      { label: 'To', value: data.survey_stop || null },
    ];
    chipDefs.forEach(({ label, value }) => {
      if (!value) return;
      const chip = document.createElement('span');
      chip.className = 'param-chip';
      chip.innerHTML = `<span class="chip-label">${label}</span>${value}`;
      chips.append(chip);
    });
    if (chips.children.length) card.append(chips);

    const sample = data.sample || [];
    if (sample.length > 0) {
      const cols = Object.keys(sample[0]).slice(0, 5);
      const table = el('table', 'cc-table');
      const thead = document.createElement('thead');
      const hrow = document.createElement('tr');
      cols.forEach(c => { const th = el('th', null, c); hrow.append(th); });
      thead.append(hrow);
      table.append(thead);
      const tbody = document.createElement('tbody');
      sample.forEach(row => {
        const tr = document.createElement('tr');
        cols.forEach(c => {
          const val = row[c] != null ? String(row[c]) : '';
          tr.append(el('td', null, val.length > 20 ? val.slice(0, 18) + '…' : val));
        });
        tbody.append(tr);
      });
      table.append(tbody);
      card.append(table);
    }

    chatArea.append(card);
    scrollBottom();

  } else if (data.kind === 'data_preview' && data.preview) {
    const pre = el('div', 'artifact-preview',
      `${data.param_id} — ${data.n_points} pts\n${data.preview}`);
    chatArea.append(pre);
    scrollBottom();
  }
}

// ── Lightbox ─────────────────────────────────────────────────────────────────

const lightbox   = document.getElementById('lightbox');
const lbImg      = document.getElementById('lb-img');
const lbDownload = document.getElementById('lb-download');
const lbPdf      = document.getElementById('lb-pdf');

function openLightbox(url) {
  lbImg.src = url;
  lbDownload.href = url;
  const pdfUrl = url.replace(/\.png(\?|$)/, '.pdf$1').replace(/path=[^&]+/, m => {
    const p = decodeURIComponent(m.slice(5)).replace(/\.png$/, '.pdf');
    return 'path=' + encodeURIComponent(p);
  });
  lbPdf.href = pdfUrl;
  lightbox.classList.add('open');
}

function closeLightbox() {
  lightbox.classList.remove('open');
  lbImg.src = '';
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
  resetDock();
  openDock();
  setStreaming(true);

  const headers = { 'Content-Type': 'application/json' };
  const devToken = getDevToken();
  if (devToken) headers['X-Helio-Dev-Token'] = devToken;

  try {
    abortController = new AbortController();
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers,
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
  resetDock();
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
  resetDock();
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
        (m.catalogs || []).forEach(c => renderArtifact(c));
        (m.code || []).forEach(c => renderArtifact(c));
        (m.recipes || []).forEach(c => renderArtifact(c));
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
document.getElementById('lb-close').addEventListener('click', closeLightbox);
document.getElementById('lb-backdrop').addEventListener('click', closeLightbox);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

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
