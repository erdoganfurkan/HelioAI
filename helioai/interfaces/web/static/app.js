/* HelioAI Web UI — vanilla JS, no framework */

marked.setOptions({ breaks: false, gfm: true });

// One view per session. A reply streams into the view of the session that asked for
// it, whether or not that view is the one on screen: switching sessions mid-stream used
// to render A's answer into B, because everything appended to the single chat area.
// The view is kept when the user navigates away, so coming back shows what arrived.
const views = new Map();
let activeView = null;
let sessionId = null;

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
  btnSend.style.display = on ? 'none' : '';
  btnCancel.style.display = on ? '' : 'none';
  input.disabled = on;
}

function cancelStreaming() {
  if (activeView && activeView.abort) activeView.abort.abort();
  setStreaming(false);
}

// ── Session views ───────────────────────────────────────────────────────────

function createView(sid) {
  const view = {
    sid,
    chat: el('div', 'session-view'),
    dock: el('div', 'ad-session'),
    summary: '',
    dockOpen: false,
    steps: 0, tools: 0, subagents: 0,
    streaming: false,
    abort: null,
  };
  views.set(sid, view);
  return view;
}

function isActive(view) {
  return view === activeView;
}

function mountView(view) {
  activeView = view;
  sessionId = view.sid;
  chatArea.replaceChildren(view.chat);
  adBody.replaceChildren(view.dock);
  adSummary.textContent = view.summary;
  activityDock.classList.toggle('collapsed', !view.dockOpen);
  setStreaming(view.streaming);
  document.querySelectorAll('.session-item').forEach(i =>
    i.classList.toggle('active', i.dataset.sid === view.sid));
  closeCodePanel();
  scrollBottom();
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

function resetDock(view) {
  view.dock.replaceChildren();
  view.summary = '';
  view.steps = 0;
  view.tools = 0;
  view.subagents = 0;
  view.dockOpen = false;
  if (isActive(view)) {
    adSummary.textContent = '';
    activityDock.classList.add('collapsed');
  }
}

function openDock(view) {
  view.dockOpen = true;
  if (isActive(view)) activityDock.classList.remove('collapsed');
}

function closeDock(view, summary) {
  view.summary = summary || '';
  view.dockOpen = false;
  if (isActive(view)) {
    adSummary.textContent = view.summary;
    activityDock.classList.add('collapsed');
  }
}

document.getElementById('ad-header').addEventListener('click', () => {
  const open = !activityDock.classList.toggle('collapsed');
  if (activeView) activeView.dockOpen = open;
});

// ── Welcome screen ──────────────────────────────────────────────────────────

const SUGGESTED_PROMPTS = [
  'Solar wind speed and density from ACE, 2005-01-17 to 2005-01-18, with a plot',
  'Find an interplanetary shock in WIND data around 2004-11-07 and compute θ_Bn',
  'List the strongest ICMEs seen by Wind in 2023 (HELIO4CAST ICMECAT catalog)',
  'Superposed epoch of |B| over the 2023 Wind ICMEs from the HELIO4CAST catalog',
  'Where was MMS1 on 2019-02-27 12:00? Plot its GSM position against the Shue magnetopause and the Jelinek bow shock',
  'Find recent papers on electron-scale magnetic reconnection at MMS, with bibcodes',
  'Compute θ_Bn for the 2008-01-01 WIND shock, then find papers reporting similar quasi-perpendicular shocks',
  'Compute the plasma beta in the magnetosheath: B=20 nT, n=15 cm⁻³, T=200 eV',
];

function renderWelcome(view) {
  const wrap = el('div', 'welcome');
  const title = el('div', 'welcome-title', 'What do you want to explore?');
  const sub = el('div', 'welcome-sub', '70+ missions · 83k parameters · event catalogs · literature search · sandboxed Python analysis');
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
  view.chat.append(wrap);
}

// ── Event rendering ─────────────────────────────────────────────────────────

function appendTlEvent(view, iconText, text, extraClass) {
  const row = el('div', 'tl-event ' + (extraClass || ''));
  const icon = el('span', 'tl-icon', iconText);
  const span = el('span', 'tl-text', text);
  row.append(icon, span);
  view.dock.append(row);
  if (isActive(view)) scrollDock();
  view.steps++;
  return row;
}

function renderEvent(view, ev) {
  const { event, data } = ev;
  const nested = !!data.sub_agent_ctx;
  const nestCls = nested ? ' tl-nested' : '';

  if (event === 'tool_call') {
    view.tools++;
    // `display` is built server-side by core/event_display.py so this timeline, the CLI
    // and the Jupyter magic word things identically. The argsStr fallback keeps replays
    // of sessions recorded before that field existed readable.
    const detail = data.display !== undefined && data.display !== null
      ? data.display
      : argsStr(data.arguments);
    appendTlEvent(view, '→', detail ? `${data.name} ${detail}` : data.name, 'tl-tool-call' + nestCls);

  } else if (event === 'tool_result') {
    appendTlEvent(view, '←', `${data.name}: ${data.display || data.summary || ''}`,
                  'tl-tool-result' + nestCls);

  } else if (event === 'sub_agent_start') {
    view.subagents++;
    appendTlEvent(view, '⚡', `spawning ${data.role}…`, 'tl-subagent');

  } else if (event === 'sub_agent_end') {
    const summary = (data.summary || '').slice(0, 100);
    const icon = data.error ? '✗' : '✓';
    const row = appendTlEvent(view, icon, `${data.role}: ${data.error || summary}`, 'tl-subagent');
    // The measured values, not the prose: one line per finding, capped like the CLI.
    const findings = Object.entries(data.findings || {});
    if (findings.length) {
      const list = el('ul', 'tl-findings');
      for (const [name, f] of findings.slice(0, 8)) {
        let text = `${name} = ${f.value}${f.units ? ' ' + f.units : ''}`;
        if (f.min !== undefined && f.min !== null) text += ` [${f.min}, ${f.max}]`;
        list.append(el('li', '', text));
      }
      if (findings.length > 8) list.append(el('li', '', `… and ${findings.length - 8} more`));
      row.append(list);
    }

  } else if (event === 'skill_loaded') {
    appendTlEvent(view, '📖', `skill: ${data.name}`, 'tl-skill' + nestCls);

  } else if (event === 'artifact') {
    renderArtifact(view, data);

  } else if (event === 'plan') {
    // The loop has always emitted this and the SSE has always forwarded it; only the
    // web client dropped it, so a plan showed up in the CLI and the notebook but never
    // in the browser. Rendered in the chat, not the timeline: it is addressed to the
    // reader, not a trace of what the agent did.
    renderPlan(view, data);

  } else if (event === 'figure_review') {
    renderFigureReview(view, data.text);

  } else if (event === 'recipe_bypassed') {
    // Advisory, not a banner: exports resemble a computation with a calibrated recipe,
    // either never loaded or loaded and not actually called — worth a glance, not an alarm.
    const suffix = { shallow_use: ' (outputs missing)', not_called: ' (never called)' };
    const names = (data.recipes || [])
      .map(r => `${r.recipe || r}${suffix[r.reason] || ''}`)
      .join(', ');
    appendTlEvent(view, '⚠', `recipe check — ${names}, verify the exported numbers`, 'tl-issue');

  } else if (event === 'provenance') {
    renderProvenance(view, data);

  } else if (event === 'invalid_ids') {
    // A sub-agent quoting parameter ids that exist in no catalogue is the most
    // damaging thing it can produce, so this one is a banner, not a timeline line.
    const box = el('div', 'invalid-ids');
    box.append(el('div', 'invalid-ids-title', '⚠️ ids not found in the catalogue — do not use these:'));
    const ul = el('ul');
    for (const id of data.ids || []) ul.append(el('li', null, id));
    box.append(ul);
    view.chat.append(box);
    if (isActive(view)) scrollBottom();

  } else if (event === 'reply') {
    const bubble = el('div', 'msg-ai');
    bubble.innerHTML = DOMPurify.sanitize(marked.parse(data.text || ''));
    view.chat.append(bubble);
    if (isActive(view)) scrollBottom();

  } else if (event === 'done') {
    const parts = [`✓ ${data.n_iterations} iter`];
    if (view.tools > 0) parts.push(`${view.tools} tools`);
    if (view.subagents > 0) parts.push(`${view.subagents} sub-agents`);
    closeDock(view, parts.join(' · '));
    loadHistory();

  } else if (event === 'error') {
    const banner = el('div', 'error-banner', `Error: ${data.message}`);
    view.chat.append(banner);
    if (isActive(view)) scrollBottom();
  }
}

function renderFigureReview(view, text) {
  const ok = (text || '').startsWith('OK');
  const row = appendTlEvent(view, ok ? '✓' : '⚠', text || '', ok ? 'tl-ok' : 'tl-issue');
  row.title = text || '';
  return row;
}

function renderProvenance(view, data) {
  // A timeline row, not a banner: it annotates the answer, it does not overrule it.
  // Collapsed by default — the counts are the signal, the list is for whoever doubts it.
  const flagged = (data.contradicted || 0) + (data.unsourced || 0);
  const summary = `📐 provenance — ${data.matched || 0} traced, ${data.contradicted || 0} contradicted, ` +
                  `${data.unsourced || 0} unsourced, ${data.derived || 0} derived`;
  const row = appendTlEvent(view, flagged ? '⚠' : '✓', summary, flagged ? 'tl-issue' : 'tl-ok');
  const details = data.details || [];
  if (!details.length) return row;

  const box = el('details', 'provenance-details');
  box.append(el('summary', null, `${details.length} number${details.length > 1 ? 's' : ''} to check`));
  const ul = el('ul');
  for (const d of details) {
    const li = el('li');
    li.append(el('span', 'provenance-status', d.status));
    li.append(document.createTextNode(' ' + d.text + (d.name ? ` — the session computed ${d.name}` : '')));
    ul.append(li);
  }
  box.append(ul);
  row.append(box);
  return row;
}

function renderPlan(view, data) {
  if (!data || !(data.steps || []).length) return;
  const card = el('div', 'plan-card');
  card.append(el('div', 'plan-title', `🗺 ${data.title || 'Plan'}`));
  const ol = document.createElement('ol');
  ol.className = 'plan-steps';
  (data.steps || []).forEach(s => {
    const li = document.createElement('li');
    li.textContent = s.description || '';
    if (s.tool) {
      const sp = document.createElement('span');
      sp.className = 'plan-tool';
      sp.textContent = s.tool;
      li.append(' ', sp);
    }
    ol.append(li);
  });
  card.append(ol);
  view.chat.append(card);
  if (isActive(view)) scrollBottom();
}

function renderArtifact(view, data) {
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
      view.chat.append(wrap);
    });
    if (isActive(view)) scrollBottom();
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
    const comps = data.components || [];
    const COMP_MAX = 6;
    const compVal = comps.length
      ? (comps.length > COMP_MAX
          ? `${comps.slice(0, COMP_MAX).join(', ')} +${comps.length - COMP_MAX} more`
          : comps.join(', '))
      : null;
    const chipDefs = [
      { label: 'Mission', value: data.mission },
      { label: 'Instrument', value: data.instrument },
      { label: 'Units', value: data.units },
      { label: 'Cadence', value: data.cadence },
      { label: 'Frame', value: data.coord_sys || null },
      { label: 'Components', value: compVal, title: comps.length > COMP_MAX ? comps.join(', ') : null },
      { label: 'Points', value: data.n_points != null ? String(data.n_points) : null },
    ];
    chipDefs.forEach(({ label, value, title }) => {
      if (!value) return;
      const chip = document.createElement('span');
      chip.className = 'param-chip';
      const lbl = document.createElement('span');
      lbl.className = 'chip-label';
      lbl.textContent = label;
      chip.append(lbl);
      chip.append(String(value));
      if (title) chip.title = title;
      chips.append(chip);
    });
    if (chips.children.length) card.append(chips);

    if (data.start && data.stop) {
      const period = el('div', 'pc-period', `${data.start}  →  ${data.stop}`);
      card.append(period);
    }

    view.chat.append(card);
    if (isActive(view)) scrollBottom();
  } else if (data.kind === 'code' && data.code_path) {
    const chip = el('div', 'artifact-code');
    const lines = data.n_lines != null ? ` · ${data.n_lines} lines` : '';
    { const ico = document.createElement('span'); ico.className = 'ac-icon'; ico.textContent = '\u{1F4C4}'; chip.append(ico); }
    chip.append(document.createTextNode((data.name || 'code.py') + lines));
    chip.addEventListener('click', () => openCodePanel(data.code_path, data.name));
    view.chat.append(chip);
    if (isActive(view)) scrollBottom();

  } else if (data.kind === 'recipe_used') {
    const chip = el('div', 'artifact-recipe');
    const ref = data.reference ? ` — ${data.reference}` : '';
    { const ico = document.createElement('span'); ico.className = 'ar-icon'; ico.textContent = '\u{1F4D0}'; chip.append(ico); }
    { const nm = document.createElement('span'); nm.className = 'ar-name'; nm.textContent = data.name || ''; chip.append(nm); }
    chip.append(document.createTextNode(ref));
    if (data.description) chip.title = data.description;
    view.chat.append(chip);
    if (isActive(view)) scrollBottom();

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
      const lbl = document.createElement('span');
      lbl.className = 'chip-label';
      lbl.textContent = label;
      chip.append(lbl);
      chip.append(String(value));
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

    view.chat.append(card);
    if (isActive(view)) scrollBottom();
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
  const view = activeView;
  const text = input.value.trim();
  if (!text || !view || view.streaming) return;

  view.chat.querySelector('.welcome')?.remove();
  view.chat.append(el('div', 'msg-user', text));
  input.value = '';
  input.style.height = 'auto';
  scrollBottom();
  resetDock(view);
  openDock(view);
  view.streaming = true;
  setStreaming(true);

  const headers = { 'Content-Type': 'application/json' };
  const devToken = getDevToken();
  if (devToken) headers['X-Helio-Dev-Token'] = devToken;

  // Until /api/config has answered, the selector shows the markup's first option, not
  // the server's setting; sending it would override a provider the user never chose.
  const provider = (provSel.dataset.synced || provSel.dataset.touched) ? provSel.value : null;

  try {
    view.abort = new AbortController();
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify({ message: text, session_id: view.sid, provider }),
      signal: view.abort.signal,
    });

    if (!resp.ok) {
      throw new Error(resp.status === 409
        ? 'a reply is already streaming for this session — wait for it to finish'
        : `HTTP ${resp.status}`);
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
          let ev = null;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          renderEvent(view, ev);
        }
      }
    }
  } catch (err) {
    const banner = el('div', 'error-banner',
      err.name === 'AbortError' ? 'Cancelled.' : `Connection error: ${err.message}`);
    view.chat.append(banner);
    if (isActive(view)) scrollBottom();
  } finally {
    view.abort = null;
    view.streaming = false;
    if (isActive(view)) setStreaming(false);
  }
}

// ── Session management ───────────────────────────────────────────────────────

function closeCodePanel() {
  document.getElementById('code-panel').classList.remove('open');
}

function newSession() {
  const view = createView(crypto.randomUUID());
  renderWelcome(view);
  mountView(view);
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
      item.dataset.sid = s.session_id;
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
    const view = views.get(sid);
    if (view && view.abort) view.abort.abort();
    views.delete(sid);
    if (sid === sessionId) newSession();
    await loadHistory();
  } catch { /* non-critical */ }
}

async function resumeSession(sid, itemEl) {
  // A session already held in memory — streaming or finished — is shown as it is.
  // Refetching would wipe a reply that arrived while the user was elsewhere.
  const held = views.get(sid);
  if (held) {
    mountView(held);
    if (itemEl) itemEl.classList.add('active');
    return;
  }
  const view = createView(sid);
  mountView(view);
  if (itemEl) itemEl.classList.add('active');

  try {
    const resp = await fetch(`/api/sessions/${sid}/messages`);
    const data = await resp.json();
    const messages = data.messages || data;
    messages.forEach(m => {
      if (m.role === 'user') {
        view.chat.append(el('div', 'msg-user', m.content));
      } else if (m.role === 'system') {
        view.chat.append(el('div', 'msg-system', m.content));
      } else if (m.role === 'assistant') {
        // Artifacts come before the text even when the text is empty: a turn cut short
        // still produced its figure and its script, and the replay must show them.
        (m.cards || []).forEach(c => renderArtifact(view, c));
        (m.catalogs || []).forEach(c => renderArtifact(view, c));
        (m.code || []).forEach(c => renderArtifact(view, c));
        (m.recipes || []).forEach(c => renderArtifact(view, c));
        if (m.figures && m.figures.length > 0) {
          renderArtifact(view, { kind: 'image', figure_paths: m.figures });
        }
        if (m.content) {
          const div = el('div', 'msg-ai');
          div.innerHTML = DOMPurify.sanitize(marked.parse(m.content));
          view.chat.append(div);
        }
      }
    });
    if (isActive(view)) scrollBottom();
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
// The <select> lists providers in markup order, so without this the browser sent
// whichever came first — azure — and silently overrode the server's own setting.
async function syncProvider() {
  try {
    const r = await fetch('/api/config');
    if (!r.ok) return;
    const { provider } = await r.json();
    if (provider && [...provSel.options].some(o => o.value === provider)) {
      provSel.value = provider;
    }
    provSel.dataset.synced = '1';
  } catch { /* leave the markup default; a failed probe must not block the UI */ }
}

provSel.addEventListener('change', () => { provSel.dataset.touched = '1'; });

syncProvider();
loadHistory();
newSession();
