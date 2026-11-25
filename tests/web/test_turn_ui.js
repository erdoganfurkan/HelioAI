// Drives the real app.js through one replayed turn and the server's /api/config: the
// data cards of a turn share one folding box without repeats, the answer can be copied
// and the session exported, a provider with no key is not offered, and the token field
// says what it is for.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require('./dom_stub.js');

const CONFIG = {
  provider: 'opencode',
  providers: { opencode: true, groq: false, gemini: false, azure: false, ollama: true },
  auth: true,
  dev_token: false,
};

const card = (param_id, extra = {}) => ({ event: 'artifact', data: { kind: 'parameter_card', param_id, ...extra } });
const JOURNAL = [
  { event: 'user', data: { text: 'Where was MMS1?' } },
  card('cda/MMS1_MEC_SRVY_L2_EPHT89D/mms1_mec_r_gsm', { mission: 'MMS1', start: '2019-02-27', stop: '2019-02-28' }),
  card('cda/OMNI_HRO_1MIN/Pressure', { mission: 'OMNI', start: '2019-02-27', stop: '2019-02-28' }),
  card('cda/OMNI_HRO_1MIN/BZ_GSM', { mission: 'OMNI', start: '2019-02-27', stop: '2019-02-28' }),
  // The analysis script reading the first series again: same param, no window of its own.
  card('cda/MMS1_MEC_SRVY_L2_EPHT89D/mms1_mec_r_gsm', { mission: 'MMS1_MEC_SRVY_L2_EPHT89D', n_points: 2880 }),
  // A series only the script loaded: its card carries the dataset where the mission goes.
  card('cda/WI_H0_MFI/B3GSE', { mission: 'WI_H0_MFI', n_points: 10 }),
  { event: 'reply', data: { text: 'MMS1 was in the **solar wind**.' } },
  { event: 'artifact', data: { kind: 'image', figure_paths: ['/ws/fig_0_0.png'] } },
  { event: 'done', data: { n_iterations: 2 } },
];

const authedFetches = [];
let deleteResponses = [];
globalThis.URL = {
  createObjectURL: () => 'blob:mock-object-url',
  revokeObjectURL: () => {},
};
globalThis.localStorage.setItem('helioai_dev_token', 'tok-alice');

globalThis.fetch = async (url, opts = {}) => {
  authedFetches.push({ url, method: opts.method || 'GET', headers: opts.headers || {} });
  if (url === '/api/config') return { ok: true, status: 200, json: async () => CONFIG };
  if (url === '/api/sessions') return { ok: true, status: 200, json: async () => [] };
  if (url === '/api/sessions/turn/events') return { ok: true, status: 200, json: async () => ({ events: JOURNAL }) };
  if (url === '/api/sessions/capped/events') {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        events: [
          { event: 'user', data: { text: 'Loop' } },
          { event: 'error', data: { message: 'agent loop exceeded 25 iterations' } },
        ],
      }),
    };
  }
  if (opts.method === 'DELETE' && url.startsWith('/api/sessions/')) {
    const status = deleteResponses.shift() ?? 200;
    return { ok: status >= 200 && status < 300, status };
  }
  if (url.startsWith('/figure?') || url.startsWith('/api/export?')) {
    return { ok: true, status: 200, blob: async () => ({ size: 4 }) };
  }
  throw new Error(`unexpected fetch ${url}`);
};

const appJs = fs.readFileSync(path.join(__dirname, '../../helioai/interfaces/web/static/app.js'), 'utf8');
new Function(appJs + '\n;globalThis.__app = { resumeSession, deleteSession, openDock, activeView: () => activeView };')();
const app = globalThis.__app;
const tick = () => new Promise(r => setTimeout(r, 0));

(async () => {
  await tick(); await tick();
  const $ = id => document.getElementById(id);

  // /api/config: a provider without a key is labelled and cannot be picked.
  const options = Object.fromEntries($('provider-select').options.map(o => [o.value, o]));
  assert.equal($('provider-select').value, 'opencode');
  assert.equal(options.groq.disabled, true);
  assert.ok(options.groq.textContent.includes('not configured'));
  assert.equal(options.ollama.disabled, false);
  assert.ok(!options.opencode.textContent.includes('not configured'));

  // With HELIOAI_USERS the field is the access token, and it is shown.
  assert.equal($('token-row').hidden, false);
  assert.equal($('token-label').textContent, 'Access token');

  app.resumeSession('turn', document.createElement('div'));
  await tick(); await tick();
  const chat = $('chat-area');

  const groups = chat.querySelectorAll('.data-used');
  assert.equal(groups.length, 1, 'one box for the turn');
  const cards = groups[0].querySelectorAll('.parameter-card');
  assert.equal(cards.length, 4, 'the series the script re-read is not shown twice');
  assert.equal(groups[0].querySelector('.data-used-summary').textContent, 'Data used (4)');
  assert.equal(groups[0].open, false, 'more than two cards fold once the turn is done');
  assert.ok(cards[3].textContent.includes('Dataset'), 'a dataset id is not called a mission');
  assert.ok(cards[0].textContent.includes('Mission'));

  const reply = chat.querySelector('.msg-ai');
  assert.ok(reply.querySelector('.msg-copy'), 'the answer has a copy button');
  assert.equal(chat.querySelectorAll('.turn-actions').length, 1, 'one export button, after the answer');
  assert.ok(chat.querySelector('.btn-turn').textContent.includes('notebook'));

  // Under HELIOAI_USERS auth, figures and notebook export must go through fetch() with X-Helio-Token.
  const img = chat.querySelector('.artifact-image').querySelector('img');
  assert.equal(img.src, 'blob:mock-object-url', 'authenticated figure loads via fetch+blob');
  assert.ok(
    authedFetches.some(f => f.url.startsWith('/figure?') && f.headers['X-Helio-Token'] === 'tok-alice'),
    'figure fetch carries X-Helio-Token'
  );
  chat.querySelector('.btn-turn').dispatch('click');
  await tick();
  assert.ok(
    authedFetches.some(f => f.url.startsWith('/api/export?') && f.headers['X-Helio-Token'] === 'tok-alice'),
    'notebook export fetch carries X-Helio-Token'
  );

  // Error event closes an open activity dock (e.g. capped turn ending without done).
  app.openDock(app.activeView());
  assert.equal($('activity-dock').classList.contains('collapsed'), false);
  app.resumeSession('capped', document.createElement('div'));
  await tick(); await tick();
  assert.equal($('activity-dock').classList.contains('collapsed'), true, 'error closes the activity dock');
  assert.ok(chat.querySelector('.error-banner').textContent.includes('exceeded 25 iterations'));

  // deleteSession retries a transient 409 (unwinding turn lock) and succeeds.
  deleteResponses = [409, 200];
  const delBefore = authedFetches.filter(f => f.method === 'DELETE').length;
  await app.deleteSession('turn');
  assert.equal(authedFetches.filter(f => f.method === 'DELETE').length - delBefore, 2, 'retried once on 409');

  // Persistent 409 surfaces an error banner instead of silently doing nothing.
  deleteResponses = [409, 409, 409];
  await app.deleteSession('capped');
  assert.ok(
    chat.querySelectorAll('.error-banner').some(e => e.textContent.includes('Could not delete session')),
    'persistent delete failure is shown to the user'
  );

  console.log('OK web turn ui');
})().catch(e => { console.error(e); process.exit(1); });
