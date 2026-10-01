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
  { event: 'done', data: { n_iterations: 2 } },
];

globalThis.fetch = async (url) => {
  if (url === '/api/config') return { ok: true, status: 200, json: async () => CONFIG };
  if (url === '/api/sessions') return { ok: true, status: 200, json: async () => [] };
  if (url === '/api/sessions/turn/events') return { ok: true, status: 200, json: async () => ({ events: JOURNAL }) };
  throw new Error(`unexpected fetch ${url}`);
};

const appJs = fs.readFileSync(path.join(__dirname, '../../helioai/interfaces/web/static/app.js'), 'utf8');
new Function(appJs + '\n;globalThis.__app = { resumeSession };')();
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

  console.log('OK web turn ui');
})().catch(e => { console.error(e); process.exit(1); });
