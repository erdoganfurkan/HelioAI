// Drives the real app.js: a reply streaming for session A must land in A's view — never
// in the session the user switched to — and must still be there when A is reopened,
// without a history refetch that would wipe or duplicate it.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require('./dom_stub.js');

const fetchLog = [];
const pendingStreams = [];

function streamResponse() {
  const chunks = [];
  let resolveNext = null, closed = false;
  const reader = {
    read() {
      if (chunks.length) return Promise.resolve({ done: false, value: chunks.shift() });
      if (closed) return Promise.resolve({ done: true });
      return new Promise(r => { resolveNext = r; });
    },
  };
  const push = text => {
    const value = new TextEncoder().encode(text);
    if (resolveNext) { const r = resolveNext; resolveNext = null; r({ done: false, value }); }
    else chunks.push(value);
  };
  const close = () => { closed = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r({ done: true }); } };
  return { resp: { ok: true, body: { getReader: () => reader } }, push, close };
}

globalThis.fetch = async (url, opts = {}) => {
  fetchLog.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
  if (url === '/api/config') return { ok: true, json: async () => ({ provider: 'groq' }) };
  if (url === '/api/sessions') return { ok: true, json: async () => [] };
  if (url.startsWith('/api/sessions/')) return { ok: true, json: async () => ({ messages: [] }) };
  if (url === '/chat/stream') { const s = streamResponse(); pendingStreams.push(s); return s.resp; }
  throw new Error(`unexpected fetch ${url}`);
};

const appJs = fs.readFileSync(path.join(__dirname, '../../helioai/interfaces/web/static/app.js'), 'utf8');
// app.js is a classic script: evaluate it in this module's global scope so its
// top-level `let`/`function` declarations become reachable for the test below.
const exposed = ['sendMessage', 'newSession', 'resumeSession', 'cancelStreaming'];
new Function(appJs + '\n;globalThis.__app = {' + exposed.map(n => `${n}`).join(',') + ', get sessionId() { return sessionId; }};')();
const app = globalThis.__app;
const tick = () => new Promise(r => setTimeout(r, 0));

(async () => {
  await tick();
  const chatArea = document.getElementById('chat-area');
  const input = document.getElementById('input');
  const sse = ev => `data: ${JSON.stringify(ev)}\n\n`;

  // 1. Ask in A, start streaming.
  input.value = 'Question A';
  const p = app.sendMessage();
  await tick();
  const sidA = app.sessionId;
  const streamA = pendingStreams[0];
  assert.ok(streamA, 'A opened a stream');
  assert.equal(fetchLog.at(-1).body.session_id, sidA);
  assert.equal(fetchLog.at(-1).body.provider, 'groq', 'synced provider is sent, not the markup default');

  // 2. Switch to a new session B while A is still streaming.
  app.newSession();
  const sidB = app.sessionId;
  assert.notEqual(sidA, sidB);
  assert.equal(document.getElementById('input').disabled, false, 'B is usable while A streams');

  // 3. A's reply arrives.
  streamA.push(sse({ event: 'tool_call', data: { turn: 1, name: 'search_parameters', display: 'q' } }));
  streamA.push(sse({ event: 'reply', data: { text: 'Answer A' } }));
  streamA.push(sse({ event: 'done', data: { n_iterations: 1 } }));
  streamA.close();
  await p; await tick();

  assert.ok(!chatArea.textContent.includes('Answer A'), `A's answer must not appear in B: ${chatArea.textContent}`);
  assert.ok(!document.getElementById('ad-body').textContent.includes('search_parameters'), "A's activity must not appear in B's dock");

  // 4. Back to A: the answer is there, and no history refetch overwrote the live view.
  const fetchesBefore = fetchLog.length;
  const item = document.createElement('div');
  app.resumeSession(sidA, item);
  await tick();
  assert.ok(chatArea.textContent.includes('Question A'), 'A keeps its question');
  assert.ok(chatArea.textContent.includes('Answer A'), 'A shows the answer that streamed in the background');
  assert.ok(document.getElementById('ad-body').textContent.includes('search_parameters'), 'A keeps its activity');
  assert.equal(fetchLog.slice(fetchesBefore).filter(f => f.url.startsWith('/api/sessions/')).length, 0,
    'a session already held in memory is not refetched');
  assert.equal(document.getElementById('input').disabled, false, 'A finished, input enabled');

  // 5. Two concurrent streams: send in B while A is mid-stream, cancel only B.
  input.value = 'Question A2';
  const pA2 = app.sendMessage(); await tick();
  const streamA2 = pendingStreams[1];
  app.newSession(); // fresh C
  input.value = 'Question C';
  const pC = app.sendMessage(); await tick();
  const streamC = pendingStreams[2];
  assert.notEqual(fetchLog.at(-1).body.session_id, sidA);
  assert.equal(document.getElementById('input').disabled, true, 'C is streaming');
  app.cancelStreaming(); await tick();
  streamA2.push(sse({ event: 'reply', data: { text: 'Answer A2' } })); streamA2.close();
  streamC.close();
  await Promise.allSettled([pA2, pC]); await tick();
  assert.ok(!chatArea.textContent.includes('Answer A2'), 'cancelling C did not touch A, nor leak A into C');
  app.resumeSession(sidA, document.createElement('div')); await tick();
  assert.ok(chatArea.textContent.includes('Answer A2'), 'A2 streamed to completion in the background');

  console.log('OK web session streams');
})().catch(e => { console.error(e); process.exit(1); });
