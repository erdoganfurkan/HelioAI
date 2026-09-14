// Minimal DOM for driving the real app.js under Node. Covers only what app.js touches;
// it is a test fixture, not a browser. Keep it dumb: any smartness here would be a
// second implementation to keep in sync with the file it is meant to exercise.
'use strict';

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach(x => this.set.add(x)); }
  remove(...c) { c.forEach(x => this.set.delete(x)); }
  toggle(c, force) {
    if (force === undefined) force = !this.set.has(c);
    force ? this.set.add(c) : this.set.delete(c);
    return force;
  }
  contains(c) { return this.set.has(c); }
}

let _nextId = 1;
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.classList = new ClassList(this);
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this._text = '';
    this._listeners = {};
    this.id = '';
    this.value = '';
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this._uid = _nextId++;
  }
  get className() { return [...this.classList.set].join(' '); }
  set className(v) { this.classList.set = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get childNodes() { return this.children; }
  get firstChild() { return this.children[0] || null; }
  get textContent() {
    return this.children.length ? this.children.map(c => c.textContent).join('') : this._text;
  }
  set textContent(v) { this.children = []; this._text = String(v); }
  get innerHTML() { return this.textContent; }
  set innerHTML(v) { this.children = []; this._text = String(v); }
  append(...nodes) {
    for (let n of nodes) {
      if (typeof n === 'string') n = new Text(n);
      if (n.parentNode) n.parentNode.children = n.parentNode.children.filter(c => c !== n);
      n.parentNode = this;
      this.children.push(n);
    }
  }
  appendChild(n) { this.append(n); return n; }
  replaceChildren(...nodes) { this.children.forEach(c => (c.parentNode = null)); this.children = []; this._text = ''; this.append(...nodes); }
  remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter(c => c !== this); this.parentNode = null; } }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'id') this.id = v; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  dispatch(type, ev = {}) { (this._listeners[type] || []).forEach(fn => fn({ type, target: this, preventDefault() {}, stopPropagation() {}, ...ev })); }
  matches(sel) {
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    return this.tagName === sel.toUpperCase();
  }
  *walk() { for (const c of this.children) { if (c instanceof Element) { yield c; yield* c.walk(); } } }
  querySelector(sel) { for (const e of this.walk()) if (e.matches(sel)) return e; return null; }
  querySelectorAll(sel) { return [...this.walk()].filter(e => e.matches(sel)); }
  get options() { return this.children.filter(c => c.tagName === 'OPTION'); }
}
class Text {
  constructor(t) { this._text = String(t); this.parentNode = null; }
  get textContent() { return this._text; }
}

const body = new Element('body');
const byId = new Map();
function mk(tag, id, parent = body) {
  const e = new Element(tag); e.id = id; byId.set(id, e); parent.append(e); return e;
}
// The ids app.js resolves at load time — mirrors index.html.
['chat-area', 'input', 'btn-send', 'btn-cancel', 'btn-new', 'session-list', 'dev-token-input',
 'dev-indicator', 'activity-dock', 'ad-body', 'ad-summary', 'ad-header', 'code-panel', 'code-content',
 'cp-close', 'lightbox', 'lb-img', 'lb-download', 'lb-pdf', 'lb-close', 'lb-backdrop']
  .forEach(id => mk('div', id));
const sel = mk('select', 'provider-select');
for (const v of ['azure', 'groq', 'gemini', 'opencode', 'ollama']) {
  const o = new Element('option'); o.value = v; o.tagName = 'OPTION'; sel.append(o);
}
const cpTitle = new Element('span'); cpTitle.className = 'cp-title'; body.append(cpTitle);

globalThis.document = {
  body,
  getElementById: id => byId.get(id) || null,
  createElement: tag => new Element(tag),
  createTextNode: t => new Text(t),
  querySelector: s => body.querySelector(s),
  querySelectorAll: s => body.querySelectorAll(s),
  addEventListener() {},
};
globalThis.window = globalThis;
globalThis.localStorage = { _m: {}, getItem(k) { return this._m[k] ?? null; }, setItem(k, v) { this._m[k] = String(v); } };
globalThis.marked = { setOptions() {}, parse: s => s };
globalThis.DOMPurify = { sanitize: s => s };
globalThis.Prism = { highlightElement() {} };
module.exports = { Element, body };
