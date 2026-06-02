// Automated tests for the Vermögensübersicht date filtering (getFilteredEntries):
// the YTD default, the relative presets, "Alles", and the custom Von/Bis range.

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

const startTag = '<script>';
const sIdx = html.indexOf(startTag);
const eIdx = html.indexOf('</script>', sIdx);
assert.ok(sIdx !== -1 && eIdx !== -1, 'Could not locate the inline <script> block');
const scriptSrc = html.slice(sIdx + startTag.length, eIdx);

function fakeElement() {
    return {
        value: '', textContent: '', innerHTML: '', className: '', checked: false,
        style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
        addEventListener() {}, querySelectorAll() { return []; }
    };
}

const store = new Map();
const localStorage = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k), clear: () => store.clear()
};

const chainable = new Proxy(function () {}, { get: () => chainable, apply: () => chainable });
const firebase = {
    initializeApp() {},
    firestore: Object.assign(() => chainable, { FieldValue: chainable }),
    auth: Object.assign(() => ({ onAuthStateChanged() {}, signOut() {}, signInWithPopup() { return { catch() {} }; } }),
        { GoogleAuthProvider: function () {} })
};

const sandbox = {
    firebase, localStorage, console,
    Intl, Date, Math, JSON, parseFloat, parseInt, isFinite, Array, Object, Number, String, Set, RegExp,
    alert() {}, confirm() { return true; },
    Chart: function () {},
    document: { addEventListener() {}, getElementById: () => fakeElement(), querySelector: () => fakeElement(), querySelectorAll: () => [] },
    window: {}
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
const exportLine = '\nObject.assign(globalThis, { AppState, getFilteredEntries });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { AppState, getFilteredEntries } = sandbox;

const Y = new Date().getFullYear();
const now = new Date();
const daysAgo = n => { const d = new Date(now); d.setDate(d.getDate() - n); return d.toISOString().slice(0, 10); };

function reset() {
    AppState.entries = [];
    AppState.activeTimeRange = 'ytd';
    AppState.customFrom = null;
    AppState.customTo = null;
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('YTD keeps only entries from the current calendar year', () => {
    reset();
    AppState.entries = [
        { date: `${Y}-01-05`, tr_konto: 1 },
        { date: `${Y - 1}-12-20`, tr_konto: 2 },
        { date: `${Y - 2}-06-01`, tr_konto: 3 }
    ];
    AppState.activeTimeRange = 'ytd';
    const out = getFilteredEntries();
    assert.strictEqual(out.length, 1);
    assert.strictEqual(out[0].date, `${Y}-01-05`);
});

test('"Alles" returns every entry untouched', () => {
    reset();
    AppState.entries = [{ date: `${Y}-01-05` }, { date: `${Y - 5}-01-05` }];
    AppState.activeTimeRange = 'all';
    assert.strictEqual(getFilteredEntries().length, 2);
});

test('relative preset (3m) keeps only the last three months', () => {
    reset();
    AppState.entries = [
        { date: daysAgo(10) },   // ~now
        { date: daysAgo(120) },  // ~4 months ago
        { date: daysAgo(400) }   // >1 year ago
    ];
    AppState.activeTimeRange = '3m';
    const out = getFilteredEntries();
    assert.strictEqual(out.length, 1);
    assert.strictEqual(out[0].date, daysAgo(10));
});

test('custom Von/Bis range filters inclusively between the two dates', () => {
    reset();
    AppState.entries = [
        { date: '2023-01-01' }, { date: '2023-06-01' }, { date: '2023-12-01' }
    ];
    AppState.activeTimeRange = 'custom';
    AppState.customFrom = '2023-05-01';
    AppState.customTo = '2023-07-01';
    const out = getFilteredEntries();
    assert.strictEqual(out.length, 1);
    assert.strictEqual(out[0].date, '2023-06-01');
});

test('custom range with only a Von date keeps everything from that date on', () => {
    reset();
    AppState.entries = [
        { date: '2023-01-01' }, { date: '2023-06-01' }, { date: '2023-12-01' }
    ];
    AppState.activeTimeRange = 'custom';
    AppState.customFrom = '2023-06-01';
    AppState.customTo = null;
    const out = getFilteredEntries();
    assert.strictEqual(out.length, 2);
    assert.deepStrictEqual(out.map(e => e.date), ['2023-06-01', '2023-12-01']);
});

let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
