// Automated tests for the user-defined ("custom") bank registry. Like the
// other suites, we load the real inline <script> from index.html into an
// isolated VM context and exercise the pure registry logic: slug/prefix
// generation, merging custom banks into BANKS/accounts via rebuildRegistry,
// and calculateTotals picking up custom account ids.

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
    removeItem: k => store.delete(k),
    clear: () => store.clear()
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
const exportLine = '\nObject.assign(globalThis, { DEFAULT_BANKS, Settings, rebuildRegistry, buildBanks, buildAccounts, calculateTotals, slugifyBank, uniqueBankPrefix, isBav, isRiester, getBanks: () => BANKS, getAccounts: () => accounts });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { DEFAULT_BANKS, Settings, rebuildRegistry, buildBanks, buildAccounts, calculateTotals, slugifyBank, uniqueBankPrefix, isBav, isRiester, getBanks, getAccounts } = sandbox;

// Reset persisted state and restore the registry to its built-in defaults.
function reset() { store.clear(); Settings._data = null; rebuildRegistry(); }

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('slugifyBank produces a lowercase alphanumeric id', () => {
    assert.strictEqual(slugifyBank('Scalable Capital'), 'scalablecapital');
    assert.strictEqual(slugifyBank('Trade-Republic 2!'), 'traderepublic2');
    assert.strictEqual(slugifyBank('   '), '');
});

test('uniqueBankPrefix avoids collisions with existing prefixes', () => {
    reset();
    // 'MLP' slugs to 'mlp', which already exists as a built-in → suffixed.
    assert.strictEqual(uniqueBankPrefix('MLP'), 'mlp2');
    // A fresh name keeps its plain slug.
    assert.strictEqual(uniqueBankPrefix('Scalable'), 'scalable');
});

test('custom banks are merged into BANKS by buildBanks/rebuildRegistry', () => {
    reset();
    const before = getBanks().length;
    Settings.setCustomBanks([{ prefix: 'scalable', label: 'Scalable', cbPrefix: 'scalable', color: '#fff', accounts: [{ suffix: 'depot', label: 'Depot' }] }]);
    rebuildRegistry();
    assert.strictEqual(getBanks().length, before + 1);
    assert.ok(getBanks().some(b => b.prefix === 'scalable'));
    assert.ok(getAccounts().some(a => a.id === 'scalable_depot'));
});

test('a custom bank can expose both Konto and Depot', () => {
    reset();
    Settings.setCustomBanks([{ prefix: 'neo', label: 'Neo', cbPrefix: 'neo', color: '#fff', accounts: [{ suffix: 'konto', label: 'Konto' }, { suffix: 'depot', label: 'Depot' }] }]);
    rebuildRegistry();
    const ids = getAccounts().map(a => a.id);
    assert.ok(ids.includes('neo_konto'));
    assert.ok(ids.includes('neo_depot'));
});

test('calculateTotals counts custom Konto/Depot in the right buckets', () => {
    reset();
    Settings.setCustomBanks([{ prefix: 'neo', label: 'Neo', cbPrefix: 'neo', color: '#fff', accounts: [{ suffix: 'konto', label: 'Konto' }, { suffix: 'depot', label: 'Depot' }] }]);
    rebuildRegistry();
    const t = calculateTotals({ neo_konto: 300, neo_depot: 700 });
    assert.strictEqual(t.konten, 300);
    assert.strictEqual(t.depots, 700);
    assert.strictEqual(t.total, 1000);
});

test('removing a custom bank drops its accounts from the registry', () => {
    reset();
    Settings.setCustomBanks([{ prefix: 'neo', label: 'Neo', cbPrefix: 'neo', color: '#fff', accounts: [{ suffix: 'konto', label: 'Konto' }] }]);
    rebuildRegistry();
    assert.ok(getAccounts().some(a => a.id === 'neo_konto'));
    Settings.setCustomBanks(Settings.getCustomBanks().filter(b => b.prefix !== 'neo'));
    rebuildRegistry();
    assert.ok(!getAccounts().some(a => a.id === 'neo_konto'));
    assert.strictEqual(getBanks().length, DEFAULT_BANKS.length);
});

test('a custom account can be designated as bAV or Riester', () => {
    reset();
    Settings.setCustomBanks([{ prefix: 'neo', label: 'Neo', cbPrefix: 'neo', color: '#fff', accounts: [{ suffix: 'depot', label: 'Depot' }] }]);
    rebuildRegistry();
    Settings.setBavAccountIds(['neo_depot']);
    Settings.setRiesterAccountIds([]);
    assert.strictEqual(isBav('neo_depot'), true);
    const t = calculateTotals({ neo_depot: 5000 });
    assert.strictEqual(t.bav, 5000);
    assert.strictEqual(t.riester, 0);
});

test('custom banks persist across a fresh Settings cache', () => {
    reset();
    Settings.setCustomBanks([{ prefix: 'neo', label: 'Neo', cbPrefix: 'neo', color: '#fff', accounts: [{ suffix: 'depot', label: 'Depot' }] }]);
    Settings._data = null; // force a reload from storage
    rebuildRegistry();
    assert.ok(getBanks().some(b => b.prefix === 'neo'));
});

let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
