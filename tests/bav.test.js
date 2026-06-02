// Automated tests for the company pension (bAV) feature.
//
// The app is a single static index.html with no build system. To test the real
// shipping code (not a duplicated copy), we load the entire inline <script> into
// an isolated VM context with lightweight stubs for the browser/Firebase globals
// it touches at load time. We then exercise the pure logic: the Settings store,
// the effective `isBav` resolver, `calculateTotals` (now incl. bAV), and the
// `annuitizeMonthly` conversion used to feed the Rentenlücke Betriebsrente.

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

// ── Pull the inline app script (the only <script> with no attributes) ──────
const startTag = '<script>';
const sIdx = html.indexOf(startTag);
const eIdx = html.indexOf('</script>', sIdx);
assert.ok(sIdx !== -1 && eIdx !== -1, 'Could not locate the inline <script> block');
const scriptSrc = html.slice(sIdx + startTag.length, eIdx);

// ── Minimal browser / Firebase stubs the script needs at load time ─────────
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

const chainable = new Proxy(function () {}, {
    get: () => chainable,
    apply: () => chainable
});
const firebase = {
    initializeApp() {},
    firestore: Object.assign(() => chainable, { FieldValue: chainable }),
    auth: Object.assign(() => ({ onAuthStateChanged() {}, signOut() {}, signInWithPopup() { return { catch() {} }; } }),
        { GoogleAuthProvider: function () {} })
};

const sandbox = {
    firebase, localStorage, console,
    Intl, Date, Math, JSON, parseFloat, parseInt, isFinite, Array, Object, Number, String,
    alert() {}, confirm() { return true; },
    Chart: function () {},
    document: {
        addEventListener() {},
        getElementById: () => fakeElement(),
        querySelector: () => fakeElement(),
        querySelectorAll: () => []
    },
    window: {}
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
// `const`/`function` at top level don't attach to the context, so re-export the
// symbols we want to test (the appended code shares the script's scope).
const exportLine = '\nObject.assign(globalThis, { accounts, BANKS, Settings, AppState, isBav, isRiester, calculateTotals, annuitizeMonthly, bavIndicator, riesterIndicator, currentBavKapital, currentBetriebsrenteMonatlich, recomputeBetriebsrenteFromKapital });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { accounts, Settings, AppState, isBav, isRiester, calculateTotals, annuitizeMonthly, bavIndicator, riesterIndicator, currentBavKapital, currentBetriebsrenteMonatlich, recomputeBetriebsrenteFromKapital } = sandbox;

// Stub document.getElementById to serve values from a plain id→value map.
function stubFields(map) {
    sandbox.document.getElementById = id => ({ value: map[id] !== undefined ? map[id] : '' });
}

// Like stubFields, but returns persistent element objects so writes stick.
function stubElements(map) {
    const els = {};
    Object.keys(map).forEach(id => { els[id] = { value: map[id] }; });
    sandbox.document.getElementById = id => (els[id] || (els[id] = { value: '' }));
    return els;
}

// Reset persisted settings + cache between tests for isolation.
function reset() { store.clear(); Settings._data = null; }

// ── Tests ───────────────────────────────────────────────────────────────
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('default bAV designation comes from the registry (adidas Konto)', () => {
    reset();
    assert.strictEqual(isBav('adidas_konto'), true);
    assert.strictEqual(isBav('consors_depot'), false);
    assert.strictEqual(isBav('adidas_lta'), false);
});

test('Settings.setBavAccountIds persists and overrides the default', () => {
    reset();
    Settings.setBavAccountIds(['consors_depot', 'tr_depot']);
    assert.strictEqual(isBav('consors_depot'), true);
    assert.strictEqual(isBav('tr_depot'), true);
    assert.strictEqual(isBav('adidas_konto'), false, 'default no longer applies once configured');
    // A fresh cache must read the same persisted value back from storage.
    Settings._data = null;
    assert.strictEqual(isBav('consors_depot'), true);
});

test('an empty bАV selection is respected (not treated as unset)', () => {
    reset();
    Settings.setBavAccountIds([]);
    assert.strictEqual(isBav('adidas_konto'), false);
});

test('calculateTotals sums bAV separately without disturbing konto/depot/total', () => {
    reset(); // adidas_konto is bAV by default
    const entry = { adidas_konto: 1000, adidas_depot: 2000, consors_konto: 500, consors_depot: 700 };
    const t = calculateTotals(entry);
    assert.strictEqual(t.konten, 1500, 'adidas_konto + consors_konto');
    assert.strictEqual(t.depots, 2700, 'adidas_depot + consors_depot');
    assert.strictEqual(t.total, 4200);
    assert.strictEqual(t.bav, 1000, 'only the bAV account (adidas_konto) counts');
});

test('calculateTotals.bav follows Settings changes', () => {
    reset();
    Settings.setBavAccountIds(['consors_depot']);
    const entry = { adidas_konto: 1000, consors_depot: 700 };
    const t = calculateTotals(entry);
    assert.strictEqual(t.bav, 700);
    assert.strictEqual(t.total, 1700, 'total is unaffected by bAV designation');
});

test('annuitizeMonthly spreads the pot evenly over Entnahmezeitraum (no growth)', () => {
    assert.strictEqual(annuitizeMonthly(12000, 25), 12000 / (25 * 12));
    assert.strictEqual(annuitizeMonthly(36000, 25), 120);
});

test('annuitizeMonthly guards invalid payout periods', () => {
    assert.strictEqual(annuitizeMonthly(12000, 0), 0);
    assert.strictEqual(annuitizeMonthly(12000, -5), 0);
    assert.strictEqual(annuitizeMonthly(12000, NaN), 0);
});

test('bavIndicator renders icon + German tooltip only for bAV accounts', () => {
    const on = bavIndicator(true);
    assert.ok(on.includes('class="bav-badge"'));
    assert.ok(on.includes('title="Betriebliche Altersvorsorge"'));
    assert.ok(on.includes('🏢'));
    assert.strictEqual(bavIndicator(false), '');
});

test('Riester designation defaults to none and is independently configurable', () => {
    reset();
    assert.strictEqual(isRiester('allianz_konto'), false, 'no Riester accounts by default');
    Settings.setRiesterAccountIds(['allianz_konto']);
    assert.strictEqual(isRiester('allianz_konto'), true);
    // A fresh cache must read the same persisted value back.
    Settings._data = null;
    assert.strictEqual(isRiester('allianz_konto'), true);
});

test('bAV and Riester designations are independent (no cross-contamination)', () => {
    reset();
    Settings.setBavAccountIds(['adidas_konto']);
    Settings.setRiesterAccountIds(['allianz_konto']);
    assert.strictEqual(isBav('adidas_konto'), true);
    assert.strictEqual(isRiester('adidas_konto'), false);
    assert.strictEqual(isRiester('allianz_konto'), true);
    assert.strictEqual(isBav('allianz_konto'), false);
});

test('calculateTotals sums the Riester pot separately without touching total', () => {
    reset();
    Settings.setBavAccountIds(['adidas_konto']);
    Settings.setRiesterAccountIds(['allianz_konto']);
    const entry = { adidas_konto: 1000, allianz_konto: 800, consors_depot: 500 };
    const t = calculateTotals(entry);
    assert.strictEqual(t.bav, 1000);
    assert.strictEqual(t.riester, 800);
    assert.strictEqual(t.total, 2300, 'total is unaffected by bAV/Riester designation');
});

test('Bestehendes Vermögen takeover excludes both bAV and Riester (no double counting)', () => {
    reset();
    Settings.setBavAccountIds(['adidas_konto']);
    Settings.setRiesterAccountIds(['allianz_konto']);
    const entry = { adidas_konto: 1000, allianz_konto: 800, consors_depot: 500 };
    const { total, bav, riester } = calculateTotals(entry);
    // Mirrors uebernahmeVermoegen2: total - bav - riester.
    assert.strictEqual(total - bav - riester, 500, 'only the non-bAV, non-Riester rest remains');
});

test('riesterIndicator renders icon + German tooltip only for Riester accounts', () => {
    const on = riesterIndicator(true);
    assert.ok(on.includes('title="Riester-Vorsorge"'));
    assert.ok(on.includes('🅁'));
    assert.strictEqual(riesterIndicator(false), '');
});

test('currentBavKapital prefers the manual bAV-Gesamtkapital field', () => {
    reset();
    AppState.entries = [{ adidas_konto: 99999 }]; // would be the fallback
    stubFields({ rl2_bav_kapital: '60000' });
    assert.strictEqual(currentBavKapital(), 60000);
});

test('currentBavKapital falls back to the bAV pot when the field is empty', () => {
    reset(); // adidas_konto is bAV by default
    AppState.entries = [{ adidas_konto: 36000 }];
    stubFields({ rl2_bav_kapital: '' });
    assert.strictEqual(currentBavKapital(), 36000);
});

test('currentBetriebsrenteMonatlich annuitizes the bAV-Gesamtkapital over Entnahmezeitraum', () => {
    reset();
    AppState.entries = [];
    stubFields({ rl2_bav_kapital: '60000', rl2_entnahmezeitraum: '25' });
    assert.strictEqual(currentBetriebsrenteMonatlich(), 60000 / (25 * 12)); // 200 €/Mon.
});

test('currentBetriebsrenteMonatlich needs a payout period', () => {
    reset();
    AppState.entries = [];
    stubFields({ rl2_bav_kapital: '60000', rl2_entnahmezeitraum: '' });
    assert.strictEqual(currentBetriebsrenteMonatlich(), null);
});

test('recomputeBetriebsrenteFromKapital derives the monthly Betriebsrente automatically', () => {
    reset();
    const els = stubElements({ rl2_bav_kapital: '60000', rl2_entnahmezeitraum: '25', rl2_betriebsrente: '' });
    recomputeBetriebsrenteFromKapital();
    assert.strictEqual(els.rl2_betriebsrente.value, (200).toFixed(2)); // 60000/(25*12)
});

test('recomputeBetriebsrenteFromKapital is a no-op without capital (manual value kept)', () => {
    reset();
    const els = stubElements({ rl2_bav_kapital: '', rl2_entnahmezeitraum: '25', rl2_betriebsrente: '350.00' });
    recomputeBetriebsrenteFromKapital();
    assert.strictEqual(els.rl2_betriebsrente.value, '350.00');
});

// ── Runner ────────────────────────────────────────────────────────────────
let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
