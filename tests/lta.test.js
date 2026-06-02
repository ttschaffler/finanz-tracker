// Automated tests for the Lebensarbeitszeitkonto (LTA) feature in the
// Rentenlücke. Loads the real inline <script> from index.html into a VM and
// exercises the LTA designation (calculateTotals.lta, isLta) plus the three
// usage options modelled in calculateRL2:
//   1) Altersteilzeit  → no effect on the retirement gap (informational)
//   2) monatlich       → Betriebsrenten-like taxed monthly payout over ltaDauer
//   3) einmal          → gross lump sum added to the drawdown capital pot

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
const exportLine = '\nObject.assign(globalThis, { Settings, rebuildRegistry, calculateTotals, calculateRL2, isLta, isBav });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { Settings, rebuildRegistry, calculateTotals, calculateRL2, isLta } = sandbox;

function reset() { store.clear(); Settings._data = null; rebuildRegistry(); }
const close = (a, b, eps = 1e-3) => Math.abs(a - b) < eps;

function baseInputs(overrides = {}) {
    return {
        alter: 40, rentenalter: 67, gesetzlichBrutto: 1500, rentensteigerung: 1.5,
        betriebsrenteBrutto: 0, riesterBrutto: 0, riesterKapital: 0,
        ltaWert: 0, ltaOption: 'altersteilzeit', ltaDauer: 0,
        wunschrenteNetto: 2500, inflation: 2.0, vermoegen: 100000, verzinsung: 5.0,
        entnahmezeitraum: 25, istVerheiratet: false, anzahlKinder: 0, ...overrides
    };
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('LTA designation defaults to the registry LTA account (adidas_lta)', () => {
    reset();
    assert.strictEqual(isLta('adidas_lta'), true);
    assert.strictEqual(isLta('adidas_konto'), false);
    assert.strictEqual(isLta('consors_depot'), false);
});

test('LTA designation is configurable and persists', () => {
    reset();
    Settings.setLtaAccountIds(['consors_depot']);
    assert.strictEqual(isLta('consors_depot'), true);
    assert.strictEqual(isLta('adidas_lta'), false, 'default no longer applies once configured');
    Settings._data = null;
    assert.strictEqual(isLta('consors_depot'), true);
});

test('calculateTotals sums the LTA pot separately without touching total', () => {
    reset(); // adidas_lta is LTA by default
    const entry = { adidas_lta: 40000, consors_depot: 1000 };
    const t = calculateTotals(entry);
    assert.strictEqual(t.lta, 40000);
    assert.strictEqual(t.total, 41000, 'total is unaffected by the LTA designation');
});

test('Bestehendes Vermögen takeover excludes bAV, Riester AND LTA', () => {
    reset();
    Settings.setBavAccountIds(['adidas_konto']);
    Settings.setRiesterAccountIds(['allianz_konto']);
    Settings.setLtaAccountIds(['adidas_lta']);
    const entry = { adidas_konto: 1000, allianz_konto: 800, adidas_lta: 40000, consors_depot: 500 };
    const { total, bav, riester, lta } = calculateTotals(entry);
    assert.strictEqual(total - bav - riester - lta, 500, 'only the plain rest remains');
});

test('Option Altersteilzeit has no effect on the retirement calculation', () => {
    const baseline = calculateRL2(baseInputs());
    const atz = calculateRL2(baseInputs({ ltaWert: 50000, ltaOption: 'altersteilzeit' }));
    assert.ok(close(baseline.vermoegenBeiRente, atz.vermoegenBeiRente));
    assert.ok(close(baseline.rentenNettoOhneEntnahme, atz.rentenNettoOhneEntnahme));
    assert.strictEqual(atz.ltaKapital, 0);
    assert.strictEqual(atz.ltaMonatlich, 0);
});

test('Option Einmalbetrag adds the LTA gross as capital to the pot', () => {
    const without = calculateRL2(baseInputs());
    const einmal = calculateRL2(baseInputs({ ltaWert: 50000, ltaOption: 'einmal' }));
    assert.strictEqual(einmal.ltaKapital, 50000);
    assert.strictEqual(einmal.ltaMonatlich, 0);
    const expectedExtra = 50000 * Math.pow(1.05, 27); // grown to retirement
    assert.ok(close(einmal.vermoegenBeiRente - without.vermoegenBeiRente, expectedExtra),
        'lump sum is compounded into the retirement pot');
    // Capital does not change the taxed monthly pension.
    assert.ok(close(without.rentenNettoOhneEntnahme, einmal.rentenNettoOhneEntnahme));
});

test('Option monatlich annuitizes the LTA over ltaDauer years', () => {
    const r = calculateRL2(baseInputs({ ltaWert: 48000, ltaOption: 'monatlich', ltaDauer: 10 }));
    assert.strictEqual(r.ltaKapital, 0);
    assert.ok(close(r.ltaMonatlich, 48000 / (10 * 12))); // 400 €/Mon.
});

test('Option monatlich raises early net pension and reverts after the payout window', () => {
    const without = calculateRL2(baseInputs());
    const withLta = calculateRL2(baseInputs({ ltaWert: 48000, ltaOption: 'monatlich', ltaDauer: 10 }));
    // Year 0: LTA is being paid out → higher net pension than without.
    assert.ok(withLta.rentenNettoOhneEntnahme > without.rentenNettoOhneEntnahme,
        'LTA monthly payout lifts the net pension while it runs');
    // Year 10 (index 10): LTA has stopped → net pension matches the no-LTA case.
    assert.ok(close(withLta.nettoRenteVerlauf[10], without.nettoRenteVerlauf[10], 0.5),
        'after ltaDauer years the LTA payout is gone again');
    // While LTA runs, the net pension exceeds the post-payout level.
    assert.ok(withLta.nettoRenteVerlauf[0] > withLta.nettoRenteVerlauf[10]);
});

let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
