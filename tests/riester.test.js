// Automated tests for the Riester-Rente addition to the Rentenlücke net
// calculation. Like bav.test.js, we load the real inline <script> from
// index.html into an isolated VM context so we exercise the shipping logic.
//
// Riester contract under test:
//   - fully taxable in the payout phase (100% adds to zvE, like Betriebsrente)
//   - KV-/PV-beitragsfrei for pensioners (no social contributions)
//   - flows into bruttoGesamt / nettoGesamt as additional pension income

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
    Intl, Date, Math, JSON, parseFloat, parseInt, isFinite, Array, Object, Number, String,
    alert() {}, confirm() { return true; },
    Chart: function () {},
    document: { addEventListener() {}, getElementById: () => fakeElement(), querySelector: () => fakeElement(), querySelectorAll: () => [] },
    window: {}
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
const exportLine = '\nObject.assign(globalThis, { berechneNetto, KV_BETRIEB_VOLL, BETRIEBSRENTE_FREIGRENZE_MONAT });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { berechneNetto } = sandbox;

// Common parameters for a single, unmarried pensioner.
const BEST = 0.83;      // Besteuerungsanteil (fixed for the test)
const PV = 0.04;        // kinderlos
const close = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('Riester is KV-/PV-beitragsfrei: adding it does not change Sozialabgaben', () => {
    const without = berechneNetto(1500, 400, 0, 0, BEST, false, PV);
    const withRiester = berechneNetto(1500, 400, 300, 0, BEST, false, PV);
    assert.ok(close(without.sozialAbgaben, withRiester.sozialAbgaben),
        'social contributions must be identical with and without Riester');
});

test('Riester is fully taxable: zvE rises by the full annual Riester amount', () => {
    const without = berechneNetto(1500, 400, 0, 0, BEST, false, PV);
    const withRiester = berechneNetto(1500, 400, 300, 0, BEST, false, PV);
    assert.ok(close(withRiester.zvE - without.zvE, 300 * 12),
        'zvE must increase by the full yearly Riester (100% nachgelagert)');
    assert.ok(close(withRiester.steuerpflRiester, 300 * 12));
});

test('Riester adds to gross/net pension income (net rises by Riester minus extra tax)', () => {
    const without = berechneNetto(1500, 400, 0, 0, BEST, false, PV);
    const withRiester = berechneNetto(1500, 400, 300, 0, BEST, false, PV);
    const extraTax = withRiester.steuer - without.steuer;
    const expectedNetGain = 300 * 12 - extraTax; // no extra social contributions
    assert.ok(close(withRiester.nettoGesamt - without.nettoGesamt, expectedNetGain));
    assert.ok(withRiester.nettoGesamt > without.nettoGesamt, 'Riester increases net income');
});

test('zero Riester is a no-op versus omitting it', () => {
    const a = berechneNetto(1500, 400, 0, 0, BEST, false, PV);
    assert.strictEqual(a.riesterJahr, 0);
    assert.strictEqual(a.steuerpflRiester, 0);
});

test('Riester is taxed like Betriebsrente on income but cheaper overall (no KV/PV)', () => {
    // Same gross monthly amount routed through Riester vs. Betriebsrente.
    const viaRiester = berechneNetto(1500, 0, 500, 0, BEST, false, PV);
    const viaBetrieb = berechneNetto(1500, 500, 0, 0, BEST, false, PV);
    // Identical taxable income (both 100% steuerpflichtig)...
    assert.ok(close(viaRiester.zvE, viaBetrieb.zvE), 'same zvE for equal gross');
    // ...but Riester carries no KV/PV, so its net is higher.
    assert.ok(viaRiester.nettoGesamt > viaBetrieb.nettoGesamt,
        'Riester nets more than an equal Betriebsrente because it is KV-/PV-frei');
});

let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
