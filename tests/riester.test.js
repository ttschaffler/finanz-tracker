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
const exportLine = '\nObject.assign(globalThis, { berechneNetto, calculateRL2, KV_BETRIEB_VOLL, BETRIEBSRENTE_FREIGRENZE_MONAT, KV_RENTNER_SATZ });';
vm.runInContext(scriptSrc + exportLine, sandbox);

const { berechneNetto, calculateRL2, KV_RENTNER_SATZ } = sandbox;

// Baseline inputs object for calculateRL2 (matches readRL2Inputs' shape).
function baseInputs(overrides = {}) {
    return {
        alter: 40, rentenalter: 67, gesetzlichBrutto: 1500, rentensteigerung: 1.5,
        betriebsrenteBrutto: 0, riesterBrutto: 0, riesterKapital: 0,
        wunschrenteNetto: 2500, inflation: 2.0, vermoegen: 100000, verzinsung: 5.0,
        entnahmezeitraum: 25, istVerheiratet: false, anzahlKinder: 0, ...overrides
    };
}

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

test('Riester-Kapital joins the drawdown pot and raises Vermögen bei Renteneintritt', () => {
    const without = calculateRL2(baseInputs());
    const withKapital = calculateRL2(baseInputs({ riesterKapital: 50000 }));
    // 50k extra capital, compounded over (67-40)=27 years at 5%.
    const expectedExtra = 50000 * Math.pow(1.05, 27);
    assert.ok(close(withKapital.vermoegenBeiRente - without.vermoegenBeiRente, expectedExtra, 1e-3),
        'Riester capital is compounded into the retirement pot');
});

test('Riester-Kapital shrinks the Vermögenslücke (more available capital)', () => {
    const without = calculateRL2(baseInputs());
    const withKapital = calculateRL2(baseInputs({ riesterKapital: 50000 }));
    assert.ok(withKapital.vermoegenslücke <= without.vermoegenslücke,
        'extra capital can only reduce (never increase) the wealth gap');
});

test('Riester-Kapital does not change the net pension (it is capital, not income)', () => {
    const without = calculateRL2(baseInputs());
    const withKapital = calculateRL2(baseInputs({ riesterKapital: 50000 }));
    assert.ok(close(without.rentenNettoOhneEntnahme, withKapital.rentenNettoOhneEntnahme),
        'capital must not affect the taxed monthly pension');
});

test('GKV is the default: undefined kvModus equals explicit "gesetzlich"', () => {
    const a = berechneNetto(1500, 400, 0, 0, BEST, false, PV);
    const b = berechneNetto(1500, 400, 0, 0, BEST, false, PV, 'gesetzlich', 0);
    assert.ok(close(a.sozialAbgaben, b.sozialAbgaben));
    assert.ok(close(a.nettoGesamt, b.nettoGesamt));
    assert.strictEqual(b.kvModus, 'gesetzlich');
});

test('PKV uses a fixed premium minus the RV-Zuschuss instead of percentages', () => {
    const r = berechneNetto(1500, 400, 0, 0, BEST, false, PV, 'privat', 800);
    const pkvJahr = 800 * 12;                       // 9600
    const erwarteterZuschuss = 1500 * 12 * KV_RENTNER_SATZ; // < halber Beitrag
    assert.ok(close(r.pkvBeitrag, pkvJahr));
    assert.ok(close(r.rvZuschuss, erwarteterZuschuss));
    assert.ok(close(r.sozialAbgaben, pkvJahr - erwarteterZuschuss));
    // No percentage-based GKV contributions in PKV mode.
    assert.strictEqual(r.kvGesetzl, 0);
    assert.strictEqual(r.kvBetrieb, 0);
    assert.strictEqual(r.kvModus, 'privat');
});

test('PKV premium is independent of the Betriebsrente (fixed contribution)', () => {
    const ohneBetrieb = berechneNetto(1500, 0, 0, 0, BEST, false, PV, 'privat', 800);
    const mitBetrieb = berechneNetto(1500, 1000, 0, 0, BEST, false, PV, 'privat', 800);
    assert.ok(close(ohneBetrieb.sozialAbgaben, mitBetrieb.sozialAbgaben),
        'extra Betriebsrente does not raise the PKV social cost');
    // ...but the Betriebsrente is still taxed (zvE rises).
    assert.ok(mitBetrieb.zvE > ohneBetrieb.zvE);
});

test('RV-Zuschuss is capped at half of the PKV premium', () => {
    // Small premium → the half-premium cap binds, not the pension-based amount.
    const r = berechneNetto(1500, 0, 0, 0, BEST, false, PV, 'privat', 100);
    const pkvJahr = 100 * 12; // 1200
    assert.ok(close(r.rvZuschuss, pkvJahr / 2), 'Zuschuss limited to half the premium');
    assert.ok(close(r.sozialAbgaben, pkvJahr / 2));
});

test('Tax (zvE) is unaffected by the health-insurance mode', () => {
    const gkv = berechneNetto(1500, 400, 0, 0, BEST, false, PV, 'gesetzlich', 0);
    const pkv = berechneNetto(1500, 400, 0, 0, BEST, false, PV, 'privat', 800);
    assert.ok(close(gkv.zvE, pkv.zvE));
    assert.ok(close(gkv.steuer, pkv.steuer));
});

function pkvInputs(o = {}) {
    return baseInputs({ kvModus: 'privat', pkvBeitrag: 1000, pkvSteigerung: 0, pkvEntlastung: 0, ...o });
}

test('PKV: higher annual increase raises the premium at retirement (forecast to Rentenbeginn)', () => {
    const flat = calculateRL2(pkvInputs({ pkvSteigerung: 0 }));
    const rising = calculateRL2(pkvInputs({ pkvSteigerung: 4 }));
    // First retirement year social cost (premium − RV-Zuschuss), monthly.
    assert.ok(rising.jahresDetails[0].sozialMonat > flat.jahresDetails[0].sozialMonat,
        'a rising premium compounds to a higher value at retirement');
});

test('PKV: the one-time relief at retirement lowers the premium', () => {
    const ohne = calculateRL2(pkvInputs({ pkvEntlastung: 0 }));
    const mit = calculateRL2(pkvInputs({ pkvEntlastung: 20 }));
    assert.ok(mit.jahresDetails[0].sozialMonat < ohne.jahresDetails[0].sozialMonat,
        'the retirement-entry relief reduces the PKV cost');
});

test('PKV: the premium keeps rising through retirement', () => {
    const r = calculateRL2(pkvInputs({ pkvSteigerung: 4, entnahmezeitraum: 20 }));
    assert.ok(r.jahresDetails[10].sozialMonat > r.jahresDetails[0].sozialMonat,
        'the PKV social cost grows over the drawdown years');
});

let failed = 0;
tests.forEach(([name, fn]) => {
    try { fn(); console.log(`  ✓ ${name}`); }
    catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
});
console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
