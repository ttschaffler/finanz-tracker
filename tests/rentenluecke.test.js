// Automated tests for the Rentenlücke calculation (taxation, social
// contributions, capital-need projection). Exercises the pure logic via the
// shared VM harness against the real shipping code in index.html.

const assert = require('assert');
const { loadApp, createRunner } = require('./harness');

const { sandbox } = loadApp([
    'besteuerungsanteil', 'einkommensteuerGrund', 'einkommensteuer',
    'solidaritaetszuschlag', 'pvRentnerSatz', 'berechneNetto', 'calculateRL2',
    'annuitizeMonthly', 'taxParamsForYear', 'TAX_PARAMS_BY_YEAR',
    'KV_RENTNER_SATZ', 'KV_BETRIEB_VOLL', 'BETRIEBSRENTE_FREIGRENZE_MONAT'
]);
const {
    besteuerungsanteil, einkommensteuerGrund, einkommensteuer,
    solidaritaetszuschlag, pvRentnerSatz, berechneNetto, calculateRL2,
    taxParamsForYear, TAX_PARAMS_BY_YEAR,
    KV_RENTNER_SATZ, KV_BETRIEB_VOLL, BETRIEBSRENTE_FREIGRENZE_MONAT
} = sandbox;

const approx = (actual, expected, eps = 0.01, msg) =>
    assert.ok(Math.abs(actual - expected) < eps, msg || `expected ~${expected}, got ${actual}`);

const { test, run } = createRunner();

// ── Besteuerungsanteil ────────────────────────────────────────────────────
test('besteuerungsanteil follows the statutory phase-in schedule', () => {
    assert.strictEqual(besteuerungsanteil(2005), 0.50);
    approx(besteuerungsanteil(2010), 0.60);          // +2%/Jahr bis 2020
    approx(besteuerungsanteil(2020), 0.80);
    approx(besteuerungsanteil(2022), 0.82);          // +1%/Jahr 2021–2022
    approx(besteuerungsanteil(2040), 0.82 + 18 * 0.005); // +0,5%/Jahr ab 2023
    assert.strictEqual(besteuerungsanteil(2058), 1.00);
    assert.strictEqual(besteuerungsanteil(2070), 1.00);
});

// ── Einkommensteuer §32a EStG 2025 ───────────────────────────────────────
test('einkommensteuerGrund: zero up to Grundfreibetrag, statutory values above', () => {
    assert.strictEqual(einkommensteuerGrund(0), 0);
    assert.strictEqual(einkommensteuerGrund(12084), 0);
    // Proportionalzone 42 % and Reichensteuer 45 % have closed-form values
    approx(einkommensteuerGrund(100000), 0.42 * 100000 - 10636.31);
    approx(einkommensteuerGrund(300000), 0.45 * 300000 - 18971.06);
});

test('einkommensteuerGrund is monotonically increasing across zone boundaries', () => {
    let prev = -1;
    for (const zvE of [12084, 12085, 17005, 17006, 30000, 66760, 66761, 277825, 277826]) {
        const t = einkommensteuerGrund(zvE);
        assert.ok(t >= prev, `tax must not decrease at zvE=${zvE}`);
        prev = t;
    }
});

test('einkommensteuer applies Splitting for married taxpayers', () => {
    const zvE = 40000;
    approx(einkommensteuer(zvE, true), 2 * einkommensteuerGrund(zvE / 2));
    assert.ok(einkommensteuer(zvE, true) <= einkommensteuer(zvE, false),
        'Splitting must never be worse than Grundtarif');
});

// ── Solidaritätszuschlag ─────────────────────────────────────────────────
test('solidaritaetszuschlag: Freigrenze, Milderungszone, full rate', () => {
    assert.strictEqual(solidaritaetszuschlag(0), 0);
    assert.strictEqual(solidaritaetszuschlag(18130), 0);
    approx(solidaritaetszuschlag(20000), (20000 - 18130) * 0.119); // Milderungszone
    approx(solidaritaetszuschlag(100000), 100000 * 0.055);          // voller Satz
});

// ── Pflegeversicherung ───────────────────────────────────────────────────
test('pvRentnerSatz: 4,0 % kinderlos, 3,4 % mit Kind(ern)', () => {
    assert.strictEqual(pvRentnerSatz(0), 0.04);
    assert.strictEqual(pvRentnerSatz(1), 0.034);
    assert.strictEqual(pvRentnerSatz(5), 0.034);
});

// ── berechneNetto ────────────────────────────────────────────────────────
test('berechneNetto: components are internally consistent', () => {
    const r = berechneNetto(2000, 0, 0, 0.83, false, 0.034);
    approx(r.kvGesetzl, 24000 * KV_RENTNER_SATZ);
    approx(r.pvGesetzl, 24000 * 0.034);
    assert.strictEqual(r.kvBetrieb, 0);
    assert.strictEqual(r.pvBetrieb, 0);
    approx(r.zvE, 24000 * 0.83 - 102);                     // Werbungskostenpauschale
    approx(r.steuer, r.est + r.soli);
    approx(r.nettoGesamt, 24000 - r.sozialAbgaben - r.steuer);
});

test('berechneNetto: Betriebsrente below Freigrenze pays no KV, above pays full rate', () => {
    const below = berechneNetto(1500, 150, 0, 0.83, false, 0.034);
    assert.strictEqual(below.kvBetrieb, 0, '150 €/Mon. liegt unter der Freigrenze');
    const above = berechneNetto(1500, 500, 0, 0.83, false, 0.034);
    approx(above.kvBetrieb, (500 * 12 - BETRIEBSRENTE_FREIGRENZE_MONAT * 12) * KV_BETRIEB_VOLL);
});

test('berechneNetto: Entnahme is added net (no further deductions on it)', () => {
    const without = berechneNetto(2000, 0, 0, 0.83, false, 0.034);
    const withEntnahme = berechneNetto(2000, 0, 500, 0.83, false, 0.034);
    approx(withEntnahme.nettoGesamt - without.nettoGesamt, 500 * 12);
});

// ── calculateRL2 ─────────────────────────────────────────────────────────
const baseInputs = {
    alter: 35, rentenalter: 67, gesetzlichBrutto: 1800, rentensteigerung: 1.5,
    betriebsrenteBrutto: 0, wunschrenteNetto: 2500, inflation: 2.5,
    vermoegen: 100000, verzinsung: 5, entnahmezeitraum: 25,
    istVerheiratet: false, anzahlKinder: 0
};

test('calculateRL2: derived basics (years, entry year, compounding)', () => {
    const r = calculateRL2(baseInputs);
    assert.strictEqual(r.jahresBisRente, 32);
    assert.strictEqual(r.renteneintrittsjahr, new Date().getFullYear() + 32);
    approx(r.vermoegenBeiRente, 100000 * Math.pow(1.05, 32), 0.01);
    approx(r.wunschNettoBeiRente, 2500 * Math.pow(1.025, 32), 0.01);
});

test('calculateRL2: chart series span Entnahmezeitraum + 1 years', () => {
    const r = calculateRL2(baseInputs);
    for (const key of ['labels', 'vermoegenVerlauf', 'nettoBedarfVerlauf', 'nettoRenteVerlauf', 'entnahmeVerlauf']) {
        assert.strictEqual(r[key].length, baseInputs.entnahmezeitraum + 1, `${key} length`);
    }
    approx(r.vermoegenVerlauf[0], r.vermoegenBeiRente, 0.01);
});

test('calculateRL2: benötigtes Kapital equals the discounted withdrawal stream', () => {
    const r = calculateRL2(baseInputs);
    let expected = 0;
    for (let j = 0; j < baseInputs.entnahmezeitraum; j++) {
        expected += r.entnahmeVerlauf[j] * 12 / Math.pow(1.05, j);
    }
    // entnahmeVerlauf is rounded to cents, so allow a small tolerance
    approx(r.benoetigtesKapital, expected, 1);
});

test('calculateRL2: no gap when the desired pension is fully covered', () => {
    const r = calculateRL2({ ...baseInputs, wunschrenteNetto: 100, vermoegen: 500000 });
    assert.strictEqual(r.benoetigteEntnahmeMonat1, 0);
    assert.strictEqual(r.vermoegenslücke, 0);
    assert.strictEqual(r.monatlicheSparrate, 0);
});

test('calculateRL2: gap produces a positive monthly savings rate', () => {
    const r = calculateRL2({ ...baseInputs, vermoegen: 0, wunschrenteNetto: 4000 });
    assert.ok(r.vermoegenslücke > 0);
    assert.ok(r.monatlicheSparrate > 0);
    // Sparrate must amortize exactly to the Vermögenslücke at retirement
    const i = 0.05 / 12, n = 32 * 12;
    approx(r.monatlicheSparrate, r.vermoegenslücke * i / (Math.pow(1 + i, n) - 1), 0.01);
});

test('calculateRL2: Betriebsrente reduces the monthly gap', () => {
    const without = calculateRL2(baseInputs);
    const withBav = calculateRL2({ ...baseInputs, betriebsrenteBrutto: 400 });
    assert.ok(withBav.benoetigteEntnahmeMonat1 < without.benoetigteEntnahmeMonat1);
    assert.ok(withBav.benoetigtesKapital < without.benoetigtesKapital);
});

// ── Tax parameter table (B1) ─────────────────────────────────────────────
test('taxParamsForYear picks the latest year <= requested, fallback earliest', () => {
    assert.strictEqual(taxParamsForYear(2025), TAX_PARAMS_BY_YEAR[2025]);
    assert.strictEqual(taxParamsForYear(2040), TAX_PARAMS_BY_YEAR[2025], 'future years use the latest known set');
    assert.strictEqual(taxParamsForYear(2000), TAX_PARAMS_BY_YEAR[2025], 'years before the table use the earliest set');
});

test('einkommensteuerGrund honors an injected parameter set', () => {
    const custom = { ...TAX_PARAMS_BY_YEAR[2025].tarif, grundfreibetrag: 20000 };
    assert.strictEqual(einkommensteuerGrund(15000, custom), 0, 'higher Grundfreibetrag exempts 15k');
    assert.ok(einkommensteuerGrund(15000) > 0, 'default 2025 set still taxes 15k');
});

// ── Separate drawdown return (B5) ────────────────────────────────────────
test('calculateRL2: lower Ruhestand return raises the capital need, leaves accumulation untouched', () => {
    const base = calculateRL2(baseInputs);
    const safer = calculateRL2({ ...baseInputs, verzinsungRuhestand: 2 });
    assert.ok(safer.benoetigtesKapital > base.benoetigtesKapital);
    approx(safer.vermoegenBeiRente, base.vermoegenBeiRente, 0.01, 'Ansparphase keeps its own rate');
    assert.strictEqual(base.verzinsungRuhestand, baseInputs.verzinsung, 'fallback: drawdown = accumulation rate');
});

// ── Withdrawal tax gross-up (B3) ─────────────────────────────────────────
test('calculateRL2: withdrawal tax grosses up the Entnahme and the capital need', () => {
    const noTax = calculateRL2(baseInputs);
    const taxed = calculateRL2({ ...baseInputs, entnahmeSteuer: 20 });
    approx(taxed.entnahmeVerlauf[0], noTax.entnahmeVerlauf[0] / 0.8, 0.02);
    approx(taxed.benoetigtesKapital, noTax.benoetigtesKapital / 0.8, 1);
    assert.strictEqual(taxed.benoetigteEntnahmeMonat1, noTax.benoetigteEntnahmeMonat1,
        'the displayed net gap is unaffected by the gross-up');
});

run();
