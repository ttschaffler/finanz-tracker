// Automated tests for the company pension (bAV) feature.
//
// Exercises the pure logic via the shared VM harness (tests/harness.js)
// against the real shipping code in index.html: the Settings store, the
// effective `isBav` resolver, `calculateTotals` (incl. bAV), and the
// `annuitizeMonthly` conversion used to feed the Rentenlücke Betriebsrente.

const assert = require('assert');
const { loadApp, createRunner } = require('./harness');

const { sandbox, store } = loadApp(['accounts', 'BANKS', 'Settings', 'isBav', 'calculateTotals', 'annuitizeMonthly', 'bavIndicator']);
const { Settings, isBav, calculateTotals, annuitizeMonthly, bavIndicator } = sandbox;

// Reset persisted settings + cache between tests for isolation.
function reset() { store.clear(); Settings._data = null; }

const { test, run } = createRunner();

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

run();
