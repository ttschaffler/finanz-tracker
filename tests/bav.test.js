// Automated tests for the company pension (bAV) indicator feature.
//
// The app is a single static index.html with no build system, so these tests
// extract the relevant *pure* logic (the BANKS registry, the derived `accounts`
// list, and the `bavIndicator` helper) straight from index.html and evaluate it
// in an isolated VM context. This keeps the test honest: it runs the exact source
// that ships, with no duplicated copy to drift out of sync.

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

// ── Extract the pieces we need from the <script> block ────────────────────
function extract(label, startMarker, endMarker) {
    const start = html.indexOf(startMarker);
    assert.ok(start !== -1, `Could not find ${label} (marker: ${startMarker})`);
    const end = html.indexOf(endMarker, start);
    assert.ok(end !== -1, `Could not find end of ${label} (marker: ${endMarker})`);
    return html.slice(start, end + endMarker.length);
}

const banksSrc      = extract('BANKS registry',   'const BANKS = [',               '];');
const accountsSrc   = extract('accounts derive',  'const accounts = BANKS.flatMap', ');');
const indicatorSrc  = extract('bavIndicator',     'function bavIndicator(',         '\n        }');

const sandbox = {};
vm.createContext(sandbox);
// `const`/`let` don't attach to the VM context, so re-export onto globalThis.
const exportLine = '\nObject.assign(globalThis, { BANKS, accounts, bavIndicator });';
vm.runInContext(`${banksSrc}\n${accountsSrc}\n${indicatorSrc}${exportLine}`, sandbox);

const { BANKS, accounts, bavIndicator } = sandbox;

// ── Tests ─────────────────────────────────────────────────────────────────
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('adidas Konto is flagged as bAV in the registry', () => {
    const adidas = BANKS.find(b => b.prefix === 'adidas');
    assert.ok(adidas, 'adidas bank should exist');
    const konto = adidas.accounts.find(a => a.suffix === 'konto');
    assert.strictEqual(konto.bav, true);
});

test('exactly one account is flagged as bAV', () => {
    const flagged = accounts.filter(a => a.bav);
    assert.strictEqual(flagged.length, 1, 'only one account should be marked');
    assert.strictEqual(flagged[0].id, 'adidas_konto');
});

test('derived accounts default bav to false (never undefined)', () => {
    accounts.forEach(a => {
        assert.strictEqual(typeof a.bav, 'boolean', `${a.id} should have a boolean bav`);
    });
    assert.strictEqual(accounts.find(a => a.id === 'consors_depot').bav, false);
    assert.strictEqual(accounts.find(a => a.id === 'adidas_lta').bav, false);
});

test('bavIndicator renders an icon + German tooltip for bAV accounts', () => {
    const out = bavIndicator(true);
    assert.ok(out.includes('class="bav-badge"'), 'should use the bav-badge class');
    assert.ok(out.includes('title="Betriebliche Altersvorsorge"'), 'should carry the German tooltip');
    assert.ok(out.includes('🏢'), 'should include the icon');
});

test('bavIndicator renders nothing for non-bAV accounts', () => {
    assert.strictEqual(bavIndicator(false), '');
    assert.strictEqual(bavIndicator(undefined), '');
});

// ── Runner ──────────────────────────────────────────────────────────────
let failed = 0;
tests.forEach(([name, fn]) => {
    try {
        fn();
        console.log(`  ✓ ${name}`);
    } catch (err) {
        failed++;
        console.error(`  ✗ ${name}\n      ${err.message}`);
    }
});

console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
process.exit(failed === 0 ? 0 : 1);
