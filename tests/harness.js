// Shared test harness for the single-file app.
//
// The app is a single static index.html with no build system. To test the real
// shipping code (not a duplicated copy), we load the entire inline <script> into
// an isolated VM context with lightweight stubs for the browser/Firebase globals
// it touches at load time. Each test file lists the top-level symbols it needs;
// they are re-exported onto the sandbox because `const`/`function` declarations
// at the top level of a VM script do not attach to the context object.

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

function fakeElement() {
    return {
        value: '', textContent: '', innerHTML: '', className: '', checked: false,
        style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
        addEventListener() {}, querySelectorAll() { return []; }
    };
}

function loadApp(exportNames) {
    const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

    // Pull the inline app script (the only <script> with no attributes)
    const startTag = '<script>';
    const sIdx = html.indexOf(startTag);
    const eIdx = html.indexOf('</script>', sIdx);
    assert.ok(sIdx !== -1 && eIdx !== -1, 'Could not locate the inline <script> block');
    const scriptSrc = html.slice(sIdx + startTag.length, eIdx);

    // Minimal browser / Firebase stubs the script needs at load time
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
        Intl, Date, Math, JSON, parseFloat, parseInt, isFinite, isNaN, Array, Object, Number, String,
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
    const exportLine = `\nObject.assign(globalThis, { ${exportNames.join(', ')} });`;
    vm.runInContext(scriptSrc + exportLine, sandbox);

    return { sandbox, store };
}

// Minimal sequential test runner shared by all test files.
function createRunner() {
    const tests = [];
    const test = (name, fn) => tests.push([name, fn]);
    const run = () => {
        let failed = 0;
        tests.forEach(([name, fn]) => {
            try { fn(); console.log(`  ✓ ${name}`); }
            catch (err) { failed++; console.error(`  ✗ ${name}\n      ${err.message}`); }
        });
        console.log(`\n${tests.length - failed}/${tests.length} tests passed`);
        process.exit(failed === 0 ? 0 : 1);
    };
    return { test, run };
}

module.exports = { loadApp, createRunner };
