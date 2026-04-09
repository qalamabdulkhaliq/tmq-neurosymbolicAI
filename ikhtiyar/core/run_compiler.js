/**
 * run_compiler.js — Node.js bootstrapper for mushaf_compiler.qlb
 *
 * Loads the Qalb runtime, adds Arabic-named Node.js primitives for
 * file I/O and object access, then executes the Mushaf compiler.
 *
 * Usage:
 *   node ikhtiyar/core/run_compiler.js
 *
 * Output:
 *   ikhtiyar/constraint_ruleset.json
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ── Resolve paths ─────────────────────────────────────────────────────────────

const PROJECT_ROOT  = path.resolve(__dirname, '..', '..');
const QALB_DIR      = path.join(PROJECT_ROOT, 'Qalb', '----master', 'public', 'qlb');
const COMPILER_FILE = path.join(__dirname, 'mushaf_compiler.qlb');
const TMQ_PATH      = path.join(PROJECT_ROOT, 'bismillah', 'TMQ_v12.json');
const OUTPUT_PATH   = path.join(PROJECT_ROOT, 'ikhtiyar', 'constraint_ruleset.json');

// ── Load Qalb runtime ─────────────────────────────────────────────────────────

// Load all three Qalb runtime files in a single Function() scope so `var Qlb`
// stays visible across all three files. Return it explicitly.
global.window = { setTimeout: setTimeout };  // stub for primitives.js أجل

const Qlb = new Function('global', 'window',
  fs.readFileSync(path.join(QALB_DIR, 'qlb.js'),        'utf8') + '\n' +
  fs.readFileSync(path.join(QALB_DIR, 'parser.js'),     'utf8') + '\n' +
  fs.readFileSync(path.join(QALB_DIR, 'primitives.js'), 'utf8') + '\n' +
  'return Qlb;'
)(global, global.window);

// Redirect Qalb console to Node console
Qlb.console = console;

// ── Arabic-named Node.js extension primitives ─────────────────────────────────
//
// Naming convention: Arabic functional names, no transliteration.
// Each primitive is documented with its English meaning and purpose.

Qlb.globalEnvironment.merge({

  // ── File I/O ────────────────────────────────────────────────────────────────

  // اقرأ-جسون : read-json — load and parse a JSON file, return JS object
  'اقرأ-جسون': function(filePath) {
    const resolved = path.resolve(PROJECT_ROOT, filePath);
    console.log('  جاري التحميل:', resolved);
    return JSON.parse(fs.readFileSync(resolved, 'utf8'));
  },

  // اكتب-جسون : write-json — serialise JS object to JSON file
  'اكتب-جسون': function(filePath, data) {
    const resolved = path.resolve(PROJECT_ROOT, filePath);
    fs.writeFileSync(resolved, JSON.stringify(data, null, 2), 'utf8');
    console.log('  كُتب الملف:', resolved);
    return resolved;
  },

  // ── Object / dictionary access ───────────────────────────────────────────────

  // احصل : get — property access (احصل "مفتاح" كائن)
  'احصل': function(key, obj) {
    if (obj === null || obj === undefined) return null;
    return obj[key] !== undefined ? obj[key] : null;
  },

  // اضبط : set! — mutate object property, return object
  'اضبط': function(obj, key, val) {
    if (obj !== null && obj !== undefined) obj[key] = val;
    return obj;
  },

  // مفاتيح : keys — Object.keys
  'مفاتيح': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.keys(obj);
  },

  // قيم : values — Object.values
  'قيم': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.values(obj);
  },

  // مدخلات : entries — Object.entries → list of [key, value] pairs
  'مدخلات': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.entries(obj);
  },

  // كائن : object — create object from alternating key/value args
  // (كائن "أ" ١ "ب" ٢) → {"أ": 1, "ب": 2}
  'كائن': function() {
    const obj = {};
    for (let i = 0; i < arguments.length - 1; i += 2) {
      obj[arguments[i]] = arguments[i + 1];
    }
    return obj;
  },

  // أنشئ-كائن : new-object — empty JS object
  'أنشئ-كائن': function() { return {}; },

  // ── List / array extras ──────────────────────────────────────────────────────

  // رشح : filter — (رشح دالة-شرط قائمة)
  'رشح': function(pred, lst) {
    if (!Array.isArray(lst)) return [];
    return lst.filter(function(x) { return pred(x); });
  },

  // ادمج : flatten-concat — flatten one level (list of lists → list)
  'ادمج': function(lists) {
    if (!Array.isArray(lists)) return [];
    return [].concat.apply([], lists.filter(Array.isArray));
  },

  // حذف-مكررات : deduplicate — remove duplicates, drop nulls/undefineds
  'حذف-مكررات': function(lst) {
    if (!Array.isArray(lst)) return [];
    return [...new Set(lst.filter(function(x) {
      return x !== null && x !== undefined && x !== '';
    }))];
  },

  // أضف : append — add element to end of list (returns new list)
  'أضف': function(lst, item) {
    if (!Array.isArray(lst)) return [item];
    return lst.concat([item]);
  },

  // اختصر : reduce — (اختصر دالة بداية قائمة)
  'اختصر': function(fn, init, lst) {
    if (!Array.isArray(lst)) return init;
    return lst.reduce(function(acc, x) { return fn(acc, x); }, init);
  },

  // يحتوي؟ : contains? — list includes item
  'يحتوي؟': function(lst, item) {
    if (!Array.isArray(lst)) return false;
    return lst.indexOf(item) !== -1;
  },

  // ── String operations ────────────────────────────────────────────────────────

  // يبدأ-بـ : starts-with
  'يبدأ-بـ': function(str, prefix) {
    if (typeof str !== 'string') return false;
    return str.startsWith(prefix);
  },

  // يساوي-نص؟ : string-equal?
  'يساوي-نص؟': function(a, b) { return a === b; },

  // ── TMQ-specific helpers ─────────────────────────────────────────────────────

  // أضلاع-الفصيلة : edges-by-family — filter hyperedges dict by family name
  // Returns list of edge objects (values only, not keyed)
  'أضلاع-الفصيلة': function(edgesDict, family) {
    if (!edgesDict || typeof edgesDict !== 'object') return [];
    return Object.values(edgesDict).filter(function(e) {
      return e && typeof e === 'object' && e.family === family;
    });
  },

  // موقع-آية؟ : same-ayah? — do two loc arrays share surah+verse?
  'موقع-آية؟': function(locA, locB) {
    if (!locA || !locB) return false;
    return locA[0] === locB[0] && locA[1] === locB[1];
  },

  // عقد-في-آية-سريع : fast-nodes-in-ayah — O(1) lookup via pre-built index
  // Built below after TMQ is loaded; placeholder replaced at runtime.
  'عقد-في-آية-سريع': function(surah, ayah) { return []; },

  // صحيح : true literal
  'صحيح': true,

  // خطأ : false literal
  'خطأ': false,

  // لا-شيء : null literal
  'لا-شيء': null,

  // ── Logging ──────────────────────────────────────────────────────────────────

  // أعلن : announce — log a section header
  'أعلن': function(msg) {
    console.log('\n══ ' + msg + ' ══');
    return msg;
  },

});

// ── Pre-build ayah index (O(n) once, then O(1) per lookup) ────────────────────

const tmqData      = JSON.parse(fs.readFileSync(TMQ_PATH, 'utf8'));
const _nodeRegistry = tmqData.node_registry || {};
const _ayahIndex   = {};

for (var _id in _nodeRegistry) {
  var _node = _nodeRegistry[_id];
  if (_node && Array.isArray(_node.loc) && _node.loc.length >= 2) {
    var _key = _node.loc[0] + ':' + _node.loc[1];
    if (!_ayahIndex[_key]) _ayahIndex[_key] = [];
    _ayahIndex[_key].push(_id);
  }
}
console.log('فهرس الآيات — عدد الآيات المفهرسة:', Object.keys(_ayahIndex).length);

// Replace the placeholder with the real index lookup
Qlb.globalEnvironment.table['عقد-في-آية-سريع'] = function(surah, ayah) {
  return _ayahIndex[surah + ':' + ayah] || [];
};

// ── Execute compiler ──────────────────────────────────────────────────────────

console.log('بسم الله الرحمن الرحيم');
console.log('مجمّع المصحف — يبدأ التشغيل\n');

// Qalb's PEG grammar has no comment syntax — strip ; comments before executing
const rawCode = fs.readFileSync(COMPILER_FILE, 'utf8');
const code = rawCode.split('\n')
  .map(function(line) { return line.replace(/;.*$/, ''); })
  .join('\n');
Qlb.execute(code);
