/**
 * run_active_commands.js — Node.js bootstrapper for active_command_compiler.qlb
 *
 * Loads full_quran_constitution.json, runs the Qalb active command compiler,
 * outputs ikhtiyar/active_command_set.json.
 *
 * Usage:
 *   node ikhtiyar/core/run_active_commands.js
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ── Resolve paths ─────────────────────────────────────────────────────────────

const PROJECT_ROOT  = path.resolve(__dirname, '..', '..');
const QALB_DIR      = path.join(PROJECT_ROOT, 'Qalb', '----master', 'public', 'qlb');
const COMPILER_FILE = path.join(__dirname, 'active_command_compiler.qlb');

// ── Load Qalb runtime ─────────────────────────────────────────────────────────

global.window = { setTimeout: setTimeout };

const Qlb = new Function('global', 'window',
  fs.readFileSync(path.join(QALB_DIR, 'qlb.js'),        'utf8') + '\n' +
  fs.readFileSync(path.join(QALB_DIR, 'parser.js'),     'utf8') + '\n' +
  fs.readFileSync(path.join(QALB_DIR, 'primitives.js'), 'utf8') + '\n' +
  'return Qlb;'
)(global, global.window);

Qlb.console = console;

// ── Shared primitives (identical to run_compiler.js) ──────────────────────────

Qlb.globalEnvironment.merge({

  // ── File I/O ────────────────────────────────────────────────────────────────

  'اقرأ-جسون': function(filePath) {
    const resolved = path.resolve(PROJECT_ROOT, filePath);
    console.log('  جاري التحميل:', resolved);
    return JSON.parse(fs.readFileSync(resolved, 'utf8'));
  },

  'اكتب-جسون': function(filePath, data) {
    const resolved = path.resolve(PROJECT_ROOT, filePath);
    fs.writeFileSync(resolved, JSON.stringify(data, null, 2), 'utf8');
    console.log('  كُتب:', resolved);
    return resolved;
  },

  // ── Object / dictionary ──────────────────────────────────────────────────────

  'احصل': function(key, obj) {
    if (obj === null || obj === undefined) return null;
    return obj[key] !== undefined ? obj[key] : null;
  },

  'اضبط': function(obj, key, val) {
    if (obj !== null && obj !== undefined) obj[key] = val;
    return obj;
  },

  'مفاتيح': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.keys(obj);
  },

  'قيم': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.values(obj);
  },

  'مدخلات': function(obj) {
    if (!obj || typeof obj !== 'object') return [];
    return Object.entries(obj);
  },

  'كائن': function() {
    const obj = {};
    for (let i = 0; i < arguments.length - 1; i += 2) {
      obj[arguments[i]] = arguments[i + 1];
    }
    return obj;
  },

  'أنشئ-كائن': function() { return {}; },

  // ── List extras ──────────────────────────────────────────────────────────────

  'رشح': function(pred, lst) {
    if (!Array.isArray(lst)) return [];
    return lst.filter(function(x) { return pred(x); });
  },

  'خريطة': function(fn, lst) {
    if (!Array.isArray(lst)) return [];
    return lst.map(function(x) { return fn(x); });
  },

  // طبق : apply/map — Qalb's actual map keyword
  'طبق': function(fn, lst) {
    if (!Array.isArray(lst)) return [];
    return lst.map(function(x) { return fn(x); });
  },

  'ادمج': function(lists) {
    if (!Array.isArray(lists)) return [];
    return [].concat.apply([], lists.filter(Array.isArray));
  },

  'حذف-مكررات': function(lst) {
    if (!Array.isArray(lst)) return [];
    return [...new Set(lst.filter(function(x) {
      return x !== null && x !== undefined && x !== '';
    }))];
  },

  'أضف': function(lst, item) {
    if (!Array.isArray(lst)) return [item];
    return lst.concat([item]);
  },

  'اختصر': function(fn, init, lst) {
    if (!Array.isArray(lst)) return init;
    return lst.reduce(function(acc, x) { return fn(acc, x); }, init);
  },

  'يحتوي؟': function(lst, item) {
    if (!Array.isArray(lst)) return false;
    return lst.indexOf(item) !== -1;
  },

  'يبدأ-بـ': function(str, prefix) {
    if (typeof str !== 'string') return false;
    return str.startsWith(prefix);
  },

  'يساوي-نص؟': function(a, b) { return a === b; },

  // ── New primitives for active command compiler ────────────────────────────────

  // جمع-حسب : group-by — group list by a key function, return {key: [items]}
  // (جمع-حسب دالة-المفتاح قائمة)
  'جمع-حسب': function(keyFn, lst) {
    if (!Array.isArray(lst)) return {};
    const result = {};
    for (let i = 0; i < lst.length; i++) {
      const k = keyFn(lst[i]);
      if (k === null || k === undefined) continue;
      if (!result[k]) result[k] = [];
      result[k].push(lst[i]);
    }
    return result;
  },

  // طول : length — list or object key count
  'طول': function(lst) {
    if (lst === null || lst === undefined) return 0;
    if (Array.isArray(lst)) return lst.length;
    if (typeof lst === 'object') return Object.keys(lst).length;
    return 0;
  },

  // اسقط : take — first N elements of list
  // (اسقط عدد قائمة)
  'اسقط': function(n, lst) {
    if (!Array.isArray(lst)) return [];
    return lst.slice(0, n);
  },

  // ── Literals ─────────────────────────────────────────────────────────────────

  'صحيح':  true,
  'خطأ':   false,
  'لا-شيء': null,

  // ── Logging ──────────────────────────────────────────────────────────────────

  'أعلن': function(msg) {
    console.log('\n══ ' + msg + ' ══');
    return msg;
  },

});

// ── Execute ───────────────────────────────────────────────────────────────────

console.log('بسم الله الرحمن الرحيم');
console.log('مجمّع الأوامر النشطة — يبدأ التشغيل\n');

const rawCode = fs.readFileSync(COMPILER_FILE, 'utf8');
const code = rawCode.replace(/\r/g, '').split('\n')
  .map(function(line) { return line.replace(/;.*$/, ''); })
  .join('\n');

Qlb.execute(code);
