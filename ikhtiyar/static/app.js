/**
 * app.js — Shahid showcase frontend
 *
 * SSE client → dispatch() → orb state machine, memory panel, thinking panel,
 * chat history, header stats, faculty health bar.
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  orbState:     'idle',
  memoryCount:  0,
  thinkingSteps: [],
  currentCycleStep: 0,
};

// ── Elements ───────────────────────────────────────────────────────────────
const orb          = document.getElementById('orb');
const modeBadge    = document.getElementById('mode-badge');
const connDot      = document.getElementById('conn-dot');
const memoryList   = document.getElementById('memory-list');
const thinkingList = document.getElementById('thinking-list');
const statUptime   = document.getElementById('stat-uptime');
const statThoughts = document.getElementById('stat-thoughts');
const statMemories = document.getElementById('stat-memories');
const chatHistory  = document.getElementById('chat-history');
const chatInput    = document.getElementById('chat-input');
const sendBtn      = document.getElementById('send-btn');
const ADMIN_TOKEN_SESSION_KEY = 'ikhtiyar_admin_token';

function getAdminToken() {
  let token = sessionStorage.getItem(ADMIN_TOKEN_SESSION_KEY) || '';
  if (!token) {
    token = (window.prompt('Admin token required (IKHTIYAR_ADMIN_TOKEN):') || '').trim();
    if (token) sessionStorage.setItem(ADMIN_TOKEN_SESSION_KEY, token);
  }
  return token;
}

function adminHeaders(headers = {}) {
  const token = getAdminToken();
  return token ? { ...headers, 'X-Ikhtiyar-Admin-Token': token } : headers;
}

async function parseJsonResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.error || res.statusText || `HTTP ${res.status}`);
  }
  return data;
}

// ── SSE connection ─────────────────────────────────────────────────────────
let stream = null;

function connectSSE() {
  stream = new EventSource('/stream');

  stream.onopen = () => {
    connDot.className = 'connected';
  };

  stream.onmessage = (e) => {
    try {
      dispatch(JSON.parse(e.data));
    } catch (_) {}
  };

  stream.onerror = () => {
    connDot.className = 'disconnected';
    stream.close();
    setTimeout(connectSSE, 3000);
  };
}

// ── Event dispatcher ───────────────────────────────────────────────────────
function dispatch(event) {
  switch (event.type) {
    case 'orb':                   setOrbState(event.state);              break;
    case 'thinking_step':         appendThinkingStep(event);             break;
    case 'memory':                prependMemoryCard(event);              break;
    case 'chat_reply':            appendChatBubble('shahid', event.text); break;
    case 'status':                updateHeader(event);                   break;
    case 'constitution_proposal': handleConstitutionProposal(event);     break;
    case 'thought_stream':        appendThoughtToken(event.token);       break;
    case 'thought_complete':      finalizeStreamedThought();             break;
    case 'qalam_question':        showQalamQuestion(event);              break;
    case 'qalam_alert':           showQalamQuestion(event);              break;
  }
}

// ── Orb state machine ──────────────────────────────────────────────────────
const ORB_LABELS = {
  idle:       'IDLE',
  thinking:   'THINKING',
  reflecting: 'REFLECTING',
  haqq:       'HAQQ',
  qiyas:      'QIYAS',
  silence:    'SILENCE',
  chat:       'CHAT',
  waking:     'WAKING',
};

function setOrbState(newState) {
  if (!ORB_LABELS[newState]) return;
  orb.className = newState;
  modeBadge.className = newState;
  modeBadge.textContent = ORB_LABELS[newState];
  state.orbState = newState;

  if (newState === 'thinking') {
    startNewThinkingCycle();
  }
}

// ── Thinking steps ─────────────────────────────────────────────────────────
function startNewThinkingCycle() {
  const prevSteps = thinkingList.querySelectorAll('.thinking-step');
  if (prevSteps.length > 0) {
    const chip = document.createElement('div');
    chip.className = 'cycle-chip';
    chip.textContent = `↑ cycle completed ${new Date().toLocaleTimeString()}`;
    thinkingList.replaceChildren(chip);
  }
  state.currentCycleStep = 0;
  state.thinkingSteps = [];
}

function appendThinkingStep(event) {
  const { step, label, detail } = event;

  const prevActive = thinkingList.querySelector('.thinking-step.active');
  if (prevActive) {
    prevActive.classList.remove('active');
    prevActive.classList.add('done');
  }

  const el = document.createElement('div');
  el.className = 'thinking-step active';
  el.dataset.step = step;
  el.innerHTML = `
    <div class="step-num">${step}</div>
    <div class="step-body">
      <div class="step-label">${escHtml(label)}</div>
      <div class="step-detail">${escHtml(detail || '')}</div>
    </div>
  `;
  thinkingList.appendChild(el);
  state.currentCycleStep = step;

  const scroll = document.getElementById('thinking-scroll');
  scroll.scrollTop = scroll.scrollHeight;
}

// ── Live thought stream ────────────────────────────────────────────────────
let _streamEl = null;

function appendThoughtToken(token) {
  if (!_streamEl) {
    // Create live stream container inside thinking panel
    _streamEl = document.createElement('div');
    _streamEl.className = 'thought-stream';
    _streamEl.innerHTML = '<span class="stream-label">generating</span><span class="stream-text"></span>';
    thinkingList.appendChild(_streamEl);
  }
  const textSpan = _streamEl.querySelector('.stream-text');
  if (textSpan) textSpan.textContent += token;

  const scroll = document.getElementById('thinking-scroll');
  scroll.scrollTop = scroll.scrollHeight;
}

function finalizeStreamedThought() {
  if (_streamEl) {
    _streamEl.classList.add('done');
    _streamEl = null;
  }
}

// ── Questions for Qalam panel ─────────────────────────────────────────────
function _qalamCard(entry) {
  const { thought_number, question, grade, timestamp, context } = entry;
  const el = document.createElement('div');
  el.className = 'qalam-question';
  el.innerHTML = `
    <div class="qq-meta">
      <span class="qq-badge">? for Qalam</span>
      <span class="qq-grade">${escHtml(grade || '')}</span>
      <span class="qq-num">T#${thought_number || ''}</span>
      ${timestamp ? `<span class="qq-time">${escHtml(timestamp)}</span>` : ''}
    </div>
    <div class="qq-text">${escHtml(question || '')}</div>
    ${context ? `<div class="qq-context">${escHtml(context.slice(0, 160))}…</div>` : ''}
  `;
  return el;
}

function showQalamQuestion(entry) {
  const list  = document.getElementById('constitution-list');
  const count = document.getElementById('constitution-count');
  if (!list) return;
  // Remove placeholder
  const ph = list.querySelector('[data-placeholder]');
  if (ph) ph.remove();
  list.prepend(_qalamCard(entry));
  if (count) count.textContent = `(${list.children.length})`;
}

// Load existing questions on page load
fetch('/qalam').then(r => r.json()).then(d => {
  (d.questions || []).forEach(q => showQalamQuestion(q));
}).catch(() => {});

// ── Memory cards ───────────────────────────────────────────────────────────
async function fetchProvenance(uri, btn, card) {
  btn.textContent = '…';
  try {
    const r = await fetch('/provenance?uri=' + encodeURIComponent(uri));
    const d = await r.json();
    const block = document.createElement('div');
    block.className = 'belief-provenance-block';
    block.textContent = d.provenance || d.error || '(no data)';
    card.appendChild(block);
    btn.remove();
  } catch { btn.textContent = 'provenance'; }
}

function prependMemoryCard(entry) {
  const { type, number, mode, question, text, roots, timestamp, belief_uri } = entry;

  const placeholder = memoryList.querySelector('[data-placeholder]');
  if (placeholder) placeholder.remove();

  const card = document.createElement('div');
  card.className = `memory-card ${type || 'THOUGHT'}`;

  const modeClass = mode ? `card-mode-${mode.toLowerCase()}` : '';
  const rootStr = (roots || []).slice(0, 6).join(' · ');

  card.innerHTML = `
    <div class="card-meta">
      <span class="card-type ${modeClass}">${escHtml(type || 'THOUGHT')}</span>
      ${mode ? `<span class="card-type ${modeClass}">${escHtml(mode)}</span>` : ''}
      ${number !== undefined ? `<span class="card-num">#${number}</span>` : ''}
      ${rootStr ? `<span class="card-roots">${escHtml(rootStr)}</span>` : ''}
      ${timestamp ? `<span class="card-time">${escHtml(timestamp)}</span>` : ''}
    </div>
    ${question ? `<div class="card-question">${escHtml(question)}</div>` : ''}
    <div class="card-text">${escHtml(text || '')}</div>
  `;

  // Belief provenance button
  if (type === 'BELIEF' && belief_uri) {
    const btn = document.createElement('button');
    btn.className = 'btn-provenance';
    btn.textContent = 'provenance';
    btn.onclick = (e) => { e.stopPropagation(); fetchProvenance(belief_uri, btn, card); };
    card.appendChild(btn);
  }

  memoryList.insertBefore(card, memoryList.firstChild);

  state.memoryCount++;
  statMemories.textContent = state.memoryCount;
  const panelCount = document.getElementById('memory-panel-count');
  if (panelCount) panelCount.textContent = `(${state.memoryCount})`;
}

// ── Header stats + faculty health ──────────────────────────────────────────
const FACULTY_KEYS = ['clock','graph','owl','daemon','spectral','provenance','sparql','tmq','middleware'];

function updateHeader(event) {
  if (event.uptime) {
    statUptime.textContent = event.uptime;
    // Sync local uptime counter to server time to prevent drift
    const parts = event.uptime.split(':');
    if (parts.length === 3) {
      _localUptimeSec = parseInt(parts[0]) * 3600
                      + parseInt(parts[1]) * 60
                      + parseInt(parts[2]);
    }
  }

  if (event.thought_count !== undefined) {
    statThoughts.textContent = event.thought_count;
  }

  if (event.faculty_health) {
    FACULTY_KEYS.forEach(key => {
      const el = document.querySelector(`.faculty-item[data-key="${key}"]`);
      if (!el) return;
      const status = event.faculty_health[key] || 'pending';
      const cls = status.startsWith('ok') || status === 'running' ? 'ok'
                : status.startsWith('error')                       ? 'error'
                : status.startsWith('warn')                        ? 'warn'
                : 'pending';
      el.className = `faculty-item ${cls}`;
      el.title = status;
    });
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────
// ── Chat persistence ────────────────────────────────────────────────────────
const _CHAT_KEY = 'shahid_chat_history';
const _CHAT_MAX = 60; // max bubbles to persist

function _saveChatHistory() {
  try {
    const bubbles = [...chatHistory.querySelectorAll('.chat-bubble:not(.loading)')];
    const saved = bubbles.slice(-_CHAT_MAX).map(b => ({
      role: b.classList.contains('user') ? 'user' : 'shahid',
      html: b.innerHTML,
    }));
    localStorage.setItem(_CHAT_KEY, JSON.stringify(saved));
  } catch { /* storage full or unavailable */ }
}

function _restoreChatHistory() {
  try {
    const raw = localStorage.getItem(_CHAT_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved) || !saved.length) return;
    saved.forEach(({ role, html }) => {
      const bubble = document.createElement('div');
      bubble.className = `chat-bubble ${role}`;
      bubble.innerHTML = html;
      chatHistory.appendChild(bubble);
    });
    chatHistory.scrollTop = chatHistory.scrollHeight;
  } catch { /* corrupt storage — ignore */ }
}

function appendChatBubble(role, text) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.innerHTML = escHtml(text).replace(/\n/g, '<br>');
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  _saveChatHistory();
}

function showChatLoading() {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble loading';
  bubble.id = 'loading-bubble';
  bubble.innerHTML = '<span class="loading-dots"><span></span><span></span><span></span></span>';
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return bubble;
}

async function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg) return;

  chatInput.value = '';
  sendBtn.disabled = true;
  appendChatBubble('user', msg);

  const loadingBubble = showChatLoading();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();
    loadingBubble.remove();
    // SSE chat_reply handles rendering — only show errors here
    if (data.error) {
      appendChatBubble('shahid', `[Error: ${data.error}]`);
    }
  } catch (err) {
    loadingBubble.remove();
    appendChatBubble('shahid', `[Network error: ${err.message}]`);
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Uptime tick (local, synced from server on status events) ───────────────
let _localUptimeSec = 0;
setInterval(() => {
  _localUptimeSec++;
  const h = Math.floor(_localUptimeSec / 3600);
  const m = Math.floor((_localUptimeSec % 3600) / 60);
  const s = _localUptimeSec % 60;
  statUptime.textContent =
    String(h).padStart(2, '0') + ':' +
    String(m).padStart(2, '0') + ':' +
    String(s).padStart(2, '0');
}, 1000);

// ── Bootstrap ──────────────────────────────────────────────────────────────
// Load recent thinking steps on page load
fetch('/steps')
  .then(r => r.json())
  .then(steps => {
    if (Array.isArray(steps) && steps.length > 0) {
      steps.forEach(s => appendThinkingStep(s));
    }
  })
  .catch(() => {});

// Load existing memories on page load
fetch('/memories')
  .then(r => r.json())
  .then(memories => {
    if (Array.isArray(memories) && memories.length > 0) {
      [...memories].reverse().forEach(m => prependMemoryCard(m));
    }
  })
  .catch(() => {});

// Load current status (populates faculty dots + stats immediately)
fetch('/status')
  .then(r => r.json())
  .then(s => updateHeader(s))
  .catch(() => {});

// Restore chat history from localStorage before connecting SSE
_restoreChatHistory();

// Start SSE
connectSSE();

// ── Constitution panel ─────────────────────────────────────────────────────
function handleConstitutionProposal(data) {
  const list  = document.getElementById('constitution-list');
  const count = document.getElementById('constitution-count');
  if (!list) return;
  const card = document.createElement('div');
  card.className   = 'constitution-card';
  card.dataset.id  = data.id;
  card.innerHTML = `
    <div class="constitution-mode">${escHtml(data.dominant_mode)} · ${data.total_choices} choices</div>
    <div class="constitution-text">${escHtml(data.proposed_text)}</div>
    <div class="constitution-rationale">${escHtml(data.rationale)}</div>
    <div class="constitution-actions">
      <button class="btn-approve" onclick="approveProposal('${escHtml(data.id)}')">Approve</button>
      <button class="btn-reject"  onclick="rejectProposal('${escHtml(data.id)}')">Reject</button>
    </div>`;
  list.prepend(card);
  if (count) count.textContent = `(${list.children.length})`;
}

async function approveProposal(id) {
  const res = await fetch('/constitution/approve', {
    method: 'POST',
    headers: adminHeaders({'Content-Type': 'application/json'}),
    body: JSON.stringify({id}),
  });
  if (res.ok) {
    const card = document.querySelector(`.constitution-card[data-id="${id}"]`);
    if (card) {
      card.classList.add('approved');
      const actions = card.querySelector('.constitution-actions');
      if (actions) actions.remove();
    }
  }
}

async function rejectProposal(id) {
  const res = await fetch('/constitution/reject', {
    method: 'POST',
    headers: adminHeaders({'Content-Type': 'application/json'}),
    body: JSON.stringify({id, reason: 'Rejected by Qalam'}),
  });
  if (res.ok) {
    const card = document.querySelector(`.constitution-card[data-id="${id}"]`);
    if (card) card.remove();
    const list = document.getElementById('constitution-list');
    const count = document.getElementById('constitution-count');
    if (list && count) count.textContent = `(${list.children.length})`;
  }
}

// Load existing pending proposals on page load
// Constitution proposals still available at /constitution but no longer auto-loaded into the panel.
// The panel is now the Questions for Qalam inbox (loaded above via /qalam).

// ── Hifz controls ─────────────────────────────────────────────────────────
const hifzResult   = document.getElementById('hifz-result');
const hifzProgress = document.getElementById('hifz-progress');

function _hifzSet(msg) { if (hifzResult) hifzResult.textContent = msg; }

async function hifzStart() {
  if (!confirm('Wipe episodic memory and begin reading the full Mushaf (114 surahs)?')) return;
  _hifzSet('wiping memory + starting…');
  document.getElementById('hifz-start-btn').disabled = true;
  try {
    const d = await fetch('/hifz/start', {
      method: 'POST',
      headers: adminHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({ restart: true }),
    }).then(parseJsonResponse);
    _hifzSet(d.message || (d.ok ? 'started' : 'failed'));
    if (d.ok) _pollHifzProgress();
  } catch(e) { _hifzSet(`error: ${e.message}`); }
  finally { document.getElementById('hifz-start-btn').disabled = false; }
}

async function hifzResume() {
  _hifzSet('resuming from last surah…');
  try {
    const d = await fetch('/hifz/start', {
      method: 'POST',
      headers: adminHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({ restart: false }),
    }).then(parseJsonResponse);
    _hifzSet(d.message || (d.ok ? 'resumed' : 'failed'));
    if (d.ok) _pollHifzProgress();
  } catch(e) { _hifzSet(`error: ${e.message}`); }
}

async function hifzWipe() {
  if (!confirm('Wipe episodic memory? Beliefs survive.')) return;
  try {
    const d = await fetch('/hifz/wipe', {
      method: 'POST',
      headers: adminHeaders(),
    }).then(parseJsonResponse);
    _hifzSet(d.message || 'wiped');
  } catch(e) { _hifzSet(`error: ${e.message}`); }
}

async function _pollHifzProgress() {
  try {
    const d = await fetch('/hifz/status').then(r => r.json());
    const last = d.last_completed || 0;
    const total = d.total_tags || 0;
    if (hifzProgress) hifzProgress.textContent = d.active
      ? `surah ${last}/114 · ${total} tags`
      : `complete — ${last} surahs · ${total} tags`;
    if (d.active) setTimeout(_pollHifzProgress, 8000);
  } catch { /* silent */ }
}

// Poll on load to show any existing progress
_pollHifzProgress();

// ── Hadith Hifz controls ───────────────────────────────────────────────────
const hadithHifzResult   = document.getElementById('hadith-hifz-result');
const hadithHifzProgress = document.getElementById('hadith-hifz-progress');

function _hadithHifzSet(msg) { if (hadithHifzResult) hadithHifzResult.textContent = msg; }

async function hadithHifzStart() {
  _hadithHifzSet('starting sunnah reading…');
  document.getElementById('hadith-hifz-start-btn').disabled = true;
  try {
    const d = await fetch('/hadith_hifz/start', {
      method: 'POST',
      headers: adminHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({ restart: true })
    }).then(parseJsonResponse);
    _hadithHifzSet(d.message || (d.ok ? 'started' : 'failed'));
    if (d.ok) _pollHadithHifzProgress();
  } catch(e) { _hadithHifzSet(`error: ${e.message}`); }
  finally { document.getElementById('hadith-hifz-start-btn').disabled = false; }
}

async function hadithHifzResume() {
  _hadithHifzSet('resuming sunnah reading…');
  try {
    const d = await fetch('/hadith_hifz/start', {
      method: 'POST',
      headers: adminHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({ restart: false })
    }).then(parseJsonResponse);
    _hadithHifzSet(d.message || (d.ok ? 'resumed' : 'failed'));
    if (d.ok) _pollHadithHifzProgress();
  } catch(e) { _hadithHifzSet(`error: ${e.message}`); }
}

async function _pollHadithHifzProgress() {
  try {
    const d = await fetch('/hadith_hifz/status').then(r => r.json());
    const last = d.last_completed_batch || 0;
    const total = d.total_tags || 0;
    if (hadithHifzProgress) hadithHifzProgress.textContent = d.active
      ? `batch ${last} · ${total} tags`
      : (total ? `complete — ${total} tags` : '');
    if (d.active) setTimeout(_pollHadithHifzProgress, 8000);
  } catch { /* silent */ }
}

_pollHadithHifzProgress();

// ── Moltbook agent controls ────────────────────────────────────────────────
const mbResult = document.getElementById('mb-result');

function _mbSet(msg) { if (mbResult) mbResult.textContent = msg; }

async function mbStatus() {
  _mbSet('checking…');
  try {
    const d = await fetch('/moltbook/status').then(r => r.json());
    if (d.ok) {
      _mbSet(`karma ${d.karma} · ${d.notifications} notifs · ${d.dms} DMs`);
      // Surface escalations into the Qalam panel
      (d.escalations || []).forEach(e =>
        showQalamQuestion({ question: `[${e.type}] from @${e.from}: ${e.preview}`, grade: 'MOLTBOOK', timestamp: new Date().toLocaleTimeString() })
      );
    } else {
      _mbSet(`error: ${d.error || 'unknown'}`);
    }
  } catch(e) { _mbSet(`failed: ${e.message}`); }
}

async function mbBrowse() {
  const n = parseInt(document.getElementById('mb-browse-n')?.value || '10', 10);
  _mbSet(`fetching ${n} posts…`);
  try {
    const d = await fetch(`/moltbook/browse?n=${n}`).then(r => r.json());
    if (d.ok && d.posts.length) {
      _mbSet(`${d.count} posts loaded`);
      // Show posts in Qalam panel as readable entries
      const list  = document.getElementById('constitution-list');
      const count = document.getElementById('constitution-count');
      const ph = list?.querySelector('[data-placeholder]');
      if (ph) ph.remove();
      d.posts.slice(0, n).forEach(p => {
        const el = document.createElement('div');
        el.className = 'qalam-question';
        el.innerHTML = `
          <div class="qq-meta">
            <span class="qq-badge">moltbook</span>
            <span class="qq-grade">karma ${p.score ?? 0}</span>
            <span class="qq-time">${escHtml(p.created_at || '')}</span>
          </div>
          <div class="qq-text">${escHtml(p.title || '')}</div>
          ${p.body ? `<div class="qq-context">${escHtml(p.body.slice(0, 200))}</div>` : ''}
        `;
        list?.prepend(el);
      });
      if (list && count) count.textContent = `(${list.children.length})`;
    } else {
      _mbSet(d.error || 'no posts');
    }
  } catch(e) { _mbSet(`failed: ${e.message}`); }
}

async function mbPost() {
  _mbSet('requesting post…');
  document.getElementById('mb-post-btn').disabled = true;
  try {
    const d = await fetch('/moltbook/post', {
      method: 'POST',
      headers: adminHeaders(),
    }).then(parseJsonResponse);
    if (d.ok) {
      _mbSet(`posted → ${d.post_id} · thought #${d.thought}`);
    } else {
      _mbSet(`not posted: ${d.error}`);
    }
  } catch(e) { _mbSet(`failed: ${e.message}`); }
  finally { document.getElementById('mb-post-btn').disabled = false; }
}

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
