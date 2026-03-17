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
  }
}

// ── Orb state machine ──────────────────────────────────────────────────────
const ORB_LABELS = {
  idle:     'IDLE',
  thinking: 'THINKING',
  haqq:     'HAQQ',
  qiyas:    'QIYAS',
  silence:  'SILENCE',
  chat:     'CHAT',
  waking:   'WAKING',
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

// ── Memory cards ───────────────────────────────────────────────────────────
function prependMemoryCard(entry) {
  const { type, number, mode, question, text, roots, timestamp } = entry;

  // Remove placeholder
  const placeholder = memoryList.querySelector('[data-placeholder]');
  if (placeholder) placeholder.remove();

  const card = document.createElement('div');
  card.className = `memory-card ${type || 'THOUGHT'}`;

  const rootChips = (roots || []).slice(0, 8).map(
    r => `<span class="root-chip">${escHtml(r)}</span>`
  ).join('');

  card.innerHTML = `
    <div class="memory-card-header">
      <span class="memory-type-badge">${escHtml(type || 'THOUGHT')}</span>
      <span class="memory-mode">${escHtml(mode || '')}</span>
      ${timestamp ? `<span class="memory-ts">${escHtml(timestamp)}</span>` : ''}
    </div>
    ${question ? `<div class="memory-question">${escHtml(question)}</div>` : ''}
    <div class="memory-text">${escHtml(text || '')}</div>
    ${rootChips ? `<div class="root-chips">${rootChips}</div>` : ''}
  `;

  // Click to expand/collapse full text
  card.addEventListener('click', () => card.classList.toggle('expanded'));

  memoryList.insertBefore(card, memoryList.firstChild);
  memoryList.scrollTop = 0;

  state.memoryCount++;
  statMemories.textContent = state.memoryCount;

  // Update panel header count
  const panelCount = document.getElementById('memory-panel-count');
  if (panelCount) panelCount.textContent = `(${state.memoryCount})`;
}

// ── Header stats + faculty health ──────────────────────────────────────────
const FACULTY_KEYS = ['clock','graph','owl','daemon','spectral','sparql','tmq','middleware'];

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
function appendChatBubble(role, text) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  // Render newlines (Maghrib seal, formatted responses)
  bubble.innerHTML = escHtml(text).replace(/\n/g, '<br>');
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
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
    if (data.response) {
      appendChatBubble('shahid', data.response);
    } else if (data.error) {
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
    headers: {'Content-Type': 'application/json'},
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
    headers: {'Content-Type': 'application/json'},
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
fetch('/constitution').then(r => r.json()).then(d => {
  (d.pending || []).forEach(p => handleConstitutionProposal(p));
}).catch(() => {});

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
