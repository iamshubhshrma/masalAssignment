/* Masal Leads — front end. Talks to the FastAPI backend; no framework, no build step. */
'use strict';

const $ = (id) => document.getElementById(id);
function on(id, ev, fn) {
  const el = $(id);
  if (!el) { console.warn(`[masal] #${id} missing — hard-reload the page`); return; }
  el.addEventListener(ev, fn);
}

const state = { leads: [], stats: {}, health: {}, filter: 'all', selected: null, busy: false };

const SAMPLE_LEAD = {
  name: 'Rahul Mehta', location: 'Whitefield, Bangalore',
  property_requirement: '3BHK apartment, east-facing, high floor',
  budget: '1.4 crore', timeline: 'Within 2 months',
  phone: '+919876543210', source: 'Website',
  message: "Saw Aurum Heights on your site. I'm pre-approved for a home loan and want to visit this Saturday if possible. Looking for east-facing, high floor. Also need to know about the clubhouse.",
};

const SAMPLE_BULK = `[11/09, 9:14 am] +91 98765 43210: Hi, saw your ad for Aurum Heights. Looking for a 3BHK around 1.4 cr.
[11/09, 9:15 am] +91 98765 43210: Can visit this Saturday. I'm Rahul btw
[11/09, 9:40 am] You: Sure sir, I'll arrange it
[11/09, 10:02 am] Meera Nair: hello, do you have any 2bhk in sarjapur under 80 lakhs? not urgent, maybe next year
[11/09, 10:30 am] Amazon: Your package has been delivered
[11/09, 11:11 am] +91 99000 11223: need office space indiranagar 2000 sqft, budget 2.5cr, cash ready, want to close this month. - Imran Qureshi
[11/09, 11:45 am] OTP 449201 is your verification code
[11/09, 12:20 pm] Sneha: saw the villa listing. what's included? planning to move in 6 months, budget flexible around 2cr`;

/* ───────── helpers ───────── */
function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg; el.className = 'toast ' + kind; el.hidden = false;
  clearTimeout(el._t); el._t = setTimeout(() => { el.hidden = true; }, 4200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  let body = null;
  try { body = await res.json(); } catch { /* no body */ }
  if (!res.ok) {
    const d = body?.error || body?.detail || res.statusText;
    const err = new Error(typeof d === 'string' ? d : JSON.stringify(d));
    err.status = res.status;
    throw err;
  }
  return body;
}

/* The server restarting drops the in-memory store, so an open tab can still be
   showing leads that no longer exist. Rather than surfacing a bare "unknown
   lead", resync from the server and say what happened. */
async function handleError(e, fallbackMsg) {
  if (e.status === 404) {
    try {
      await loadAll();
      state.selected = null;
      render();
      toast('That lead no longer exists on the server — list refreshed.', 'err');
      return;
    } catch { /* fall through to the plain message */ }
  }
  toast(fallbackMsg ? `${fallbackMsg}: ${e.message}` : e.message, 'err');
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const INTENT = {
  ready_to_buy: 'Ready to buy', actively_searching: 'Actively searching',
  exploring: 'Exploring', investment: 'Investment', rental: 'Rental',
  price_shopping: 'Price shopping', not_serious: 'Not serious', unclear: 'Unclear',
};

/* ───────── render: chrome ───────── */
function renderBadges() {
  const h = state.health;
  const ai = h.ai_configured
    ? `<span class="badge ok"><span class="dot"></span>AI <b>${esc(h.primary_model || '')}</b></span>`
    : `<span class="badge bad"><span class="dot"></span>No AI key configured</span>`;
  const chain = (h.ai_chain || []).length > 1
    ? `<span class="badge">fallback <b>${esc(h.ai_chain.slice(1).join(' → '))}</b></span>` : '';
  const voice = h.voice_enabled
    ? `<span class="badge">voice <b>${h.voice_mock ? 'demo' : 'live'}</b></span>` : '';
  $('badges').innerHTML = ai + chain + voice +
    `<span class="badge">build <b>${esc(h.ui_build || '?')}</b></span>`;
}

function renderStats() {
  const s = state.stats || {};
  const cards = [
    ['Leads', s.total ?? 0, ''],
    ['Hot', s.hot ?? 0, 'hot'],
    ['Warm', s.warm ?? 0, 'warm'],
    ['Cold', s.cold ?? 0, 'cold'],
    ['Act today', s.act_today ?? 0, 'accent'],
    ['Avg score', s.avg_score ?? 0, 'accent'],
  ];
  $('stats').innerHTML = cards.map(([k, v, c]) =>
    `<div class="stat ${c}"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join('');
}

/* ───────── render: list ───────── */
function matches(lead) {
  const a = lead.analysis;
  switch (state.filter) {
    case 'hot': case 'warm': case 'cold': return a?.temperature === state.filter;
    case 'today': return a?.urgency === 'high';
    default: return true;
  }
}

function ringHtml(a) {
  if (!a) return `<div class="ring none" style="--p:0"><span>…</span></div>`;
  return `<div class="ring ${a.temperature}" style="--p:${a.priority_score}">
            <span>${a.priority_score}</span></div>`;
}

function leadRow(lead) {
  const a = lead.analysis;
  const t = a ? a.temperature : 'none';
  const sel = state.selected === lead.id ? 'sel' : '';
  const sub = [lead.location, lead.budget, lead.timeline].filter(Boolean).join(' · ');

  const tags = [];
  if (a) {
    tags.push(`<span class="pill ${a.temperature}">${a.temperature}</span>`);
    if (a.urgency === 'high') tags.push('<span class="pill urgent">act today</span>');
    tags.push(`<span class="pill soft">${esc(INTENT[a.intent] || a.intent)}</span>`);
    if (a.objections?.length) tags.push(`<span class="pill soft">${a.objections.length} objection${a.objections.length > 1 ? 's' : ''}</span>`);
  } else if (lead.analysis_error) {
    tags.push('<span class="pill urgent">analysis failed</span>');
  } else {
    tags.push('<span class="pill pending"><span class="spin"></span>analysing</span>');
  }
  if (lead.voice?.call_id) {
    const v = lead.voice;
    tags.push(v.outcome
      ? `<span class="pill ok">called</span>`
      : `<span class="pill pending"><span class="spin"></span>calling</span>`);
  }

  const summary = a?.summary || lead.analysis_error || lead.message || 'No message provided.';
  return `<article class="lead t-${t} ${sel}" data-lead="${lead.id}">
    ${ringHtml(a)}
    <div class="lead-main">
      <div class="lead-top"><span class="lead-name">${esc(lead.name)}</span>
        <span class="lead-sub">${esc(sub)}</span></div>
      <div class="lead-sum">${esc(summary)}</div>
      <div class="lead-tags">${tags.join('')}</div>
    </div>
  </article>`;
}

function renderList() {
  const rows = state.leads.filter(matches);
  $('lead-count').textContent = rows.length;
  $('empty').hidden = state.leads.length > 0;
  $('leads').innerHTML = rows.map(leadRow).join('');
}

/* ───────── render: detail ───────── */
function detailHtml(lead) {
  const a = lead.analysis;
  const sub = [lead.phone, lead.source].filter(Boolean).join(' · ');

  if (!a) {
    return `<div class="d-head">${ringHtml(null)}
      <div><div class="d-name">${esc(lead.name)}</div><div class="d-sub">${esc(sub)}</div></div></div>
      <div class="sect"><p>${lead.analysis_error
        ? 'Analysis failed: ' + esc(lead.analysis_error)
        : 'Analysing this lead…'}</p></div>
      <div class="d-foot">
        <button class="ghost" data-reanalyze="${lead.id}" type="button">Retry analysis</button>
        <button class="ghost danger" data-delete="${lead.id}" type="button">Delete</button>
      </div>`;
  }

  const tags = [
    `<span class="pill ${a.temperature}">${a.temperature} · ${a.priority_score}</span>`,
    `<span class="pill soft">${esc(INTENT[a.intent] || a.intent)}</span>`,
    a.urgency === 'high' ? '<span class="pill urgent">act today</span>'
      : `<span class="pill soft">${esc(a.urgency)} urgency</span>`,
  ].join('');

  const sect = (title, inner) => `<div class="sect"><h3>${title}</h3>${inner}</div>`;
  const chips = (arr, cls) => arr?.length
    ? `<div class="tags">${arr.map((x) => `<span class="tag ${cls}">${esc(x)}</span>`).join('')}</div>`
    : '<p class="hint">None identified.</p>';

  const v = lead.voice || {};
  const voiceSection = v.call_id ? sect('Voice call', `
      <p class="hint">${esc(v.status)}${v.sub_status ? ' · ' + esc(v.sub_status) : ''}${v.duration ? ' · ' + v.duration + 's' : ''}</p>
      ${v.transcript?.length ? `<div class="transcript" style="margin-top:9px">${v.transcript.map((t) =>
        `<div class="turn ${t.speaker}"><span class="who">${t.speaker}</span><span>${esc(t.text)}</span></div>`).join('')}</div>` : ''}
      ${v.outcome ? `<p class="why">${esc(v.outcome.notes)}</p>` : ''}`) : '';

  const canCall = state.health.voice_enabled && lead.phone && !v.call_id;

  return `
    <div class="d-head">${ringHtml(a)}
      <div><div class="d-name">${esc(lead.name)}</div><div class="d-sub">${esc(sub)}</div></div></div>
    <div class="d-tags">${tags}</div>

    ${sect('Summary', `<p>${esc(a.summary)}</p>
      ${a.score_reasoning ? `<p class="why">Why ${a.priority_score}: ${esc(a.score_reasoning)}</p>` : ''}`)}
    ${sect('Customer intent', `<p>${esc(INTENT[a.intent] || a.intent)}${a.intent_detail ? ' — ' + esc(a.intent_detail) : ''}</p>`)}
    ${sect('Key requirements', chips(a.key_requirements, ''))}
    ${sect('Objections &amp; concerns', chips(a.objections, 'obj'))}
    ${sect('Recommended next action', `<div class="action-box">
        <div class="when">${esc(a.next_action_timing || 'next')}</div>
        <p>${esc(a.next_action)}</p></div>`)}
    ${sect('Suggested response', `<div class="reply">
        <button class="copy ghost" data-copy="${lead.id}" type="button">Copy</button>
        <p id="reply-text">${esc(a.suggested_response)}</p></div>`)}
    ${a.missing_info?.length ? sect('Still worth asking', chips(a.missing_info, 'ask')) : ''}
    ${voiceSection}

    <div class="chat">
      <h3>Ask about this lead</h3>
      <div class="chat-log" id="chat-log">${(lead.chat || []).map((t) =>
        `<div class="msg ${t.role}">${esc(t.content)}${t.engine
          ? `<span class="eng">${esc(t.engine)}</span>` : ''}</div>`).join('')
        || '<p class="hint">Ask anything about this lead — answers are grounded in their details, not a generic chatbot.</p>'}</div>
      <div class="sugg">${(state.health.chat_suggestions || []).map((s) =>
        `<button class="ghost" data-sugg="${esc(s)}" type="button">${esc(s)}</button>`).join('')}</div>
      <form class="chat-form" id="chat-form">
        <textarea id="chat-input" rows="1" placeholder="e.g. what should I emphasise on the call?"></textarea>
        <button class="primary solid" type="submit" style="flex:none">Ask</button>
      </form>
    </div>

    <div class="d-foot">
      ${canCall ? `<button class="ghost" data-call="${lead.id}" type="button">AI confirmation call</button>` : ''}
      <button class="ghost" data-reanalyze="${lead.id}" type="button">Re-analyse</button>
      <button class="ghost danger" data-delete="${lead.id}" type="button">Delete</button>
      <span class="engine">${esc(a.engine)}</span>
    </div>`;
}

function renderDetail() {
  const panel = $('detail');
  const lead = state.leads.find((l) => l.id === state.selected);
  if (!lead) { panel.hidden = true; return; }
  panel.hidden = false;
  $('detail-body').innerHTML = detailHtml(lead);
  const log = $('chat-log');
  if (log) log.scrollTop = log.scrollHeight;
}

function render() { renderBadges(); renderStats(); renderList(); renderDetail(); }

/* ───────── data ───────── */
async function loadAll() {
  const [h, d] = await Promise.all([api('/api/health'), api('/api/leads')]);
  state.health = h; state.leads = d.leads || []; state.stats = d.stats || {};
}

function applyPayload(d) {
  if (d.leads) state.leads = d.leads;
  if (d.stats) state.stats = d.stats;
  if (d.lead) {
    const i = state.leads.findIndex((l) => l.id === d.lead.id);
    if (i >= 0) state.leads[i] = d.lead; else state.leads.unshift(d.lead);
  }
}

/* ───────── events ───────── */
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === t));
  document.querySelectorAll('.tabpane').forEach((p) =>
    p.classList.toggle('active', p.id === 'form-' + t.dataset.tab));
}));

on('btn-sample', 'click', () => {
  const f = $('form-single');
  Object.entries(SAMPLE_LEAD).forEach(([k, v]) => { if (f[k]) f[k].value = v; });
  $('single-hint').className = 'hint ok';
  $('single-hint').textContent = 'Example filled — press “Analyse lead”.';
});

on('form-single', 'submit', async (ev) => {
  ev.preventDefault();
  const btn = $('btn-add');
  const data = Object.fromEntries(new FormData(ev.target).entries());
  if (!data.name?.trim()) return toast('Name is required', 'err');

  btn.disabled = true; btn.textContent = 'Analysing…';
  try {
    const out = await api('/api/leads', { method: 'POST', body: JSON.stringify(data) });
    applyPayload(out);
    state.selected = out.lead.id;
    ev.target.reset();
    $('single-hint').className = 'hint';
    $('single-hint').textContent = 'The AI scores the lead the moment you save it.';
    render();
    toast(`Scored ${out.lead.analysis?.priority_score ?? '—'}/100 — ${out.lead.name}`, 'ok');
  } catch (e) { toast(e.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = 'Analyse lead'; }
});

on('btn-bulk-sample', 'click', () => { $('bulk-text').value = SAMPLE_BULK; });

on('btn-triage', 'click', async (ev) => {
  const btn = ev.currentTarget;
  const text = $('bulk-text').value.trim();
  if (!text) return toast('Paste some raw enquiries first', 'err');

  btn.disabled = true; btn.textContent = 'Triaging…';
  $('bulk-hint').className = 'hint';
  $('bulk-hint').textContent = 'Splitting and scoring — this takes a few seconds for a big paste.';
  try {
    const out = await api('/api/triage', { method: 'POST', body: JSON.stringify({ text }) });
    applyPayload(out); render();
    if (out.added) { $('bulk-text').value = ''; toast(`Triaged ${out.added} lead(s)`, 'ok'); }
    else toast(out.message || 'No enquiries found', 'err');
    $('bulk-hint').textContent = 'Skips OTPs, delivery alerts and your own outgoing messages.';
  } catch (e) { toast(e.message, 'err'); $('bulk-hint').className = 'hint err'; $('bulk-hint').textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = 'Triage & score'; }
});

on('filters', 'click', (ev) => {
  const b = ev.target.closest('[data-filter]'); if (!b) return;
  state.filter = b.dataset.filter;
  document.querySelectorAll('#filters .chip').forEach((c) => c.classList.toggle('active', c === b));
  renderList();
});

on('leads', 'click', (ev) => {
  const card = ev.target.closest('[data-lead]'); if (!card) return;
  state.selected = state.selected === card.dataset.lead ? null : card.dataset.lead;
  render();
});

on('detail-close', 'click', () => { state.selected = null; render(); });

on('btn-clear', 'click', async () => {
  if (!confirm('Delete every lead and its analysis?')) return;
  try {
    const out = await api('/api/leads', { method: 'DELETE' });
    state.selected = null; state.leads = []; state.stats = out.stats || {};
    render(); toast(`Cleared ${out.cleared} lead(s)`, 'ok');
  } catch (e) { toast(e.message, 'err'); }
});

/* detail panel — delegated */
on('detail-body', 'click', async (ev) => {
  const copy = ev.target.closest('[data-copy]');
  if (copy) {
    const text = $('reply-text')?.textContent || '';
    try { await navigator.clipboard.writeText(text); toast('Reply copied', 'ok'); }
    catch { toast('Copy failed — select the text manually', 'err'); }
    return;
  }

  const sugg = ev.target.closest('[data-sugg]');
  if (sugg) { $('chat-input').value = sugg.dataset.sugg; $('chat-form').requestSubmit(); return; }

  const re = ev.target.closest('[data-reanalyze]');
  if (re) {
    re.disabled = true; re.textContent = 'Analysing…';
    try {
      const out = await api(`/api/leads/${re.dataset.reanalyze}/reanalyze`, { method: 'POST' });
      applyPayload(out); render(); toast('Re-analysed', 'ok');
    } catch (e) { re.disabled = false; re.textContent = 'Re-analyse'; await handleError(e); }
    return;
  }

  const del = ev.target.closest('[data-delete]');
  if (del) {
    if (!confirm('Delete this lead?')) return;
    try {
      const out = await api(`/api/leads/${del.dataset.delete}`, { method: 'DELETE' });
      state.leads = state.leads.filter((l) => l.id !== del.dataset.delete);
      state.stats = out.stats || state.stats; state.selected = null;
      render(); toast('Lead deleted', 'ok');
    } catch (e) { await handleError(e); }
    return;
  }

  const call = ev.target.closest('[data-call]');
  if (call) {
    const lead = state.leads.find((l) => l.id === call.dataset.call);
    if (!state.health.voice_mock &&
        !confirm(`Place a REAL phone call to ${lead.name} at ${lead.phone}?`)) return;
    call.disabled = true; call.textContent = 'Dialling…';
    try {
      const out = await api(`/api/leads/${call.dataset.call}/call`, { method: 'POST' });
      applyPayload({ lead: out }); render(); toast('Call placed', 'ok');
      pollCalls();
    } catch (e) {
      call.disabled = false; call.textContent = 'AI confirmation call';
      await handleError(e);
    }
  }
});

on('detail-body', 'submit', async (ev) => {
  if (ev.target.id !== 'chat-form') return;
  ev.preventDefault();
  const input = $('chat-input');
  const q = input.value.trim();
  if (!q || state.busy) return;
  const lead = state.leads.find((l) => l.id === state.selected);
  if (!lead) return;

  state.busy = true;
  lead.chat = [...(lead.chat || []), { role: 'user', content: q },
               { role: 'assistant', content: '…', engine: '' }];
  input.value = ''; renderDetail();

  try {
    const out = await api(`/api/leads/${lead.id}/chat`,
      { method: 'POST', body: JSON.stringify({ question: q }) });
    lead.chat = out.chat;
  } catch (e) {
    lead.chat = lead.chat.slice(0, -1);
    await handleError(e);
  } finally { state.busy = false; renderDetail(); }
});

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && state.selected) { state.selected = null; render(); }
  if (ev.key === 'Enter' && !ev.shiftKey && ev.target.id === 'chat-input') {
    ev.preventDefault(); $('chat-form').requestSubmit();
  }
});

/* voice polling — only while a call is in flight */
let pollTimer = null;
async function pollCalls() {
  clearTimeout(pollTimer);
  const inFlight = state.leads.some((l) => l.voice?.call_id && !l.voice?.outcome);
  if (!inFlight) return;
  pollTimer = setTimeout(async () => {
    try {
      const out = await api('/api/calls/refresh', { method: 'POST' });
      applyPayload(out); render();
    } catch (e) { console.warn(e); }
    pollCalls();
  }, 3000);
}

/* ───────── boot ───────── */
(async function init() {
  try {
    await loadAll(); render(); pollCalls();
    if (!state.health.ai_configured) {
      toast('No AI key configured — set GROQ_API_KEY or GOOGLE_API_KEY', 'err');
    }
  } catch (e) { toast('Backend unreachable: ' + e.message, 'err'); }
})();
