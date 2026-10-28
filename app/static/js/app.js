/* OSINT Monitoring — frontend.
   Vanilla. No build step, no framework. Templates give the shell; data comes
   from /api/*. See CLAUDE.md §5 before adding a dependency. */

const OSINT = (() => {

  // ---- fetch helpers ----------------------------------------------------
  async function api(path, opts = {}) {
    const res = await fetch(`/api${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).error || detail; } catch { /* non-JSON body */ }
      throw new Error(detail);
    }
    return res.json();
  }

  const post = (path, body) =>
    api(path, { method: 'POST', body: JSON.stringify(body) });

  // ---- formatting -------------------------------------------------------
  // Storage is UTC; every display goes through here. Never render a raw stamp.
  const EAT = 'Africa/Nairobi';

  function eat(utcIso, opts = {}) {
    if (!utcIso) return '—';
    const d = new Date(String(utcIso).endsWith('Z') ? utcIso : utcIso + 'Z');
    if (isNaN(d)) return String(utcIso);
    return d.toLocaleString('en-GB', {
      timeZone: EAT, day: '2-digit', month: 'short',
      hour: '2-digit', minute: '2-digit', ...opts,
    });
  }

  function eatTime(utcIso) {
    if (!utcIso) return '—';
    const d = new Date(String(utcIso).endsWith('Z') ? utcIso : utcIso + 'Z');
    return isNaN(d) ? '—' : d.toLocaleTimeString('en-GB',
      { timeZone: EAT, hour: '2-digit', minute: '2-digit' });
  }

  function ago(utcIso) {
    if (!utcIso) return 'never';
    const d = new Date(String(utcIso).endsWith('Z') ? utcIso : utcIso + 'Z');
    const secs = (Date.now() - d.getTime()) / 1000;
    if (isNaN(secs)) return '—';
    if (secs < 60) return 'just now';
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  }

  function clock(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  function duration(seconds) {
    const s = Math.floor(seconds || 0);
    if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
    if (s >= 60) return `${Math.floor(s / 60)}m`;
    return `${s}s`;
  }

  const pct = (x, dp = 1) => `${((x || 0) * 100).toFixed(dp)}%`;

  // Never inject API text as HTML — transcripts are model output, and search
  // snippets are the one deliberate exception (they carry <mark> from SQLite).
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  // ---- shared UI bits ---------------------------------------------------
  function werClass(wer) {
    if (wer == null) return '';
    if (wer < 0.25) return 'wer-good';
    if (wer < 0.45) return 'wer-mid';
    return 'wer-bad';
  }

  function langChip(code, packs) {
    const p = (packs || []).find(x => x.code === code);
    if (!p) return `<span class="lang-chip">${esc(code)}</span>`;
    return `<span class="lang-chip">
      <span class="swatch" style="background:${esc(p.colour)}"></span>
      ${esc(p.name)}<span class="autonym">${esc(p.autonym)}</span></span>`;
  }

  // Capture health badge + the resolved endpoint, so a broken feed is obvious
  // from the card rather than only from a log file.
  const HEALTH = {
    ok:       ['pill-ok',   'Capturing'],
    degraded: ['pill-warn', 'Intermittent'],
    failing:  ['pill-live', 'Not capturing'],
    untested: ['pill-idle', 'Untested'],
  };

  function healthRow(c) {
    const h = c.health || {};
    const [cls, label] = HEALTH[h.state] || HEALTH.untested;
    const why = c.known_issue || h.last_gap_reason || '';
    return `<div class="health-row">
        <span class="pill ${cls}"><span class="dot${h.state === 'ok' ? ' pulse' : ''}"></span>${label}</span>
        ${h.gaps ? `<span class="dim">${h.gaps} gap${h.gaps === 1 ? '' : 's'}</span>` : ''}
        <span class="why" title="${esc(why)}">${esc(why)}</span>
      </div>
      <div class="endpoint" title="${esc(c.endpoint || '')}">
        ${c.endpoint_verified_utc ? '<b>verified</b> ' : ''}${esc(c.endpoint || 'no endpoint resolved')}
      </div>`;
  }

  function empty(icon, title, body, cmd) {
    return `<div class="empty">
      <svg class="ico"><use href="#${icon}"/></svg>
      <h3>${esc(title)}</h3>
      <p>${body}</p>
      ${cmd ? `<code>${esc(cmd)}</code>` : ''}
    </div>`;
  }

  // ---- topbar clock -----------------------------------------------------
  function startClock() {
    const el = document.getElementById('clock');
    if (!el) return;
    const tick = () => {
      const t = new Date().toLocaleTimeString('en-GB',
        { timeZone: EAT, hour12: false });
      el.innerHTML = `${t}<small>EAT</small>`;
    };
    tick();
    setInterval(tick, 1000);
  }

  // ---- capture status pill ---------------------------------------------
  async function refreshCapturePill() {
    const pill = document.getElementById('capture-pill');
    if (!pill) return;
    try {
      const s = await api('/stats');
      if (s.segments_today > 0) {
        pill.className = 'pill pill-live';
        pill.innerHTML = `<span class="dot pulse"></span> ${s.hours_today}h captured today`;
      } else if (s.segments_total > 0) {
        pill.className = 'pill pill-warn';
        pill.innerHTML = `<span class="dot"></span> No capture today`;
      } else {
        pill.className = 'pill pill-idle';
        pill.innerHTML = `<span class="dot"></span> Capture idle`;
      }
    } catch {
      pill.className = 'pill pill-idle';
      pill.innerHTML = `<span class="dot"></span> Offline`;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    startClock();
    refreshCapturePill();
    setInterval(refreshCapturePill, 30000);
  });

  return { api, post, eat, eatTime, ago, clock, duration, pct, esc,
           werClass, langChip, empty, healthRow };
})();
