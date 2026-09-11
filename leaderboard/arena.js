/* Static, dependency-free explorer. Source measurements are never rewritten. */
(function (root) {
  'use strict';
  const axes = ['compute', 'memory', 'latency'];
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const clip = x => Math.max(0, Math.min(1, x));
  const names = {qasper: 'Qasper', multifieldqa_en: 'MultiFieldQA-en', hotpotqa: 'HotpotQA', frames: 'FRAMES', notes: 'HotpotQA · agent reports'};
  const groups = {official: 'Official implementations', port: 'Our ports', reference: 'Reference rows'};
  const axisNames = {compute: 'Compute saved', memory: 'Memory saved', latency: 'TTFT saved · cache ready', latency_e2e: 'TTFT saved · including cache build'};
  const axisLaneNames = {compute: 'payload still recomputed at answer time (log scale, cheaper →)', latency: 'time to first token still spent, share of full prefill (log scale, faster →)'};
  function recordsFor(data, task) {
    const isAR = task === 'notes';
    const qualityRows = isAR ? data.agent_reports.rows : data.retrieved_evidence.rows;
    const map = new Map();
    function entry(key) {
      if (!map.has(key)) {
        const meta = data.methods[key] || {};
        map.set(key, {key, task, ...meta, display: meta.display || key, partition: meta.partition || 'port', quality: null, compute: null, memory: null, latency: null, latency_e2e: null, evidence: null, computeSource: null});
      }
      return map.get(key);
    }
    for (const row of qualityRows) {
      const q = isAR ? {value: row.pgr, sig: row.sig, direction: row.direction, delta: row.delta, ci: row.ci} : row.pgr?.[task];
      if (!q || !finite(q.value)) continue;
      const r = entry(row.key);
      r.quality = q.value; r.evidence = q;
      const share = isAR ? row.recompute_share : row.recompute_share?.[task];
      if (finite(share)) { r.compute = 1 - share; r.computeSource = 'quality table · recompute_share'; }
    }
    const frontiers = data.frontiers || {};
    const panels = {
      compute: frontiers.compute?.panels?.[task] || [],
      memory: frontiers.memory?.panels?.[task] || [],
      latency: frontiers.runtime?.panels?.[task]?.consumer || [],
      latency_e2e: frontiers.runtime?.panels?.[task]?.e2e || []
    };
    for (const [axis, rows] of Object.entries(panels)) for (const point of rows) {
      const r = entry(point.key);
      if (finite(point.x)) r[axis] = axis === 'compute' ? 1 - point.x : point.x;
      // Compute panels retain more digits of the same quality measurement.
      if (finite(point.y) && (axis === 'compute' || !finite(r.quality))) r.quality = point.y;
      if (axis === 'compute') r.computeSource = 'compute panel · 1 − recomputation share';
    }
    return [...map.values()];
  }
  function scoreRecord(row, t, weights) {
    if (t < 1 && !finite(row.quality)) return null;
    const a = finite(row.quality) ? clip(row.quality) : 0;
    if (t === 0) return a;
    const selected = axes.filter(axis => weights[axis] > 0);
    if (!selected.length || selected.some(axis => !finite(row[axis]))) return null;
    const total = selected.reduce((s, axis) => s + weights[axis], 0);
    const c = selected.reduce((s, axis) => s + clip(row[axis]) * weights[axis], 0) / total;
    if (t === 1) return c;
    const den = (1 - t) * c + t * a;
    return den === 0 ? 0 : a * c / den;
  }
  root.KVAExplorer = {recordsFor, scoreRecord};
  if (!root.document) return;
  const $ = id => document.getElementById(id);
  const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = (x, digits = 3) => finite(x) ? x.toFixed(digits) : '—';
  const pct = x => finite(x) ? `${(x * 100).toFixed(1)}%` : '—';
  let data, allRows = [], visibleRows = [], scoredRows = [], selectedKey = null;
  let t = 1 / 11;
  const weights = {compute: 50, memory: 50, latency: 0};
  const task = () => $('dataset').value;
  const formatCell = (v, percent = false) => `<span class="${!finite(v) ? 'missing' : v < 0 ? 'negative' : ''}">${percent ? pct(v) : num(v)}</span>`;
  function setDatasets() {
    $('dataset').innerHTML = ($('track').value === 'ar' ? [{key:'notes'}] : data.retrieved_evidence.subsets).map(s => `<option value="${esc(s.key)}">${esc(names[s.key] || s.label)}</option>`).join('');
    setTask();
  }
  function setTask() {
    allRows = recordsFor(data, task());
    $('task-caption').textContent = `${task() === 'notes' ? 'Cached agent reports' : 'Separately cached evidence'} · N = ${data._meta.n[task()]}`;
    for (const axis of axes) {
      const available = allRows.some(r => r.partition !== 'reference' && finite(r[axis]));
      $(`weight-${axis}`).disabled = !available;
      document.querySelector(`[data-axis="${axis}"]`).classList.toggle('unavailable', !available);
      $('preset').querySelector(`option[value="${axis}"]`).disabled = !available;
    }
    const availableCost = axes.some(a => !$(`weight-${a}`).disabled);
    $('cost-weight').disabled = !availableCost;
    $('preset').querySelector('option[value="balanced"]').disabled = !availableCost;
    const currentPreset = $('preset').value;
    applyPreset(currentPreset === 'custom' || $('preset').selectedOptions[0].disabled ? 'quality' : currentPreset);
    const availablePlot = [...$('plot-axis').options].find(o => allRows.some(r => finite(r[o.value])));
    for (const option of $('plot-axis').options) option.disabled = !allRows.some(r => finite(r[option.value]));
    if ($('plot-axis').selectedOptions[0]?.disabled && availablePlot) $('plot-axis').value = availablePlot.value;
    render();
  }
  function applyPreset(preset) {
    $('preset').value = preset;
    t = preset === 'quality' ? 1 / 11 : .5;
    for (const axis of axes) weights[axis] = 0;
    const available = axes.filter(a => !$(`weight-${a}`).disabled);
    if (axes.includes(preset)) weights[preset] = 100;
    else {
      const defaults = preset === 'quality' ? available.filter(a => a !== 'latency') : available;
      for (const axis of defaults.length ? defaults : available) weights[axis] = 50;
    }
    if (!available.length) t = 0;
    syncSliders(); render();
  }
  function syncSliders() {
    $('cost-weight').value = t * 100;
    $('quality-pct').textContent = `${Math.round((1 - t) * 100)}%`;
    $('cost-pct').textContent = `${Math.round(t * 100)}%`;
    const total = axes.reduce((s,a) => s + weights[a],0);
    for (const axis of axes) {
      $(`weight-${axis}`).value = weights[axis];
      $(`pct-${axis}`).textContent = $(`weight-${axis}`).disabled ? 'Not measured' : `${total ? Math.round(weights[axis] / total * 100) : 0}%`;
    }
    $('weights-note').textContent = $('cost-weight').disabled ? 'No cost records for this dataset. The score uses quality only.' : t === 0 ? 'Quality only. Cost sliders do not affect this score.' : total === 0 ? 'Select at least one cost. Scores are not available with all cost weights at zero.' : 'Cost weights are normalized to 100%. Missing selected costs leave a row unscored.';
  }
  function render() {
    if (!data) return;
    const query = $('search').value.trim().toLowerCase(), partition = $('partition').value;
    const sort = $('sort').value;
    scoredRows = allRows.map(r => ({...r, score: scoreRecord(r,t,weights)}));
    // Ranks refer to the full implementation group, independent of text filtering.
    for (const group of Object.keys(groups)) {
      const rows = scoredRows.filter(r => r.partition === group).sort((a,b) => {
        if (!finite(a[sort])) return finite(b[sort]) ? 1 : a.display.localeCompare(b.display);
        if (!finite(b[sort])) return -1;
        // Scores clip at 0, so failing rows tie; raw quality keeps their order visible.
        return b[sort] - a[sort] || (finite(b.quality) && finite(a.quality) ? b.quality - a.quality : 0) || a.display.localeCompare(b.display);
      });
      let last = null, rank = null;
      rows.forEach((r,i) => {
        const value = finite(r[sort]) ? r[sort].toFixed(3) : null;
        if (value !== last) rank = i + 1;
        r.rank = group === 'reference' || value === null ? null : rank;
        last = value;
      });
    }
    visibleRows = scoredRows.filter(r => (partition === 'all' || r.partition === partition) && `${r.display} ${r.key} ${data.classes[r.class] || ''}`.toLowerCase().includes(query));
    visibleRows.sort((a,b) => Object.keys(groups).indexOf(a.partition) - Object.keys(groups).indexOf(b.partition) || (finite(a[sort]) && finite(b[sort]) ? b[sort] - a[sort] : finite(a[sort]) ? -1 : finite(b[sort]) ? 1 : 0) || (finite(b.quality) && finite(a.quality) ? b.quality - a.quality : 0) || a.display.localeCompare(b.display));
    let html = '', previous = null;
    for (const r of visibleRows) {
      if (previous !== r.partition) html += `<tr class="group-row"><td colspan="8">${esc(groups[r.partition])} · ${visibleRows.filter(x => x.partition === r.partition).length}</td></tr>`;
      previous = r.partition;
      const arrow = r.evidence?.sig ? `<span class="sig" title="Paired task-score difference vs free position alignment">${r.evidence.direction === 'up' ? '↑' : '↓'}</span>` : '';
      html += `<tr class="method-row${r.key === selectedKey ? ' selected' : ''}" data-key="${esc(r.key)}"><td>${r.rank === null ? '<span class="missing">—</span>' : `<span class="${r.rank === 1 ? 'rank-first' : ''}">${r.rank}</span>`}</td><td><button class="method-link" data-method="${esc(r.key)}">${esc(r.display)}</button><span class="method-class">${esc(data.classes[r.class] || 'Full-prefill reference')}${r.self_reported ? ' · self-reported' : ''}</span></td><td>${finite(r.score) ? `<span class="score-pill">${num(r.score)}</span>` : '<span class="missing" title="Missing a selected measurement">—</span>'}</td><td>${formatCell(r.quality)}${arrow}</td><td>${formatCell(r.compute,true)}</td><td>${formatCell(r.memory,true)}</td><td>${formatCell(r.latency,true)}</td><td><button class="details-button" data-method="${esc(r.key)}" aria-label="Details for ${esc(r.display)}">↗</button></td></tr>`;
    }
    $('result-rows').innerHTML = html || '<tr><td colspan="8">No methods match these filters. Try another name or implementation group.</td></tr>';
    $('task-label').textContent = `${$('track').selectedOptions[0].textContent} / ${names[task()]}`;
    const methods = visibleRows.filter(r => r.partition !== 'reference');
    $('result-count').textContent = `${visibleRows.length} rows shown · ${methods.filter(r => finite(r.score)).length} of ${methods.length} methods have a preference score`;
    const measured = axes.map(a => `${methods.filter(r => finite(r[a])).length}/${methods.length} ${a === 'latency' ? 'TTFT' : a}`).join(' · ');
    $('availability-note').textContent = `Cost coverage: ${measured}. A dash means no measurement, not zero cost. TTFT here assumes the source cache is ready.`;
    // Ranks are point estimates. Where no row is significantly above position
    // alignment, the order is not a result and the page has to say so.
    const winners = methods.filter(r => r.evidence?.sig && r.evidence.direction === 'up');
    const tested = methods.filter(r => r.evidence && r.evidence.sig !== null && r.evidence.sig !== undefined);
    $('significance-note').textContent = !tested.length
      ? 'This task ships no paired comparison against position alignment, so the order below is by point estimate only.'
      : !winners.length
        ? 'No method is significantly above position alignment on this task. The order below is by point estimate; the differences are ties.'
        : `${winners.length} of ${methods.length} methods are significantly above position alignment on this task (arrows in the quality column). Rows without an arrow are ties.`;
    $('significance-note').className = winners.length ? 'significance-note' : 'significance-note flat';
    renderPlot();
  }
  function renderPlot() {
    const axis = $('plot-axis').value;
    const points = visibleRows.filter(r => finite(r[axis]) && finite(r.quality));
    let svg = `<title id="plot-title">Quality versus ${esc(axisNames[axis])}</title><desc id="plot-desc">Each point is one method configuration. Higher and further right means more quality and more savings. Select a point to inspect its measurements.</desc>`;
    if (!points.length) {
      $('plot').innerHTML = svg + '<text x="390" y="150" text-anchor="middle">No measurements for this selection.</text>';
      $('plot-note').textContent = 'Quality results remain available in the table above.'; return;
    }
    const ymin = Math.min(0,...points.map(r => r.quality)), ymax = Math.max(1,...points.map(r => r.quality));
    const y = v => 263 - (v - ymin) / (ymax - ymin) * 226;
    // Most rows recompute nothing, so a linear "saved" axis stacks them on the
    // right edge. Plot what is still spent (1 - saved) on a log scale, cheaper to
    // the right, and give the rows that spend nothing their own lane.
    const LANE = axis === 'compute' || axis === 'latency';
    let hintX = 737;
    const DEC = 3;                                   // decades shown: 100% .. 0.1%
    const laneL = 596, laneR = 730, plotR = 560;
    let x, ticks;
    if (LANE) {
      const spent = v => Math.max(1 - v, 0);
      const pos = v => spent(v) <= 0 ? (laneL + laneR) / 2
        : 66 + Math.min(Math.max(-Math.log10(Math.max(spent(v), 1e-3)), 0), DEC) / DEC * (plotR - 66);
      x = pos;
      ticks = [1, .1, .01, .001].map((c,i) => ({px: 66 + i / DEC * (plotR - 66), label: c >= .01 ? `${c*100}%` : `${(c*100).toFixed(1)}%`}));
      svg += `<rect x="${laneL-14}" y="30" width="${laneR-laneL+28}" height="240" fill="#f3f6fb"/>`;
      svg += `<text x="${(laneL+laneR)/2}" y="26" text-anchor="middle" class="lane">spends nothing</text>`;
      hintX = plotR - 6;
      svg += `<text x="${(plotR+laneL)/2-8}" y="${y(ymin)+8}" text-anchor="middle" class="lane">//</text>`;
    } else {
      const lo = Math.min(0,...points.map(r => r[axis])), hi = Math.max(...points.map(r => r[axis]), 0);
      const pad = (hi - lo) * .08 || .05;
      const xmin = lo - pad, xmax = hi + pad;
      x = v => 66 + (v - xmin) / (xmax - xmin) * 664;
      ticks = [0,1,2,3,4].map(i => {const v = xmin+(xmax-xmin)*i/4; return {px: x(v), label: `${(v*100).toFixed(0)}%`};});
    }
    for (let i=0;i<=4;i++) {
      const yv = ymin+(ymax-ymin)*i/4;
      svg += `<path d="M60 ${y(yv)}H740" stroke="#edf1f7" fill="none"/><text x="53" y="${y(yv)+4}" text-anchor="end">${yv.toFixed(2)}</text>`;
    }
    for (const tk of ticks) svg += `<path d="M${tk.px} 30V270" stroke="#edf1f7" fill="none"/><text x="${tk.px}" y="287" text-anchor="middle">${tk.label}</text>`;
    svg += `<text x="398" y="313" text-anchor="middle">${esc(LANE ? axisLaneNames[axis] : axisNames[axis])}</text><text x="66" y="17">Quality (PGR) ↑</text><text x="${hintX}" y="17" text-anchor="end">Higher & further right is better</text>`;
    // References first so measured implementations are not hidden under them.
    for (const r of [...points].sort((a,b) => (a.partition === 'reference' ? -1 : 0) - (b.partition === 'reference' ? -1 : 0))) {
      const color = r.partition === 'official' ? '#3673d9' : r.partition === 'port' ? '#9577d2' : '#a5afbf';
      const label = `${r.display}: PGR ${num(r.quality)}; ${axisNames[axis]} ${pct(r[axis])}`;
      svg += `<g class="point" data-method="${esc(r.key)}" tabindex="0" role="button" aria-label="${esc(label)}"><title>${esc(label)}</title><circle cx="${x(r[axis])}" cy="${y(r.quality)}" r="${r.key === selectedKey ? 8 : 5.5}" fill="${color}" fill-opacity=".85" stroke="white" stroke-width="1.5"/></g>`;
    }
    $('plot').innerHTML = svg;
    $('plot-note').textContent = `${points.length} measured points.${LANE ? ' The horizontal axis shows what a method still spends, on a log scale, so rows that spend nothing sit in the shaded lane instead of on top of each other.' : ''} Raw values are shown, including negative savings and PGR. ${axis === 'latency_e2e' ? 'This view includes source-cache construction; it does not change the preference score.' : axis === 'latency' ? 'The source cache is ready before the request. Cache-build-inclusive latency is available in the selector.' : 'Select a point or a method name to inspect its measurements.'}`;
  }
  function showMethod(key) {
    const r = scoredRows.find(row => row.key === key);
    if (!r) return;
    selectedKey = key; render();
    $('method-title').textContent = r.display;
    const missing = axes.filter(a => weights[a] > 0 && !finite(r[a]));
    const rows = [['Quality (PGR)',num(r.quality)],['Preference score',num(r.score)],...axes.map(a => [axisNames[a],pct(r[a])]),['TTFT saved · include cache build',pct(r.latency_e2e)]];
    $('method-content').innerHTML = `<p>${esc(groups[r.partition])} · ${esc(names[task()])} · N = ${data._meta.n[task()]}</p><dl>${rows.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${!finite(r.score) ? `<p>No preference score: ${missing.length ? `missing ${esc(missing.join(', '))}` : 'no selected cost or missing quality'}.</p>` : ''}<p><strong>Paired task-score comparison:</strong> ${r.evidence?.ci ? `difference ${num(r.evidence.delta,4)}; 95% interval [${r.evidence.ci.map(v => num(v,4)).join(', ')}] versus free position alignment. These are task-score differences, not PGR differences.` : 'No paired interval in this export.'}</p><p><strong>Compute source:</strong> ${esc(r.computeSource || 'Not measured')}.</p><p><strong>Reported by:</strong> ${esc(r.submitter || 'KVShareArena authors')}${r.self_reported ? ' (self-reported)' : ''}.</p><p><strong>Source:</strong> ${esc(r.provenance || 'Frozen full-prefill reference')}<br>Export: ${esc(data._meta.generated)} · Method key: ${esc(r.key)}</p>`;
    $('method-dialog').showModal();
  }
  function exportCSV() {
    const header = ['task','method_key','method','implementation_group','rank_by_'+$('sort').value,'quality_pgr','compute_saved','memory_saved','ttft_saved_cache_ready','ttft_saved_include_cache_build','preference_score','cost_weight','compute_weight','memory_weight','latency_weight','data_date'];
    const total = axes.reduce((s,a) => s+weights[a],0);
    const lines = [header,...visibleRows.map(r => [task(),r.key,r.display,r.partition,r.rank,r.quality,r.compute,r.memory,r.latency,r.latency_e2e,r.score,t,...axes.map(a => total ? weights[a]/total : 0),data._meta.generated])];
    const csv = lines.map(row => row.map(value => {
      let s = value == null ? '' : String(value);
      if (typeof value === 'string' && /^[=+@\-\t\r]/.test(s)) s = "'"+s;
      return '"'+s.replace(/"/g,'""')+'"';
    }).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a = document.createElement('a'); a.href=url; a.download=`kvsharearena-${task()}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  function init(loaded) {
    data = loaded;
    const tasks = [...data.retrieved_evidence.subsets.map(s => s.key),'notes'];
    const records = tasks.flatMap(k => recordsFor(data,k));
    $('stat-methods').textContent = new Set(records.filter(r => r.partition !== 'reference').map(r => r.key)).size;
    $('stat-datasets').textContent = data.retrieved_evidence.subsets.length;
    $('coverage-table').innerHTML = '<thead><tr><th>Workload / dataset</th><th>N</th><th>Quality</th><th>Compute</th><th>Memory</th><th>TTFT</th></tr></thead><tbody>'+tasks.map(k => {
      const rows = recordsFor(data,k).filter(r => r.partition !== 'reference');
      return `<tr><td>${esc(names[k])}<span class="method-class">${k === 'notes' ? 'Agent Reports' : 'Retrieved Evidence'}</span></td><td>${data._meta.n[k]}</td>${['quality',...axes].map(a => `<td>${rows.filter(r => finite(r[a])).length || '—'}</td>`).join('')}</tr>`;
    }).join('')+'</tbody>';
    $('provenance').textContent = `Data export: ${data._meta.generated} · ${data._meta.schema_version} · Source measurements are unchanged. Interactive ranks use complete selected measurements.`;
    $('track').addEventListener('change',setDatasets); $('dataset').addEventListener('change',setTask);
    $('preset').addEventListener('change',() => {if ($('preset').value !== 'custom') applyPreset($('preset').value);});
    $('cost-weight').addEventListener('input',() => {t=Number($('cost-weight').value)/100; $('preset').value='custom'; syncSliders(); render();});
    for (const axis of axes) $(`weight-${axis}`).addEventListener('input',() => {weights[axis]=Number($(`weight-${axis}`).value); $('preset').value='custom'; syncSliders(); render();});
    for (const id of ['search','partition','sort','plot-axis']) $(id).addEventListener(id === 'search' ? 'input' : 'change',render);
    $('reset').addEventListener('click',() => {$('track').value='re';$('preset').value='quality';$('search').value='';$('partition').value='all';$('sort').value='score';$('plot-axis').value='compute';setDatasets();});
    $('export').addEventListener('click',exportCSV);
    document.addEventListener('click',event => {const target=event.target.closest('[data-method]');if(target) showMethod(target.dataset.method);});
    $('plot').addEventListener('keydown',event => {if ((event.key === 'Enter' || event.key === ' ') && event.target.dataset.method) {event.preventDefault();showMethod(event.target.dataset.method);}});
    $('close-dialog').addEventListener('click',() => $('method-dialog').close());
    $('method-dialog').addEventListener('click',event => {if(event.target === $('method-dialog')) {const r=event.target.getBoundingClientRect();if(event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom) event.target.close();}});
    $('preset').value='quality';setDatasets();
  }
  try {
    if (root.KVA_LEADERBOARD) init(root.KVA_LEADERBOARD);
    else fetch('leaderboard.json').then(r => {if(!r.ok) throw new Error('Data fetch failed'); return r.json();}).then(init).catch(() => {$('load-error').hidden=false;});
  } catch (error) { $('load-error').hidden=false; console.error(error); }
})(globalThis);
