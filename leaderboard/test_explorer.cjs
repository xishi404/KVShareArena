/* Run with: node leaderboard/test_explorer.cjs */
const assert = require('node:assert/strict');
require('./arena.js');
const data = require('./leaderboard.json');
const {recordsFor, scoreRecord} = globalThis.KVAExplorer;
const weights = {compute: 50, memory: 50};
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-12, `${a} != ${b}`);
const sample = {quality:.8, compute:.7, memory:.3, latency:null};
close(scoreRecord(sample,1/11,weights),1.1*.8*.5/(.1*.8+.5));
close(scoreRecord(sample,.5,{compute:1,memory:1,latency:1}),scoreRecord(sample,.5,weights));
assert.equal(scoreRecord({...sample,memory:null},.5,weights),null);
assert.equal(scoreRecord(sample,.5,{latency:1,throughput:1}),null);
for (const t of [0,1/11,.5,1]) {
  const expected=scoreRecord(sample,t,weights);
  for (const runtime of [null,-100,0,.99,100]) {
    close(scoreRecord({...sample,latency:runtime,throughput:runtime},t,
      {...weights,latency:999,throughput:999}),expected);
  }
}
assert.equal(scoreRecord(sample,.5,{compute:0,memory:0,latency:0}),null);
assert.equal(scoreRecord(sample,0,{compute:0,memory:0,latency:1}),.8);
assert.equal(scoreRecord(sample,1,weights),.5);
assert.equal(scoreRecord({...sample,quality:-.2},.5,weights),0);
assert.equal(scoreRecord({...sample,quality:2},0,weights),1);
assert.equal(scoreRecord({...sample,compute:-.2},1,{compute:1}),0);
for (const task of [...data.retrieved_evidence.subsets.map(s=>s.key),'notes']) {
  const rows=recordsFor(data,task);
  assert.equal(new Set(rows.map(r=>r.key)).size,rows.length);
  const qualityRows=task==='notes'?data.agent_reports.rows:data.retrieved_evidence.rows;
  for(const source of qualityRows) {
    const q=task==='notes'?source.pgr:source.pgr?.[task]?.value;
    if(typeof q!=='number')continue;
    const row=rows.find(r=>r.key===source.key);
    assert.ok(row,`${task}/${source.key} omitted`);
    assert.ok(Math.abs(row.quality-q)<.001,'Quality must match source rounding');
  }
  for(const point of data.frontiers.compute?.panels?.[task]||[])close(rows.find(r=>r.key===point.key).compute,1-point.x);
  for(const point of data.frontiers.memory?.panels?.[task]||[])assert.equal(rows.find(r=>r.key===point.key).memory,point.x);
  for(const point of data.frontiers.runtime?.panels?.[task]?.consumer||[])assert.equal(rows.find(r=>r.key===point.key).latency,point.x);
  for(const point of data.frontiers.runtime?.panels?.[task]?.e2e||[])assert.equal(rows.find(r=>r.key===point.key).latency_e2e,point.x);
  for(const row of rows) {
    assert.equal(scoreRecord(row,1/11,weights),
      scoreRecord({...row,latency:null,latency_e2e:999,throughput:999},1/11,weights));
  }
}
assert.equal(recordsFor(data,'notes').length,data.agent_reports.rows.length);
assert.ok(recordsFor(data,'qasper').find(r=>r.key==='minipic').memory<0);
assert.ok(recordsFor(data,'hotpotqa').some(r=>r.quality<0));
assert.ok(recordsFor(data,'frames').every(r=>r.compute===null&&r.memory===null&&r.latency===null));
console.log('PASS: formula, endpoints, missing axes, source coverage, negative values, and cost mappings.');
