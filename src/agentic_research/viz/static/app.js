'use strict';
/* Agent trace viewer. One pure derive(events) -> view; live, replay and scrubbing share it. */

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const store = { get(k){try{return localStorage.getItem(k)}catch{return null}}, set(k,v){try{localStorage.setItem(k,v)}catch{}} };
const COLORS = { supervisor:'var(--sup)', researcher:'var(--res)', analyst:'var(--ana)', writer:'var(--wri)', human_approval:'var(--hum)' };

const POS = { __start__:[56,200], supervisor:[240,200], researcher:[500,60], analyst:[500,200], writer:[500,340], human_approval:[760,340], __end__:[900,110] };
const SIZE = { __start__:[70,36], __end__:[70,36], supervisor:[128,54], researcher:[128,54], analyst:[128,54], writer:[128,54], human_approval:[128,54] };

const S = {
  scenarios: [], topo: null, runs: [], selected: null,
  group: null,            // {id, status, runIds[], interrupt}
  events: {},             // runId -> [events]
  active: null, cursor: 0, follow: true, playing: false, timer: null, es: null,
  tab: 'thoughts', insp: 'report', filter: new Set(), animatedSeq: 0, expanded: new Set(),
};

/* ------------------------------------------------------------------ derive */
function derive(evts) {
  const v = { nodes:{}, edges:{}, lastRoute:null, cards:[], rows:[], snapshot:null, status:'idle',
              report:null, error:null, tools:[], memory:[], faults:[], pausedPayload:null, next:null, why:null, query:'', meta:{} };
  const n = id => v.nodes[id] ||= { state:'idle', visits:0, attempt:0 };
  let open = {}; // node -> card
  for (const e of evts) {
    const d = e.data || {}, node = e.node;
    switch (e.type) {
      case 'run.started': v.status='running'; v.query=d.query; v.meta=d.meta||{}; n('__start__').state='done'; break;
      case 'node.started': { const x=n(node); x.state = d.attempt>1?'retrying':'active'; x.attempt=d.attempt; x.visits += d.attempt>1?0:1;
        v.next=null; const c={node,attempt:d.attempt,t:e.t_ms,items:[],status:'running',seq:e.seq}; open[node]=c; v.cards.push({kind:'card',c}); break; }
      case 'node.finished': { n(node).state='done'; if(open[node]) open[node].status='done', open[node].ms=d.duration_ms; break; }
      case 'node.error': { n(node).state = d.will_retry?'retrying':'error'; const c=open[node]; if(c){c.status='error'; c.items.push({k:'error',d});}
        v.why={tag:'fault',text:`${node} failed (attempt ${d.attempt}): ${d.message}. ${d.will_retry?'Retrying.':'Retry budget exhausted.'}`}; break; }
      case 'fault.injected': { const c=open[node]; (c?c.items:null)?.push({k:'fault',d}); if(!c) v.cards.push({kind:'note',tag:'fault',text:d.message}); v.faults.push(e); break; }
      case 'recovery.succeeded': v.cards.push({kind:'note',tag:'recovery',text:d.message}); v.why={tag:'recovery',text:d.message}; break;
      case 'llm.response': { const c=open[node]; if(c) c.items.push({k:'say',d}); break; }
      case 'tool.call': { const c=open[node]; if(c) c.items.push({k:'tool',call:d,res:null}); break; }
      case 'tool.result': { const c=open[node]; const it=c&&[...c.items].reverse().find(i=>i.k==='tool'&&!i.res&&i.call.tool===d.tool); if(it) it.res=d;
        v.tools.push({node,tool:d.tool,ok:d.success,out:d.success?d.output:d.error,ms:d.duration_ms,args:it?it.call.arguments:null}); break; }
      case 'route.decision': {
        const k = `${d.source}>${d.target}`; v.edges[k]=(v.edges[k]||0)+1; v.lastRoute={...d,seq:e.seq};
        if (d.source in v.nodes && v.nodes[d.source].state==='active') v.nodes[d.source].state='done';
        v.next = d.target; v.why={tag:d.kind,text:d.reason,route:`${d.source} → ${d.target}`};
        v.cards.push({kind:'note',tag:d.kind,text:`${d.source} → ${d.target}: ${d.reason}`}); break; }
      case 'state.snapshot': v.snapshot=d; v.snapNode=node; break;
      case 'memory.checkpoint': v.memory.push({op:'checkpoint',node,detail:`Checkpoint saved after ${node} (step ${d.step})`}); break;
      case 'memory.write': case 'memory.read': v.memory.push({op:e.type.split('.')[1],node,detail:d.summary||e.type});
        v.cards.push({kind:'note',tag:'memory',text:d.summary||e.type}); break;
      case 'human.interrupt': n('human_approval').state='paused'; n('human_approval').visits+=1; v.pausedPayload=d.payload; v.next=null;
        v.why={tag:'gate',text:'Paused at the human gate. State is checkpointed; waiting for a decision.'}; break;
      case 'run.paused': v.status='paused'; break;
      case 'run.resumed': v.status='running'; v.pausedPayload=null; n('human_approval').state='active'; break;
      case 'human.decision': v.cards.push({kind:'note',tag:'human',text:`Human decision: ${d.decision}${d.feedback?` — “${d.feedback}”`:''}`}); break;
      case 'run.completed': v.status='completed'; v.report=d.final_report; n('__end__').state='done'; v.next=null; break;
      case 'run.failed': v.status='failed'; v.error=d.error; v.report=d.final_report||v.report; n('__end__').state='error'; v.next=null;
        Object.values(v.nodes).forEach(x=>{if(x.state==='retrying'||x.state==='active')x.state='error'}); v.why={tag:'fault',text:d.error}; break;
    }
    v.rows.push(e);
  }
  if (v.status==='paused' || v.status==='running' || v.status==='idle') { /* keep */ }
  if (!v.report && v.snapshot?.state?.final_report) v.report = v.snapshot.state.final_report;
  v.draft = v.snapshot?.state?.final_report || null;
  return v;
}

/* ------------------------------------------------------------- tiny markdown */
function md(src) {
  if (!src) return '';
  const inline = t => esc(t).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/(^|\W)_(.+?)_(?=\W|$)/g,'$1<em>$2</em>').replace(/`(.+?)`/g,'<code>$1</code>');
  const lines = src.split('\n'); let out='', i=0;
  while (i<lines.length) {
    const l = lines[i];
    if (/^\s*\|/.test(l) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i+1]||'')) {
      const cells = r => r.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
      out+='<table><tr>'+cells(l).map(c=>`<th>${inline(c)}</th>`).join('')+'</tr>'; i+=2;
      while (i<lines.length && /^\s*\|/.test(lines[i])) { out+='<tr>'+cells(lines[i]).map(c=>`<td>${inline(c)}</td>`).join('')+'</tr>'; i++; }
      out+='</table>'; continue;
    }
    let m;
    if ((m=l.match(/^(#{1,3}) (.*)/))) { out+=`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; i++; continue; }
    if (/^\s*- /.test(l)) { out+='<ul>'; while(i<lines.length&&/^\s*- /.test(lines[i])){out+=`<li>${inline(lines[i].replace(/^\s*- /,''))}</li>`;i++} out+='</ul>'; continue; }
    if (l.trim()==='') { i++; continue; }
    out+=`<p>${inline(l)}</p>`; i++;
  }
  return out;
}

/* --------------------------------------------------------------------- graph */
const SVGNS='http://www.w3.org/2000/svg';
const mk = (tag, attrs={}, parent) => { const el=document.createElementNS(SVGNS,tag); for(const k in attrs) el.setAttribute(k,attrs[k]); parent&&parent.appendChild(el); return el; };
function clip(from, to, size) { // point where the ray from center `from` toward `to` leaves the node rect
  const [cx,cy]=from, [tx,ty]=to, [w,h]=size, dx=tx-cx, dy=ty-cy;
  const s = Math.min((w/2+4)/Math.abs(dx||1e-9), (h/2+4)/Math.abs(dy||1e-9));
  return [cx+dx*s, cy+dy*s];
}
const EDGE_BEND = { dispatch:16, return:16, gate:0, entry:0, end:0, revise:46 };
const edgePaths = {};
function buildGraph() {
  const svg=$('#graph'); svg.innerHTML='';
  const defs=mk('defs',{},svg);
  for (const cls of ['','seen','hot']) { const m=mk('marker',{id:'ah'+cls,viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},defs); mk('path',{d:'M0 1 L9 5 L0 9z',class:'arrow '+cls},m); }
  const gE=mk('g',{},svg), gN=mk('g',{},svg);
  for (const e of S.topo.edges) {
    const a=POS[e.source], b=POS[e.target]; let d;
    if (e.source==='supervisor' && e.target==='__end__') d='M240 172 C 240 -12, 900 -30, 900 90';
    else {
      const p0=clip(a,b,SIZE[e.source]), p1=clip(b,a,SIZE[e.target]);
      const dx=p1[0]-p0[0], dy=p1[1]-p0[1], len=Math.hypot(dx,dy)||1, bend=EDGE_BEND[e.kind]||0;
      const cx=(p0[0]+p1[0])/2 - dy/len*bend, cy=(p0[1]+p1[1])/2 + dx/len*bend;
      d = e.kind==='revise' ? (()=>{const q0=[a[0]-30,a[1]+28],q1=[b[0]+30,b[1]+28]; return `M${q0[0]} ${q0[1]} Q ${(q0[0]+q1[0])/2} ${a[1]+95} ${q1[0]} ${q1[1]}`})()
          : `M${p0[0].toFixed(1)} ${p0[1].toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`;
    }
    const path=mk('path',{d,class:`edge ${e.kind}`,'marker-end':'url(#ah)',id:`e-${e.source}>${e.target}`},gE);
    const t=mk('title',{},path); t.textContent=`${e.source} → ${e.target} (${e.kind})`;
    edgePaths[`${e.source}>${e.target}`]=path;
  }
  for (const n of S.topo.nodes) {
    const [x,y]=POS[n.id], [w,h]=SIZE[n.id];
    const g=mk('g',{class:'node idle',id:'n-'+n.id,transform:`translate(${x},${y})`},gN);
    const col=COLORS[n.id]||'var(--term)';
    mk('rect',{class:'ring',x:-w/2,y:-h/2,width:w,height:h,rx:13,stroke:col},g);
    mk('rect',{class:'body',x:-w/2,y:-h/2,width:w,height:h,rx:12},g);
    if (n.kind!=='terminal') mk('rect',{class:'accent',x:-w/2+1,y:-h/2+10,width:4,height:h-20,rx:2,fill:col},g);
    const t=mk('text',{y:n.kind==='terminal'?5:-2},g); t.textContent=n.label;
    if (n.kind!=='terminal') { const s=mk('text',{class:'sub',y:15},g); s.textContent='idle'; }
    const b=mk('g',{class:'badge',transform:`translate(${w/2-6},${-h/2+4})`,visibility:'hidden'},g); mk('circle',{r:9},b); mk('text',{y:3.5,'text-anchor':'middle'},b);
    const ti=mk('title',{},g); ti.textContent=n.blurb;
  }
  mk('circle',{id:'token',r:6,visibility:'hidden'},svg);
}
const SUB = { idle:'idle', active:'working', retrying:'retrying', done:'done', error:'failed', paused:'awaiting you' };
function paintGraph(v) {
  for (const n of S.topo.nodes) {
    const g=$('#n-'+CSS.escape(n.id)); if(!g) continue; const st=v.nodes[n.id]||{state:'idle',visits:0};
    g.setAttribute('class',`node ${st.state}${v.next===n.id?' next':''}`);
    const sub=$('.sub',g); if(sub) sub.textContent = st.state==='retrying'?`retry · attempt ${st.attempt}`:SUB[st.state];
    const b=$('.badge',g); if(b){ const show=st.visits>1; b.setAttribute('visibility',show?'visible':'hidden'); $('text',b).textContent=st.visits; }
  }
  const hot=v.lastRoute?`${v.lastRoute.source}>${v.lastRoute.target}`:null;
  for (const k in edgePaths) {
    const p=edgePaths[k], kind=p.getAttribute('class').split(' ')[1];
    const cls=(v.edges[k]?(k===hot?'hot':'seen'):'');
    p.setAttribute('class',`edge ${kind} ${cls}`.trim()); p.setAttribute('marker-end',`url(#ah${cls})`);
  }
  const w=$('#why');
  if (v.why) w.innerHTML=`<span class="tag ${esc(v.why.tag)}">${esc(({dispatch:'Routed',return:'Returned',gate:'Gate',revise:'Revision',end:'Finished',fault:'Fault',recovery:'Recovered'})[v.why.tag]||v.why.tag)}</span><span>${v.why.route?`<b>${esc(v.why.route)}</b> — `:''}${esc(v.why.text)}</span>`;
  else w.innerHTML=`<span class="muted">Pick an experiment and press Run. The graph shows who has the baton and why.</span>`;
}
function animateToken(route) {
  const p=edgePaths[`${route.source}>${route.target}`], t=$('#token'); if(!p||!t||matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const len=p.getTotalLength(), dur=Math.max(250,520/ +$('#speed').value), t0=performance.now(); t.setAttribute('visibility','visible');
  const step=now=>{ const k=Math.min(1,(now-t0)/dur), pt=p.getPointAtLength(len*k); t.setAttribute('cx',pt.x); t.setAttribute('cy',pt.y);
    if(k<1) requestAnimationFrame(step); else t.setAttribute('visibility','hidden'); };
  requestAnimationFrame(step);
}

/* ------------------------------------------------------------------- stream */
const FILTERS = { Routing:['route.decision'], LLM:['llm.request','llm.response'], Tools:['tool.call','tool.result'],
  Memory:['memory.checkpoint','memory.write','memory.read'], Errors:['node.error','fault.injected','recovery.succeeded','run.failed'],
  Human:['human.interrupt','human.decision','run.paused','run.resumed'] };
function summ(e){ const d=e.data||{}; switch(e.type){
  case 'route.decision': return `${d.source} → ${d.target}: ${d.reason}`;
  case 'llm.request': return `${e.node} → LLM (round ${d.round}, ${d.message_count} messages)`;
  case 'llm.response': return d.tool_calls?.length?`${e.node} requested ${d.tool_calls.map(c=>c.tool).join(', ')}`:`${e.node} answered`;
  case 'tool.call': return `${e.node} → ${d.tool}`; case 'tool.result': return `${d.tool} ${d.success?'ok':'failed: '+d.error}`;
  case 'node.started': return `${e.node} started (attempt ${d.attempt})`; case 'node.finished': return `${e.node} finished in ${d.duration_ms} ms`;
  case 'node.error': return `${e.node} failed (attempt ${d.attempt}): ${d.message}`; case 'fault.injected': return `Injected ${d.kind} fault on ${d.target}: ${d.message}`;
  case 'state.snapshot': return `State after ${e.node}: ${d.task_status}`; case 'run.started': return `Run started: ${d.query}`;
  case 'human.decision': return `Decision: ${d.decision}`; default: return d.summary||d.message||e.type; } }
function renderStream(v) {
  const el=$('#stream'), keep=el.scrollTop, atEnd=el.scrollHeight-el.scrollTop-el.clientHeight<40;
  $('#filters').hidden = S.tab!=='log';
  if (S.tab==='log') {
    const allowed=S.filter.size?new Set([...S.filter].flatMap(f=>FILTERS[f])):null;
    el.innerHTML = v.rows.filter(e=>!allowed||allowed.has(e.type)).map(e=>`<div><div class="logrow" data-seq="${e.seq}"><time>${(e.t_ms/1000).toFixed(2)}s</time><span class="t">${esc(e.type)}</span><span>${esc(summ(e))}</span></div>${S.expanded.has(e.seq)?`<div class="logdetail"><pre>${esc(JSON.stringify(e.data,null,2).slice(0,6000))}</pre></div>`:''}</div>`).join('') || '<div class="empty">No events yet.</div>';
  } else {
    el.innerHTML = v.cards.map(it=>{
      if (it.kind==='note') return `<div class="note"><span class="chip">${esc(it.tag)}</span><span>${esc(it.text)}</span></div>`;
      const c=it.c, col=COLORS[c.node]||'var(--term)';
      const st=c.status==='running'?'<span class="chip run">working</span>':c.status==='error'?'<span class="chip bad">failed</span>':'<span class="chip ok">done</span>';
      const body=c.items.map(i=>{
        if(i.k==='say') return i.d.content?`<p class="say">${esc(i.d.content)}</p>`:'';
        if(i.k==='tool'){ const r=i.res; return `<div class="tool"><div><b>${esc(i.call.tool)}</b> <code>${esc(JSON.stringify(i.call.arguments))}</code></div><div>${r?(r.success?`<code>${esc(String(typeof r.output==='string'?r.output:JSON.stringify(r.output)).slice(0,420))}</code>`:`<span class="fail">${esc(r.error)}</span>`):'waiting for result…'}</div></div>`; }
        if(i.k==='fault') return `<p class="say"><span class="chip bad">injected fault</span> ${esc(i.d.message)}</p>`;
        if(i.k==='error') return `<p class="say"><span class="chip bad">${esc(i.d.error_type)}</span> ${esc(i.d.message)} — ${i.d.will_retry?'will retry':'giving up'}</p>`;
        return ''; }).join('');
      return `<div class="card" style="--c:${col}"><div class="card-h"><span><b>${esc(c.node)}</b> ${c.attempt>1?`<span class="chip">attempt ${c.attempt}</span>`:''}</span><span>${st} <small>${c.ms!=null?c.ms+' ms':''}</small></span></div>${body}</div>`;
    }).join('') || '<div class="empty">Nothing here yet. Run an experiment to see each agent’s steps appear.</div>';
  }
  el.scrollTop = atEnd && S.follow ? el.scrollHeight : keep;
}

/* ----------------------------------------------------------------- inspector */
function renderInspector(v) {
  const el=$('#insp'), run=S.active, tab=S.insp;
  if (tab==='report') {
    const live = v.status==='running'||v.status==='paused';
    const txt = v.report || v.draft;
    el.innerHTML = txt ? `${v.error?`<p class="chip bad">${esc(v.error.slice(0,200))}</p>`:''}${live&&!v.report?'<p class="muted">Draft — not yet approved.</p>':''}<div class="doc">${md(txt)}</div>`
      : `<div class="empty">${v.status==='failed'?`The run failed before a report was written.<br><br>${esc(v.error||'')}`:'The final report will appear here once the Writer has produced it.'}</div>`;
  } else if (tab==='state') {
    const st=v.snapshot?.state, c=v.snapshot?.counts||{};
    if(!st){ el.innerHTML='<div class="empty">State snapshots appear after each node finishes.</div>'; return; }
    const trunc=(o)=>JSON.stringify(o,(k,x)=>typeof x==='string'&&x.length>300?x.slice(0,300)+'…':x,2);
    el.innerHTML=`<div class="stats"><div class="stat"><b>${esc(v.snapshot.task_status)}</b><span>status</span></div><div class="stat"><b>${c.messages}</b><span>messages</span></div><div class="stat"><b>${c.findings}</b><span>findings</span></div><div class="stat"><b>${c.errors}</b><span>errors</span></div></div>
    <div class="sec">Routing history</div><div class="list">${(st.routing_history||[]).map(r=>`<div class="li">${esc(r.from_agent)} → <b>${esc(r.to_agent)}</b><small>${esc(r.reason)}</small></div>`).join('')||'<div class="empty">None yet.</div>'}</div>
    <div class="sec">Findings</div><div class="list">${(st.research_findings||[]).map(f=>`<div class="li">${esc(f.content)}<small>${esc(f.source)} · confidence ${f.confidence}</small></div>`).join('')||'<div class="empty">None yet.</div>'}</div>
    <div class="sec">Error log</div><div class="list">${(st.error_log||[]).map(f=>`<div class="li"><b>${esc(f.node_name)}</b> ${esc(f.error_type)} (attempt ${f.attempt})<small>${esc(f.message)}</small></div>`).join('')||'<div class="empty">No errors recorded.</div>'}</div>
    <details><summary>Raw state (after ${esc(v.snapNode)})</summary><pre>${esc(trunc(st))}</pre></details>`;
  } else if (tab==='audit') {
    const rows=v.tools.map(t=>`<tr><td>${esc(t.node)}</td><td>${esc(t.tool)}</td><td class="mono">${esc(JSON.stringify(t.args))}</td><td>${t.ok?'ok':'<span style="color:var(--bad)">failed</span>'}</td><td class="mono">${esc(String(typeof t.out==='string'?t.out:JSON.stringify(t.out)).slice(0,140))}</td></tr>`).join('');
    el.innerHTML=`<div class="sec">Tool calls</div>${rows?`<table class="audit"><tr><th>Node</th><th>Tool</th><th>Args</th><th></th><th>Result</th></tr>${rows}</table>`:'<div class="empty">No tool calls yet.</div>'}
    <div class="sec">Memory operations</div><div class="list">${v.memory.map(m=>`<div class="li"><b>${esc(m.op)}</b> · ${esc(m.detail)}</div>`).join('')||'<div class="empty">None yet.</div>'}</div>
    <div class="sec">Faults and recovery</div><div class="list">${v.rows.filter(e=>FILTERS.Errors.includes(e.type)).map(e=>`<div class="li">${esc(summ(e))}</div>`).join('')||'<div class="empty">None.</div>'}</div>`;
  } else if (tab==='compare') {
    const runs=(S.group?.runIds||[]).map(id=>{const vv=derive(S.events[id]||[]); return {id,vv,label:vv.meta.label||id};});
    el.innerHTML = runs.length>1 ? `<div class="cols">${runs.map(r=>`<div class="col"><h3>${esc(r.label)} <span class="chip">${esc(r.vv.status)}</span></h3><div class="doc">${md(r.vv.report||r.vv.draft)||'<span class="muted">No report yet.</span>'}</div></div>`).join('')}</div>`
      : '<div class="empty">Compare shows each phase’s report side by side. Try the output lens or memory continuity experiments.</div>';
  } else {
    const id=run||''; const base=`/api/runs/${encodeURIComponent(id)}`;
    el.innerHTML = id?`<div class="exp"><p class="muted">A dossier holds the final deliverable, routing path with reasons, step timeline, and tool and memory audit trails.</p>
      <a href="${base}/dossier.md">Download dossier (Markdown)</a><a href="${base}/dossier.html">Download dossier (HTML)</a><a href="${base}/dossier.json">Download dossier (JSON, replayable)</a><a href="${base}/events.jsonl">Download raw events (JSONL)</a>
      <label class="fbtn">Open a saved trace…<input id="file" type="file" accept=".json,.jsonl"></label></div>`
      : `<div class="exp"><label class="fbtn">Open a saved trace…<input id="file" type="file" accept=".json,.jsonl"></label></div>`;
    const f=$('#file'); f&&f.addEventListener('change',openFile);
  }
}

/* -------------------------------------------------------------------- HITL */
function renderHitl(v) {
  const el=$('#hitl'), g=S.group, live = g && g.status==='paused' && v.status==='paused' && S.cursor>=(S.events[S.active]||[]).length;
  if (v.status!=='paused') { el.hidden=true; el.innerHTML=''; return; }
  const p=v.pausedPayload||{};
  el.hidden=false;
  const draft=`<div class="draft doc">${md(p.draft_report||v.draft||'')}</div>`;
  if (!live) { el.innerHTML=`<h3>Paused at the human gate</h3><p>This run was waiting for a decision at this point.</p>${draft}`; return; }
  if ($('#fb') && el.dataset.rev===String(p.revision_cycle)) return; // don't clobber typing
  el.dataset.rev=String(p.revision_cycle);
  el.innerHTML=`<h3>Your decision is needed</h3><p>${esc(p.supporting_findings_count)} findings back this draft · revision cycle ${esc(p.revision_cycle)}. The state is checkpointed, so you can take your time.</p>${draft}
  <textarea id="fb" rows="2" placeholder="Feedback for the Writer (needed for a revision)"></textarea>
  <div class="btns"><button class="btn-ok" data-d="approved">Approve</button><button class="btn-rev" data-d="revision_requested">Request revision</button><button class="btn-no" data-d="rejected">Reject</button></div>`;
}
async function decide(decision) {
  const fb=$('#fb')?.value.trim()||null;
  if (decision==='revision_requested' && !fb) { $('#fb').focus(); $('#fb').placeholder='Add feedback so the Writer knows what to change'; return; }
  $('#hitl').querySelectorAll('button').forEach(b=>b.disabled=true);
  const r=await fetch(`/api/groups/${S.group.id}/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision,feedback:fb})});
  if(!r.ok){ alert((await r.json()).error); return; }
  S.group.status='running'; connectStream(S.group.id,true);
}

/* ------------------------------------------------------------------ control */
const runEvents = () => S.events[S.active] || [];
function render(animate=false) {
  const evts=runEvents(), slice=evts.slice(0,S.cursor), v=derive(slice);
  paintGraph(v);
  if (animate && v.lastRoute && v.lastRoute.seq>S.animatedSeq) animateToken(v.lastRoute);
  if (v.lastRoute) S.animatedSeq=v.lastRoute.seq;
  renderStream(v); renderInspector(v); renderHitl(v);
  const sc=$('#scrub'); sc.max=evts.length; sc.value=S.cursor;
  $('#cursor-label').textContent=`${S.cursor} / ${evts.length}`;
  const live=S.follow && S.group && !['completed','failed'].includes(S.group.status);
  $('#golive').classList.toggle('live',!!live); $('#play').textContent=S.playing?'Pause':'Replay';
  renderTabs();
}
function renderTabs() {
  const ids=S.group?.runIds||[], el=$('#run-tabs');
  el.innerHTML = ids.length>1?ids.map(id=>{const m=derive((S.events[id]||[]).slice(0,1)).meta; return `<button data-run="${esc(id)}" ${id===S.active?'aria-current="true"':''}>${esc(m.label||id)}</button>`}).join(''):'';
}
function setActive(id, {toEnd=true}={}) { S.active=id; S.cursor=toEnd?runEvents().length:0; stopPlay(); render(); }
function stopPlay(){ S.playing=false; clearInterval(S.timer); S.timer=null; }
function play() {
  if (S.playing) { stopPlay(); render(); return; }
  S.follow=false; if (S.cursor>=runEvents().length) S.cursor=0; S.playing=true; S.animatedSeq=0;
  const tick=()=>{ if(S.cursor>=runEvents().length){stopPlay();render();return} S.cursor++; render(true); };
  const arm=()=>{ clearInterval(S.timer); S.timer=setInterval(tick, 260/+$('#speed').value); }; arm(); $('#speed').onchange=()=>S.playing&&arm(); render();
}
function goLive() { stopPlay(); S.follow=true; S.cursor=runEvents().length; render(); }

/* ----------------------------------------------------------------- transport */
function ingest(e) {
  (S.events[e.run_id] ||= []).push(e);
  if (S.group && !S.group.runIds.includes(e.run_id)) { S.group.runIds.push(e.run_id); if (S.follow) { S.active=e.run_id; S.cursor=0; } }
  if (e.type==='run.paused' && S.group) S.group.status='paused';
}
function connectStream(gid, resume=false) {
  if (S.es) S.es.close();
  const since=Math.max(0,...Object.values(S.events).flat().map(e=>e.seq),0);
  const es=S.es=new EventSource(`/api/groups/${gid}/stream?since=${resume?since:0}`);
  let raf=0;
  es.onmessage=m=>{ const e=JSON.parse(m.data); ingest(e);
    if (S.follow && e.run_id===S.active) { S.cursor=runEvents().length; if(!raf) raf=requestAnimationFrame(()=>{raf=0;render(e.type==='route.decision')}); }
    else if (e.run_id===S.active) { raf||(raf=requestAnimationFrame(()=>{raf=0;render()})); }
    if (e.type==='run.completed'||e.type==='run.failed') refreshGroup(); };
  es.addEventListener('end',()=>{ es.close(); refreshGroup(); loadRuns(); });
  es.onerror=()=>{};
}
async function refreshGroup(){ if(!S.group) return; const r=await fetch(`/api/groups/${S.group.id}`); if(r.ok){ const g=await r.json(); S.group.status=g.status; render(); } }
async function startRun() {
  const sc=S.selected; if(!sc) return;
  const body={scenario:sc.id,variant:$('#variant')?.value||null,query:sc.editable_prompt?$('#prompt').value:null};
  $('#run-btn').disabled=true;
  const r=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  $('#run-btn').disabled=false;
  if(!r.ok){ alert((await r.json()).error); return; }
  const {group_id}=await r.json();
  stopPlay(); S.events={}; S.group={id:group_id,status:'running',runIds:[]}; S.active=null; S.cursor=0; S.follow=true; S.animatedSeq=0; S.expanded.clear();
  render(); connectStream(group_id);
}
async function loadGroupFromServer(groupId) {
  const rows=S.runs.filter(r=>r.group_id===groupId).sort((a,b)=>a.started-b.started);
  if(S.es){S.es.close();S.es=null}
  stopPlay(); S.events={}; S.expanded.clear();
  for (const r of rows) S.events[r.run_id]=await (await fetch(`/api/runs/${encodeURIComponent(r.run_id)}/events`)).json();
  S.group={id:groupId,status:rows.some(r=>r.status==='paused')?'archived':rows.at(-1)?.status||'completed',runIds:rows.map(r=>r.run_id)};
  S.follow=false; setActive(rows.at(-1).run_id);
}
function openFile(ev) {
  const f=ev.target.files[0]; if(!f) return;
  f.text().then(t=>{
    let evts; try { evts=f.name.endsWith('.jsonl')?t.split('\n').filter(Boolean).map(JSON.parse):(JSON.parse(t).events); } catch { alert('That file is not a trace or dossier exported from this viewer.'); return; }
    if(!evts?.length){ alert('No events found in that file. Export the dossier as JSON or the raw events as JSONL.'); return; }
    const id=evts[0].run_id; if(S.es){S.es.close();S.es=null}
    stopPlay(); S.events={[id]:evts}; S.group={id:evts[0].group_id,status:'archived',runIds:[id]}; S.follow=false; setActive(id);
  });
}

/* ---------------------------------------------------------------- left rail */
function renderScenarios() {
  $('#scenario-list').innerHTML=S.scenarios.map(s=>`<button class="scn" role="option" data-id="${s.id}" aria-selected="${S.selected?.id===s.id}"><b>${esc(s.title)}</b><span>${esc(s.tagline)}</span></button>`).join('');
  const s=S.selected, el=$('#scenario-detail');
  if(!s){ el.innerHTML='<div class="empty">Choose an experiment to see what it demonstrates.</div>'; return; }
  el.innerHTML=`<p>${esc(s.description)}</p><div class="muted" style="font-size:12px;margin-bottom:4px">What to look for</div><ul>${s.look_for.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>
  ${s.variants.length?`<label for="variant">Variant</label><select id="variant"><option value="default">Default</option>${s.variants.map(v=>`<option value="${esc(v.id)}">${esc(v.id)} — ${esc(v.note)}</option>`).join('')}</select>`:''}
  ${s.editable_prompt?`<label for="prompt">Prompt</label><textarea id="prompt" rows="3">${esc(s.default_query)}</textarea>`:`<p class="muted" style="margin-top:8px">This experiment uses fixed prompts (${s.phases.map(esc).join(' → ')}).</p>`}
  <button id="run-btn" class="primary">Run experiment</button>`;
}
function renderRuns() {
  const groups=[...new Map(S.runs.map(r=>[r.group_id,[]])).keys()];
  const el=$('#run-list');
  if(!S.runs.length){ el.innerHTML='<div class="empty">No runs yet. Your runs are saved here and can be replayed.</div>'; return; }
  el.innerHTML=groups.map(g=>{ const rs=S.runs.filter(r=>r.group_id===g); const first=rs[rs.length-1], last=rs[0];
    const title=first.meta?.scenario_title||first.run_id; const st=rs.some(r=>r.status==='failed')?'failed':rs.some(r=>r.status==='paused')?'paused':rs.every(r=>r.status==='completed')?'completed':'running';
    return `<button class="run" data-group="${esc(g)}" ${S.group?.id===g?'aria-current="true"':''}><span class="dot ${st}"></span><span>${esc(title)}<small>${rs.length>1?rs.length+' runs · ':''}${st} · ${new Date(last.started*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small></span></button>`; }).join('');
}
async function loadRuns(){ try{ S.runs=await (await fetch('/api/runs')).json(); renderRuns(); }catch{} }

/* --------------------------------------------------------------------- init */
function wire() {
  document.addEventListener('click', e=>{
    const t=e.target.closest('button,.logrow'); if(!t) return;
    if (t.classList.contains('scn')) { S.selected=S.scenarios.find(s=>s.id===t.dataset.id); renderScenarios(); }
    else if (t.id==='run-btn') startRun();
    else if (t.dataset.group) loadGroupFromServer(t.dataset.group).then(renderRuns);
    else if (t.dataset.run) { S.follow=false; setActive(t.dataset.run); }
    else if (t.dataset.d) decide(t.dataset.d);
    else if (t.closest('#stream-tabs')) { S.tab=t.dataset.tab; document.querySelectorAll('#stream-tabs button').forEach(b=>b.setAttribute('aria-selected',b===t)); render(); }
    else if (t.closest('#insp-tabs')) { S.insp=t.dataset.tab; document.querySelectorAll('#insp-tabs button').forEach(b=>b.setAttribute('aria-selected',b===t)); render(); }
    else if (t.closest('#filters')) { const f=t.textContent; S.filter.has(f)?S.filter.delete(f):S.filter.add(f); renderFilters(); render(); }
    else if (t.classList.contains('logrow')) { const q=+t.dataset.seq; S.expanded.has(q)?S.expanded.delete(q):S.expanded.add(q); render(); }
  });
  $('#play').onclick=play; $('#golive').onclick=goLive;
  $('#scrub').oninput=e=>{ stopPlay(); S.follow=false; S.cursor=+e.target.value; S.animatedSeq=Infinity; render(); S.animatedSeq=0; };
  $('#theme').onclick=()=>{ const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'); const nx=cur==='dark'?'light':'dark'; document.documentElement.dataset.theme=nx; store.set('theme',nx); };
  const th=store.get('theme'); if(th) document.documentElement.dataset.theme=th;
}
function renderFilters(){ $('#filters').innerHTML=Object.keys(FILTERS).map(f=>`<button aria-pressed="${S.filter.has(f)}">${f}</button>`).join(''); }
async function init() {
  wire(); renderFilters();
  try {
    [S.scenarios,S.topo]=await Promise.all([fetch('/api/scenarios').then(r=>r.json()),fetch('/api/topology').then(r=>r.json())]);
    $('#conn').textContent='connected'; $('#conn').className='pill on';
  } catch { $('#conn').textContent='server unreachable'; $('#conn').className='pill off'; return; }
  S.selected=S.scenarios[0]; buildGraph(); renderScenarios(); await loadRuns(); render();
}
init();
