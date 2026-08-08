"""Web dashboard for the self-play lineage (docs/05) — the browser twin of az.dashboard.

    python -m az.webdash runs/hunt0 runs/hunt1
    python -m az.webdash runs/hunt* --port 8722

Serves one self-contained page (no external assets, stdlib-only server) that
re-reads the run logs on every refresh — default every 15 s — so it stays live
while a trainer appends to log.csv. Lineage semantics are az.dashboard's: runs
are concatenated in the order given, the generation axis keeps counting across
restarts, and each run's boundary is marked on every chart. A run dir with no
log.csv yet (a just-launched restart) is noted on the page and skipped until it
produces a generation.

The watch-list notes mirror az.dashboard.advise, with one addition: while the
step scheduler is still in buffer warm-up (steps/gen below the full quota) the
plateau playbook is suppressed — a flat Elo curve is *expected* there, and
"raise sims" advice would be misfiring.

Reading tool only: it never touches a checkpoint.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from az.dashboard import anchor_series, read_log

STEPS_FULL = 600  # az.train default --steps: the scheduler's quota once the buffer is full


def _clean(value):
    """NaN/inf -> None so the payload is strict JSON."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def load_lineage(paths):
    """-> (rows renumbered end to end, run boundaries, run dirs with no log yet)."""
    rows, runs, missing = [], [], []
    for path in paths:
        log = os.path.join(path, "log.csv") if os.path.isdir(path) else path
        name = os.path.basename(os.path.normpath(path))
        if not os.path.exists(log):
            missing.append(name)
            continue
        part = read_log(path)
        runs.append({"name": name, "start": len(rows), "count": len(part)})
        rows.extend(part)
    for i, row in enumerate(rows):
        row["x"] = float(i)
        for key, value in row.items():
            row[key] = _clean(value)
    return rows, runs, missing


def notes(rows):
    """az.dashboard.advise's rules, returned instead of printed, warm-up aware."""
    if not rows:
        return []
    out, last = [], rows[-1]

    draw = last.get("buffer_draw_frac") or last.get("draw_frac")
    if isinstance(draw, float) and draw > 0.40:
        out.append(f"draw fraction {draw:.0%} (>40%): the value head is starving. Playbook step 3 — "
                   "raise --decisive-premium, or upweight endgame plies.")
    fp = last.get("resign_fp_frac")
    if isinstance(fp, float) and fp > 0.05:
        out.append(f"resign false positives {fp:.0%} (>5%, docs/02): --resign-threshold is too hot, "
                   "move it toward -1.")

    plies = [r["avg_plies"] for r in rows[-6:] if isinstance(r.get("avg_plies"), float)]
    resign = [r["resign_frac"] for r in rows[-6:] if isinstance(r.get("resign_frac"), float)]
    if len(plies) >= 6 and plies[-1] < plies[0] * 0.85 and len(resign) >= 6 and resign[-1] > resign[0] + 0.1:
        out.append("game length falling while resignations rise — docs/05 calls that a resign "
                   "threshold that is too hot, not progress.")

    steps = last.get("steps")
    if isinstance(steps, float) and steps < STEPS_FULL:
        out.append(f"buffer warm-up: {steps:.0f} steps/gen of the full {STEPS_FULL} — a flat Elo "
                   "curve is expected until the buffer fills; the plateau playbook does not apply yet.")
        return out

    for name, _nominal, _xs, elo, _lo, _hi in anchor_series(rows):
        if len(elo) >= 6:
            recent, before = sum(elo[-3:]) / 3, sum(elo[-6:-3]) / 3
            if recent - before < 10:
                sims = last.get("sims")
                out.append(f"Elo vs {name} flat over the last 6 evaluations ({before:+.0f} -> {recent:+.0f}). "
                           "Playbook step 1: raise self-play sims"
                           + (f" ({sims:.0f} -> {sims * 1.6:.0f})" if sims else "")
                           + ", restarting from latest.pt into a fresh run dir.")
        break
    return out


def payload(paths, refresh):
    rows, runs, missing = load_lineage(paths)
    anchors = [{"name": name,
                "nominal": nominal,
                "scale": "absolute" if nominal else "relative",
                "xs": xs, "ys": ys, "lo": lo, "hi": hi}
               for name, nominal, xs, ys, lo, hi in anchor_series(rows)]
    hours = sum((r.get(k) or 0.0) for r in rows
                for k in ("sp_seconds", "train_seconds", "eval_seconds")) / 3600
    return {"rows": rows, "runs": runs, "missing": missing, "anchors": anchors,
            "notes": notes(rows),
            "generations": len(rows),
            "games": sum(r.get("games") or 0.0 for r in rows),
            "hours": round(hours, 2),
            "refresh_s": refresh}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chess_alphazero — training</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--edge:#21262d;--fg:#c9d1d9;--dim:#8b949e}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;padding:18px 22px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin-bottom:6px}
h1{font-size:18px;font-weight:600}
#meta{color:var(--dim);font-size:12px}
#err{display:none;background:#3d1d1d;border:1px solid #f85149;border-radius:8px;padding:8px 12px;margin:10px 0}
#stats{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 14px}
.stat{background:var(--card);border:1px solid var(--edge);border-radius:8px;padding:8px 14px;min-width:110px}
.stat b{display:block;font-size:17px;font-weight:600}
.stat span{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
#notes{display:flex;flex-direction:column;gap:6px;margin:0 0 14px}
.note{background:#3a2d12;border:1px solid #9e6a03;border-radius:8px;padding:8px 12px;font-size:13px}
.note.info{background:#10233a;border-color:#1f6feb}
.note.ok{background:#12261a;border-color:#238636}
#charts,#more{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px}
#moretog{color:var(--dim);font-size:12px;cursor:pointer;margin:14px 0;user-select:none}
#moretog:hover{color:var(--fg)}
.card{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:10px 12px;position:relative}
.card h3{font-size:13px;font-weight:600;margin-bottom:2px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:var(--dim);margin-bottom:4px}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}
.tip{position:absolute;pointer-events:none;background:#1c2430;border:1px solid var(--edge);border-radius:6px;padding:6px 8px;font-size:11px;display:none;z-index:2;white-space:nowrap}
.tip b{font-weight:600}
</style>
</head>
<body>
<header><h1>chess_alphazero</h1><div id="meta">loading…</div></header>
<div id="err"></div>
<div id="stats"></div>
<div id="notes"></div>
<div id="charts"></div>
<div id="moretog"></div>
<div id="more"></div>
<script>
'use strict';
const PAL=['#58a6ff','#f778ba','#ffd166','#7ee787','#c77dff','#f8961e','#76e3ea','#e3b341'];
const fE=v=>Math.round(v).toString();
const f0=v=>Math.round(v).toLocaleString('en-US');
const fP=v=>(100*v).toFixed(1)+'%';
const f2=v=>v.toFixed(2),f3=v=>v.toFixed(3),f4=v=>v.toFixed(4);
const W=560,H=230,L=46,R=10,T=12,B=22;
let timer=null;

function col(rows,key){const xs=[],ys=[];for(const r of rows){if(typeof r[key]==='number'){xs.push(r.x);ys.push(r[key])}}return{xs,ys}}

function specs(d){
  const S=[],rows=d.rows;
  const abs=d.anchors.filter(a=>a.scale==='absolute'),rel=d.anchors.filter(a=>a.scale!=='absolute');
  if(abs.length)S.push({title:'Elo — absolute on the frozen node-limited ladder, 95% CI bands',fmt:fE,
    series:abs.map((a,i)=>({label:'vs '+a.name,color:PAL[i],xs:a.xs,ys:a.ys,lo:a.lo,hi:a.hi}))});
  rel.forEach((a,i)=>S.push({title:'Elo vs '+a.name+' (relative)',fmt:fE,refline:0,
    series:[{label:'vs '+a.name,color:PAL[i],xs:a.xs,ys:a.ys,lo:a.lo,hi:a.hi}]}));
  const add=(title,fmt,list,opt)=>{
    const ss=list.map(([key,label],i)=>{const c=col(rows,key);return{label,color:PAL[i],xs:c.xs,ys:c.ys}})
                 .filter(s=>s.ys.length);
    if(ss.length)S.push(Object.assign({title,fmt,series:ss},opt||{}))};
  // core: the charts that answer "is it improving?" — everything else is
  // diagnostics behind the toggle
  add('value loss (leading indicator)',f4,[['value_loss','value loss']]);
  add('value sign agreement on decisive positions',fP,[['value_sign','agreement']]);
  add('white score (0.5 = balanced)',f3,[['white_score','white score']],{refline:0.5});
  add('game outcomes',fP,[['draw_frac','draws'],['resign_frac','resigned'],
      ['resign_fp_frac','resign false pos.'],['buffer_draw_frac','buffer draws']],{refline:0.05});
  add('policy loss',f3,[['policy_loss','policy loss']],{detail:true});
  add('search / policy entropy (nats)',f2,[['visit_entropy','visit'],['policy_entropy','policy']],{detail:true});
  add('game length (plies)',f0,[['avg_plies','avg plies']],{detail:true});
  add('replay buffer (positions)',f0,[['buffer_size','buffer']],{detail:true});
  add('gradient steps per generation',f0,[['steps','steps']],{detail:true});
  add('self-play throughput (games/h)',f0,[['games_per_h','games/h']],{detail:true});
  add('avg eval batch per forward (fleet occupancy)',f0,[['avg_batch','avg batch']],{detail:true});
  return S;
}

function build(spec,xmax,runs){
  let lo=Infinity,hi=-Infinity;
  for(const s of spec.series){
    for(const v of s.ys){if(v<lo)lo=v;if(v>hi)hi=v}
    if(s.lo)for(const v of s.lo)if(typeof v==='number'&&v<lo)lo=v;
    if(s.hi)for(const v of s.hi)if(typeof v==='number'&&v>hi)hi=v;}
  if(spec.refline!=null){lo=Math.min(lo,spec.refline);hi=Math.max(hi,spec.refline)}
  if(!isFinite(lo))return null;
  if(hi-lo<1e-9){lo-=1;hi+=1}
  const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
  const X=x=>L+x/Math.max(xmax,1)*(W-L-R),Y=y=>T+(1-(y-lo)/(hi-lo))*(H-T-B);
  let g='';
  for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3,y=Y(v);
    g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#21262d"/>`
      +`<text x="${L-5}" y="${y+3}" text-anchor="end" fill="#8b949e" font-size="10">${spec.fmt(v)}</text>`}
  for(const r of runs){
    if(r.start===0&&runs.length===1)continue;
    const x=X(r.start);
    g+=`<line x1="${x}" x2="${x}" y1="${T}" y2="${H-B}" stroke="#30363d" stroke-dasharray="3 3"/>`
      +`<text x="${x+3}" y="${T+8}" fill="#8b949e" font-size="9">${r.name}</text>`}
  if(spec.refline!=null){const y=Y(spec.refline);
    g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#8b949e" stroke-dasharray="5 4" opacity=".55"/>`}
  for(const s of spec.series){
    if(s.lo&&s.hi){
      let pts='';
      for(let i=0;i<s.xs.length;i++)pts+=`${X(s.xs[i])},${Y(s.hi[i])} `;
      for(let i=s.xs.length-1;i>=0;i--)pts+=`${X(s.xs[i])},${Y(s.lo[i])} `;
      g+=`<polygon points="${pts}" fill="${s.color}" opacity=".13"/>`}
    let p='';
    s.xs.forEach((x,i)=>{p+=(i?'L':'M')+X(x).toFixed(1)+' '+Y(s.ys[i]).toFixed(1)});
    g+=`<path d="${p}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`;
    const n=s.xs.length-1;
    g+=`<circle cx="${X(s.xs[n])}" cy="${Y(s.ys[n])}" r="2.6" fill="${s.color}"/>`}
  g+=`<text x="${L}" y="${H-6}" fill="#8b949e" font-size="10">gen 0</text>`
    +`<text x="${W-R}" y="${H-6}" text-anchor="end" fill="#8b949e" font-size="10">gen ${xmax}</text>`
    +`<line class="cross" x1="-9" x2="-9" y1="${T}" y2="${H-B}" stroke="#8b949e" opacity=".5"/>`;
  return`<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

function attachHover(card,spec,xmax,runs){
  const svg=card.querySelector('svg');if(!svg)return;
  const cross=svg.querySelector('.cross');
  const tip=document.createElement('div');tip.className='tip';card.appendChild(tip);
  svg.addEventListener('mousemove',e=>{
    const rc=svg.getBoundingClientRect();
    const px=(e.clientX-rc.left)/rc.width*W;
    let gen=Math.round((px-L)/(W-L-R)*Math.max(xmax,1));
    gen=Math.max(0,Math.min(xmax,gen));
    cross.setAttribute('x1',L+gen/Math.max(xmax,1)*(W-L-R));
    cross.setAttribute('x2',L+gen/Math.max(xmax,1)*(W-L-R));
    let run='';for(const r of runs)if(gen>=r.start)run=r.name;
    let html=`<b>gen ${gen}</b> · ${run}`;
    for(const s of spec.series){
      let best=-1,dist=1e9;
      s.xs.forEach((x,i)=>{const d=Math.abs(x-gen);if(d<dist){dist=d;best=i}});
      if(best>=0&&dist<=2){
        html+=`<br><i style="color:${s.color}">●</i> ${s.label}: <b>${spec.fmt(s.ys[best])}</b>`;
        if(s.lo&&s.hi)html+=` <span style="color:#8b949e">[${spec.fmt(s.lo[best])}, ${spec.fmt(s.hi[best])}]</span>`}}
    tip.innerHTML=html;tip.style.display='block';
    const cr=card.getBoundingClientRect();
    let tx=e.clientX-cr.left+14;
    if(tx>cr.width-170)tx=e.clientX-cr.left-tip.offsetWidth-14;
    tip.style.left=tx+'px';tip.style.top=(e.clientY-cr.top+6)+'px';});
  svg.addEventListener('mouseleave',()=>{tip.style.display='none';cross.setAttribute('x1',-9);cross.setAttribute('x2',-9)});
}

function chips(d){
  const out=[],last=d.rows[d.rows.length-1]||{};
  for(const a of d.anchors){
    const n=a.ys.length-1;if(n<0)continue;
    const v=a.scale==='absolute'?fE(a.ys[n]):(a.ys[n]>=0?'+':'')+fE(a.ys[n]);
    out.push([v,`Elo vs ${a.name} · CI ${fE(a.lo[n])}–${fE(a.hi[n])}`])}
  if(typeof last.x==='number')out.push([f0(last.x),'latest generation']);
  if(typeof last.buffer_size==='number')out.push([f0(last.buffer_size),'buffer positions']);
  if(typeof last.steps==='number')out.push([f0(last.steps),'steps last gen']);
  if(typeof last.draw_frac==='number')out.push([fP(last.draw_frac),'draws last gen']);
  if(typeof last.resign_fp_frac==='number')out.push([fP(last.resign_fp_frac),'resign false pos.']);
  if(typeof last.games_per_h==='number')out.push([f0(last.games_per_h),'games / h']);
  return out.map(([b,s])=>`<div class="stat"><b>${b}</b><span>${s}</span></div>`).join('');
}

function render(d){
  document.getElementById('err').style.display='none';
  document.getElementById('meta').textContent=
    `${d.generations} generations · ${f0(d.games)} self-play games · ${d.hours} h compute · `+
    `updated ${new Date().toLocaleTimeString()} · refresh ${d.refresh_s}s`;
  document.getElementById('stats').innerHTML=chips(d);
  let notes='';
  for(const n of d.notes)notes+=`<div class="note">${n}</div>`;
  for(const m of d.missing)notes+=`<div class="note info">${m}: no log.csv yet — will appear after its first generation.</div>`;
  if(!d.notes.length&&d.rows.length)notes+=`<div class="note ok">nothing on the watch-list is firing.</div>`;
  if(!d.rows.length)notes+=`<div class="note info">no generations logged yet.</div>`;
  document.getElementById('notes').innerHTML=notes;
  const xmax=Math.max(d.rows.length-1,1);
  const core=document.getElementById('charts'),more=document.getElementById('more');
  core.innerHTML='';more.innerHTML='';extraCount=0;
  for(const spec of specs(d)){
    const svg=build(spec,xmax,d.runs);if(!svg)continue;
    const card=document.createElement('div');card.className='card';
    const legend=spec.series.map(s=>{
      const n=s.ys.length-1;
      return`<span><i style="background:${s.color}"></i>${s.label} ${n>=0?spec.fmt(s.ys[n]):''}</span>`}).join('');
    card.innerHTML=`<h3>${spec.title}</h3><div class="legend">${legend}</div>${svg}`;
    if(spec.detail){more.appendChild(card);extraCount++}else core.appendChild(card);
    attachHover(card,spec,xmax,d.runs);}
  applyToggle();
}

let extraCount=0,showAll=localStorage.getItem('az_showall')==='1';
function applyToggle(){
  document.getElementById('more').style.display=showAll?'grid':'none';
  document.getElementById('moretog').textContent=
    showAll?'▾ hide diagnostic charts':`▸ show ${extraCount} diagnostic charts`;
}
document.getElementById('moretog').onclick=()=>{
  showAll=!showAll;
  localStorage.setItem('az_showall',showAll?'1':'0');
  applyToggle();
};

async function tick(){
  try{
    const r=await fetch('data.json',{cache:'no-store'});
    const d=await r.json();
    if(d.error)throw new Error(d.error);
    render(d);
    if(!timer)timer=setInterval(tick,(d.refresh_s||15)*1000);
  }catch(e){
    const el=document.getElementById('err');
    el.textContent='dashboard error: '+e.message;
    el.style.display='block';}
}
tick();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    runs = []
    refresh = 15.0

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        elif route == "/data.json":
            try:
                data = payload(self.runs, self.refresh)
            except Exception as exc:  # a half-written csv row must not 500 the page
                data = {"error": f"{type(exc).__name__}: {exc}"}
            body, ctype = json.dumps(data, allow_nan=False).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (or log.csv paths), oldest first")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8722)
    ap.add_argument("--refresh", type=float, default=15, help="browser auto-refresh seconds")
    args = ap.parse_args(argv)

    Handler.runs = args.runs
    Handler.refresh = args.refresh
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {' + '.join(args.runs)} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
