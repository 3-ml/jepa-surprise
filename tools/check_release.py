"""Offline byte and numeric checks. No model imports, downloads, fits or resampling."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-10):
 assert math.isfinite(a) and abs(a-b)<=tol,(a,b,tol)
def check_manifest():
 m=load('release-manifest.json');seen=set()
 for e in m['entries']:
  n=e['path'];p=PurePosixPath(n)
  assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and n not in seen,n
  seen.add(n);f=ROOT/n
  assert f.is_file() and not f.is_symlink(),n
  assert f.stat().st_size==e['bytes'] and digest(f)==e['sha256'],n
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='release-manifest.json'}
 assert actual==seen,('unmanifested',actual-seen,'missing',seen-actual)
 old=load('historical/archive_manifest.json')
 for e in old['entries']:
  f=ROOT/'historical'/e['path'];assert digest(f)==e['sha256'] and f.stat().st_size==e['bytes'],e['path']
 print(f"manifest: {len(seen)} current files; {len(old['entries'])} historical entries byte-identical")
def final256():
 p=ROOT/'release/final256/original/scores/all-scores.json'
 assert digest(p)=='055700d438bf765d381c9a142bdddaf2624f519bc2850428f9109e38ca309bde'
 rows=json.loads(p.read_text());assert len(rows)==3072
 cells={};scenes=set()
 for r in rows:
  assert r['output_status']=='complete'
  key=(r['scene_id'],r['world'],r['trajectory'],r['appearance_condition'])
  assert key not in cells;cells[key]=r['score'];scenes.add(r['scene_id'])
  assert (r['physical_condition']=='lawful')==((r['world'],r['trajectory']) in {('solid','stop'),('gap','fall')})
 assert len(scenes)==256
 expected_margins={('clean','solid'):-0.006734183640219271,('clean','gap'):0.006907887756824493,('persistent_hue','solid'):-0.007839271682314575,('persistent_hue','gap'):0.007495353929698467}
 expected_rates=[(100,45.70),(37.89,.78),(100,59.38),(30.47,.39),(100,65.62),(28.52,0)]
 expected_auc=[.083,.882,.048,.944,.042,.948];threshold=.4858863055706024
 checks=0;summary=[]
 for app in ['clean','persistent_hue','abrupt_event']:
  for world in ['solid','gap']:
   lawful='stop' if world=='solid' else 'fall';inc='fall' if world=='solid' else 'stop'
   a=[cells[s,world,lawful,app] for s in sorted(scenes)];b=[cells[s,world,inc,app] for s in sorted(scenes)]
   margin=statistics.mean(y-x for x,y in zip(a,b));i=len(summary)
   if (app,world) in expected_margins:close(margin,expected_margins[app,world],1e-9);checks+=1
   fa=100*sum(x>=threshold for x in a)/256;miss=100*sum(y<threshold for y in b)/256
   close(fa,expected_rates[i][0],.01);close(miss,expected_rates[i][1],.01);checks+=2
   auc=sum((y>x)+.5*(y==x) for y in b for x in a)/(256**2)
   close(auc,expected_auc[i],.0005);checks+=1
   close(statistics.mean(a)-statistics.mean(b),-margin,1e-9);checks+=1
   summary.append(dict(world=world,appearance=app,lawful_mean=statistics.mean(a),inconsistent_mean=statistics.mean(b),margin=margin,false_alarm_percent=fa,miss_percent=miss,auroc=auc))
 assert checks==28
 print('final256: 3,072 unique cells / 256 scenes; all 28 numeric acceptance checks passed')
 return cells,summary

def physion(letter,folder):
 prefix=f'release/{folder}/';rows=[json.loads(s) for s in (ROOT/(prefix+f'{letter}-rows.jsonl')).read_text().splitlines()]
 assert len(rows)==(20708 if letter=='A' else 960)
 groups=collections.defaultdict(dict)
 for r in rows:
  k=(r['video_id'],r['segment']);assert r['condition'] not in groups[k];groups[k][r['condition']]=r['scalar_surprise']
 video=load(prefix+f'analysis-{letter}/video-metrics.json');pairs=load(prefix+f'analysis-{letter}/pair-metrics.json');summary=load(prefix+f'analysis-{letter}/summary.json')
 assert len(video)==192 and len(pairs)==96
 cols=summary['columns'];derived=collections.defaultdict(list)
 for (v,seg),d in groups.items():
  if letter=='A':
   residual=d['original']-statistics.mean(d[c] for c in ['static_first','static_middle','static_last'])
   derived[v].append({'mean_original_minus_mean_static':residual,'fraction_nonpositive_residual':float(residual<=0)})
  else:
   vals={c+'_minus_original':d[c]-d['original'] for c in ['persistent_hue','target_only_hue','persistent_brightness','target_only_brightness']}
   vals.update({f'persistent_minus_target_only_{c}':d['persistent_'+c]-d['target_only_'+c] for c in ['hue','brightness']});derived[v].append(vals)
 bypair=collections.defaultdict(list)
 for v in video:
  for c in cols:close(v[c],statistics.mean(x[c] for x in derived[v['video_id']]))
  bypair[v['pair_id']].append(v)
 for p in pairs:
  assert len(bypair[p['pair_id']])==2
  for c in cols:close(p[c],statistics.mean(v[c] for v in bypair[p['pair_id']]))
 for c in cols:close(statistics.mean(p[c] for p in pairs),summary['bootstrap']['metrics'][c]['mean'])
 if letter=='B':
  hybrid=load(prefix+'analysis-B/hybrid-decomposition.json');assert len(hybrid)==768
  for d in hybrid:
   close(d['E11']-d['E00'],d['native_delta_fp64'])
   close(d['target_contribution']+d['predictor_contribution'],d['native_delta_fp64'])
 print(f'Physion {letter}: {len(rows):,} cells; row → video → 96 pairs → all stored summary means'+('; 768 hybrid identities' if letter=='B' else ''))

def geometry(cells):
 prefix='release/geometry-transfer/geometry/'
 table=load(prefix+'physical_margin-table.json');assert len(table)==1536
 for r in table:
  s,w,a=r['scene_id'],r['world'],r['appearance'];law='stop' if w=='solid' else 'fall';inc='fall' if w=='solid' else 'stop'
  close(r['y'],cells[s,w,inc,a]-cells[s,w,law,a])
 print('geometry transfer: 1,536 physical-margin targets match final256; no fitting performed')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--numeric-only',action='store_true');args=parser.parse_args()
 if not args.numeric_only:
  check_manifest()
  from check_paths import run_checks
  run_checks()
 cells,_=final256();physion('A','physion-static');physion('B','physion-appearance');geometry(cells)
if __name__=='__main__':main()
