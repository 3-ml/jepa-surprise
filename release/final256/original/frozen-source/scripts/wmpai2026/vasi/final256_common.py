"""Frozen final256 contract helpers; no model scoring or scene selection by outcome."""
import hashlib, json, random
from pathlib import Path
COUNT=256
SEED_BASE=2026102700
APPEARANCES=('clean','persistent_hue','abrupt_event')
WORLDS=('solid','gap')
BRANCHES=('lawful','inconsistent')
THRESHOLD=0.4858863055706024
FEATURES=('position','quaternion_xyzw','linear_velocity','angular_velocity')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parameter_fingerprint(config):
    ignored={'scene_id','split','purpose','scene_seed','world'}
    return digest({k:v for k,v in config.items() if k not in ignored})
def trajectory_fingerprint(trace):
    return digest([[row[k] for k in FEATURES] for row in trace])
def build_manifest(development):
    palette=sorted({tuple(c['base_color_rgb']) for c in development['scenes']})
    if len(palette)!=4:raise ValueError('unexpected development palette')
    old_ids={c['scene_id'] for c in development['scenes']};old_seeds={c['scene_seed'] for c in development['scenes']}
    old_fp={parameter_fingerprint(c) for c in development['scenes']};out=[]
    for i in range(COUNT):
        seed=SEED_BASE+i;rng=random.Random(seed)
        c=json.loads(json.dumps(development['scenes'][0]))
        radius=rng.uniform(.60,.70);contact=rng.uniform(1.18,1.24)
        camera=rng.choice((-8,-4,4,8));x=rng.uniform(-.035,.035);color=rng.choice(palette)
        c.update(scene_id=f'final256-{i+1:04d}',scene_seed=seed,split='confirmatory',
                 purpose='frozen final256 native-score test, conditional procedural population',
                 sphere_radius=radius,nominal_contact_time=contact,
                 sphere_initial_position=[x,-.3,radius+.5*9.81*contact**2],base_color_rgb=list(color))
        c['camera_position'][0]=camera
        if c['scene_id'] in old_ids or seed in old_seeds or parameter_fingerprint(c) in old_fp:raise ValueError('development overlap')
        out.append(c)
    if len({parameter_fingerprint(c) for c in out})!=COUNT:raise ValueError('duplicate final configuration')
    return {'schema':1,'status':'frozen_final','count':COUNT,'seed_base':SEED_BASE,
            'rng':'Python random.Random independently seeded per scene; draw radius,time,camera,x,colour in that order',
            'sampling':'independent bounded parameters; camera and four-colour palette independently sampled',
            'scenes':out}

def check_sources(repo,lock):
    for name,h in lock['source_sha256'].items():
        if sha(Path(repo)/name)!=h:raise ValueError('frozen source changed: '+name)

def crossed_and_fresh(scene,seen):
    paths={}
    for w in WORLDS:
        for b in BRANCHES:paths[w,b]=load(Path(scene)/w/(b+'-states.json'))['trace']
    stop=trajectory_fingerprint(paths['solid','lawful']);fall=trajectory_fingerprint(paths['solid','inconsistent'])
    if stop!=trajectory_fingerprint(paths['gap','inconsistent']) or fall!=trajectory_fingerprint(paths['gap','lawful']):raise ValueError('cross-world trajectory mismatch')
    if stop in seen or fall in seen:raise ValueError('trajectory duplicated against inspected or preceding candidate')
    return {'status':'pass','stop_sha256':stop,'fall_sha256':fall,'exact_crossed_target_states':True}
