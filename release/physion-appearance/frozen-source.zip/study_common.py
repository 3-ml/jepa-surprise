from pathlib import Path
import datetime,hashlib,json,os,time,sys,importlib.metadata
ROOT=Path(__file__).resolve().parent
BASE=Path(__import__("os").environ['JEPA_VASI_WORKSPACE'])
REPO=Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
OLD=BASE/'reviewer-physion-desktop-20261026-v2'
INPUTS=BASE/'reviewer-physion-canonical-20261026-v2'
def sha(p):
 with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def append(p,o):
 with Path(p).open('a',encoding='utf-8') as h:h.write(json.dumps(o,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
def deadline(*args, **kwargs):
    raise RuntimeError("Historical acquisition entry point excluded; use offline numeric checks")
def check_freeze():
 f=load(ROOT/'freeze.json')
 for n,h in f['files'].items():assert sha(ROOT/n)==h,('source hash',n)
 for n,h in f['external_files'].items():assert sha(n)==h,('external hash',n)
 for n,v in f['versions'].items():assert importlib.metadata.version(n)==v,('runtime version',n)
 return f
def wait_permit(*args, **kwargs):
    raise RuntimeError("Historical acquisition entry point excluded; use offline numeric checks")
def inference_guard(*args, **kwargs):
    raise RuntimeError("Historical acquisition entry point excluded; use offline numeric checks")
