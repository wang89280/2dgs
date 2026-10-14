"""Read repair-run state on Windows or WSL, without launching anything."""
import json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def local(p):
    if os.name=='nt' and p.startswith('/home/'):
        return Path('\\\\wsl.localhost\\Ubuntu-20.04'+p.replace('/','\\'))
    return Path(p)
def main():
    p=read(OUT/'protocol.json');s=read(OUT/'status.json');rows=[]
    for method in p['methods']:
        row=dict(method=method,training_complete=(OUT/f'trained_{method}.json').exists(),
            evaluation_complete=(OUT/'results'/method/'metrics.json').exists())
        f=local(p['models'][method])/'stability_progress.json'
        try:
            if f.exists():row.update(read(f))
        except OSError as e:row['model_access_error']=str(e)
        log=OUT/'logs'/f'train_{method}.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-16000));text=stream.read().decode('utf8',errors='replace')
            steps=re.findall(r'(\d+)/30000',text)
            if steps:row['log_iteration']=int(steps[-1])
            row['log_bytes']=log.stat().st_size
        rows.append(row)
    print(json.dumps(dict(status=s,runs=rows),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
