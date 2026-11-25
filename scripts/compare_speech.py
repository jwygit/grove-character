"""One recording, paired model comparison. Failed runs remain explicit rows."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.common import read, write, digest
from lab.metrics import speech_metrics
p=argparse.ArgumentParser()
p.add_argument('--truth',type=Path,required=True)
p.add_argument('--predictions',type=Path,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); truth=read(a.truth)
if not truth.get('reviewed'):
    raise SystemExit('review the annotation before comparison')
rows=[]; audio_hash=None
for path in a.predictions:
    pred=read(path)
    if pred['media_sha256']!=truth['media_sha256']:
        raise SystemExit('different source media cannot be a paired comparison')
    row={'prediction':str(path),'prediction_sha256':digest(path),'backend':pred['config']['backend'],
         'status':pred['status'],'wall_seconds':pred['wall_seconds']}
    if pred['status']=='completed':
        if audio_hash is None: audio_hash=pred['audio_sha256']
        if audio_hash!=pred['audio_sha256']: raise SystemExit('different decoded audio')
        if abs(pred['duration']-truth['duration'])>.01: raise SystemExit('duration mismatch')
        row['metrics']=speech_metrics(truth['segments'],pred['segments'],truth['duration'])
    rows.append(row)
write(a.out,{'truth_sha256':digest(a.truth),'synthetic':truth.get('synthetic',False),
             'eligible_for_selection':not truth.get('synthetic',False) and all(r['status']=='completed' for r in rows),
             'rows':rows})
