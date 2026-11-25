"""Synthetic wiring demo, deliberately not a model accuracy experiment."""
import asyncio
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from lab.common import write
from lab.association import resolve
from lab.grove import write_episode
from lab.metrics import speech_metrics
from tests.fixtures import bundle,predictions,episode

out=Path(sys.argv[1]) if len(sys.argv)>1 else root/'runs/smoke'
out.mkdir(parents=True,exist_ok=False)
snapshot=resolve(bundle(),predictions(),accept_links=True)
write(out/'snapshot.json',snapshot)
async def fake_llm(prompt):
    if '"facts"' in prompt:
        return json.dumps({'facts':[{'content':'person:ep1:f1 says hello','utterance_ids':['u1']},
                                   {'content':'An unknown speaker says yes','utterance_ids':['u2']}]})
    return json.dumps({'summary':'A conversation','subject':'Conversation'})
asyncio.run(write_episode(episode(),snapshot,fake_llm,out/'hypergraph.json'))
write(out/'report.json',{'synthetic':True,'eligible_for_selection':False,
                        'scope':'fixture evidence and fake LLM, real Grove graph persistence',
                        'models_executed':[]})
print(out.resolve())
