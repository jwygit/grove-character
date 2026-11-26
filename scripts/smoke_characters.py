"""CPU demonstration of timeline → Grove → retrieval → revoke → rebuild. Synthetic only."""
import asyncio
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from lab.common import write
from tests.character_fixtures import written_graph, revoke
from tests.visual_fixtures import FixtureTextIndex, write_index
from hypervideo.config import HyperVideoConfig
from hypervideo.modules.character_projection import rebuild_characters
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever


async def main(out):
    out.mkdir(parents=True,exist_ok=False)
    graph,snapshot=await written_graph(out)
    write(out/'snapshot.json',snapshot)
    graph.save(str(out/'before.json'))
    reviewed=revoke(snapshot)
    write(out/'reviewed_state.json',reviewed)
    updated=rebuild_characters(graph,reviewed['identity_state'])
    updated.save(str(out/'after.json'))
    meta=[dict(id=cid,episode_id='ep1',day=1,start_sec_in_day=a,end_sec_in_day=b)
          for cid,a,b in [('early',0,5),('late',5,10)]]
    visual=write_index(out/'visual.npz',metadata=meta,embeddings=[[1,0],[.8,.6]])
    config=HyperVideoConfig(use_visual_channel=True,fact_top_k=10)
    result=await HierarchicalRetriever(updated,FixtureTextIndex(),config,visual).retrieve(
        'Who spoke?',characters=['person:ep1:v1'],character_roles=['speaker'])
    write(out/'evidence.json',result.evidence)
    write(out/'report.json',dict(synthetic=True,eligible_for_selection=False,models_executed=[],
        state_before=snapshot['identity_state']['state_id'],state_after=reviewed['identity_state']['state_id'],
        visible_samples=sum(r['role']=='visible' for r in snapshot['temporal_evidence']),
        facts=len(updated.facts),clips=len(updated.visual_clips),retrieved=len(result.evidence),
        scope='real Grove writer/projections/persistence/retrieval; synthetic evidence, text index and query vectors'))
    print(out.resolve())


if __name__=='__main__':
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv)>1 else root/'runs/characters-smoke'))
