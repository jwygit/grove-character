"""Synthetic evidence localization → real Grove writer → exact clip references → retrieval."""
import asyncio
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from lab.common import write
from tests.evidence_fixtures import written_evidence_graph
from tests.visual_fixtures import FixtureTextIndex
from hypervideo.config import HyperVideoConfig
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever
from hypervideo.retrieval.multimodal import format_evidence


async def main(out):
    out.mkdir(parents=True,exist_ok=False)
    hg,snapshot=await written_evidence_graph(out)
    hg.save(str(out/'hypergraph.json')); write(out/'grounded_snapshot.json',snapshot)
    dual=FixtureTextIndex(); dual.hits['fact']=[(fid,1.) for fid in hg.facts]
    result=await HierarchicalRetriever(hg,dual,HyperVideoConfig(use_visual_channel=False,fact_top_k=10)).retrieve('What happened?')
    write(out/'retrieved.json',result.evidence)
    (out/'context.txt').write_text(format_evidence(result.evidence))
    write(out/'report.json',dict(synthetic=True,eligible_for_selection=False,models_executed=[],
        scope='synthetic localization and writer; real Grove persistence, projection and retrieval',
        fact_count=len(hg.facts),clip_count=len(hg.visual_clips),
        edge_count=sum(len(f.clip_evidence) for f in hg.facts.values()),
        alignment_statuses=[f.evidence_alignment['status'] for f in hg.facts.values()]))
    print(out.resolve())


if __name__=='__main__':
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv)>1 else root/'runs/fact-clips-smoke'))
