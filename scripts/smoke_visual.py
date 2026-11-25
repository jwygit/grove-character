"""Synthetic visual/fact retrieval demo: real graph, NPZ, cosine and Grove retrieval."""
import asyncio
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from tests.visual_fixtures import graph, write_index, FixtureTextIndex
from hypervideo.config import HyperVideoConfig
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever

out = Path(sys.argv[1]) if len(sys.argv)>1 else root/'runs/visual-smoke'
out.mkdir(parents=True, exist_ok=False)
hg = graph()
hg.save(str(out/'hypergraph.json'))
visual = write_index(out/'visual_index.npz')
retriever = HierarchicalRetriever(hg, FixtureTextIndex(),
    HyperVideoConfig(use_visual_channel=True, fact_top_k=4), visual)
result = asyncio.run(retriever.retrieve('fixture query', 'DAY1-00:00:00-00:00:20'))
(out/'evidence.json').write_text(json.dumps(result.evidence, ensure_ascii=False, indent=2)+'\n')
(out/'context.txt').write_text(retriever.format_for_llm(result)+'\n')
(out/'report.json').write_text(json.dumps(dict(synthetic=True, eligible_for_selection=False,
    models_executed=[], scope='fixture text hits/query vector; real graph, NPZ, cosine, mixed retrieval',
    fact_count=len(result.facts), visual_clip_count=len(result.visual_clips), debug=result.debug), indent=2)+'\n')
print(out.resolve())
