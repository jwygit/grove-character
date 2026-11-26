"""Unit tests of the new production implementations, without module stubs."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from lab.grove import load_grove
load_grove()
from hypervideo.extractors.llm_utils import call_llm_with_retry, parse_json_loose, strict_llm_calls
from hypervideo.extractors.fact_extractor import VideoFactExtractor
from hypervideo.extractors.fact_deduplicator import FactDeduplicator
from hypervideo.extractors.episode_segmenter import VideoEpisodeSegmenter
from hypervideo.extractors.topic_aggregator import VideoTopicAggregator
from hypervideo.index.bm25_index import IncrementalBM25Index
from hypervideo.index.dual_index import DualIndex
from hypervideo.index.embedding_index import HypergraphEmbeddingIndex
from hypervideo.retrieval.rrf_fusion import reciprocal_rank_fusion
from hypervideo.structure import VideoHypergraph
from hypervideo.types import VideoFact, VideoEpisode, VideoTopic, FrameCaption


def graph():
    hg = VideoHypergraph()
    hg.add_episode_node(VideoEpisode("e", "1-00:00:00-00:00:10", 86400,86410,1, summary="cup"))
    hg.add_fact_node(VideoFact("a", "blue cup", ["e"], "t"))
    hg.add_fact_node(VideoFact("b", "red kettle", ["e"], "t"))
    hg.add_fact_hyperedge("fh", {"a":"core","b":"detail"}, {"a":.9,"b":.1}, "e")
    hg.update_episode_fact_hyperedge_link("e", "fh")
    hg.add_topic_node(VideoTopic("t", "kitchen", "kitchen", ["e"], 86410), "eh")
    hg.add_episode_hyperedge("eh", {"e":"initiating"}, {"e":1.}, "t")
    return hg


class IndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_bm25_cjk_candidates_update_roundtrip(self):
        idx = IncrementalBM25Index()
        idx.upsert("a", "蓝色杯子放在桌上", "fact")
        idx.upsert("b", "红色水壶", "fact")
        self.assertEqual(idx.search("蓝色杯子", "fact")[0][0], "a")
        self.assertEqual(idx.search_within("杯子", {"b"}, "fact"), [])
        idx.upsert("a", "空房间", "fact")
        self.assertEqual(idx.search("杯子", "fact"), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bm25.pkl"
            idx.save(path)
            self.assertEqual(IncrementalBM25Index.load(path).search("水壶", "fact"), idx.search("水壶", "fact"))

    async def test_dense_normalization_candidates_and_weighted_edges(self):
        async def embed(texts):
            return [[3.,0.] if "blue" in text else [0.,4.] for text in texts]
        idx = HypergraphEmbeddingIndex(embed, propagation_mode="none")
        await idx.rebuild_from_hypergraph(graph())
        self.assertEqual(idx.search([1,0], "fact")[0], ("a",1.))
        self.assertEqual(idx.search_within([1,0], {"b"}, "fact")[0][0], "b")
        edge = idx.embeddings.fact_hyperedges["fh"]
        self.assertGreater(edge[0], edge[1]*8)
        with self.assertRaisesRegex(ValueError, "dimension"):
            idx.search([1,0,0], "fact")

    async def test_failed_dense_rebuild_preserves_previous_dual_index(self):
        failure = False
        async def embed(texts):
            return [[0.,0.] if failure else [1.,0.] for _ in texts]
        index = DualIndex(embed)
        await index.rebuild(graph())
        old = index.bm25.documents.copy()
        failure = True
        hg = graph()
        hg.facts["a"].content = "new content"
        with self.assertRaisesRegex(ValueError, "zero"):
            await index.rebuild(hg)
        self.assertEqual(index.bm25.documents, old)

    async def test_rrf_uses_ranks_and_deduplicates_each_channel(self):
        result = dict(reciprocal_rank_fusion([[('a',999),('a',998),('b',1)],[('b',.001)]]))
        self.assertAlmostEqual(result['a'], 1/61)
        self.assertAlmostEqual(result['b'], 1/63+1/61)

    async def test_strict_query_failure_cannot_fall_back_to_bm25(self):
        async def embed(texts):
            raise RuntimeError("offline")
        idx = DualIndex(embed)
        idx.bm25.rebuild_from_hypergraph(graph())
        with strict_llm_calls(), self.assertRaises(RuntimeError):
            await idx.search("cup", "fact")


class ExtractionTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_rejects_nonjson_and_nonfinite(self):
        self.assertEqual(parse_json_loose('```json\n{"ok":true}\n```'), {"ok":True})
        for text in ('text {"ok":true}', '{"n":NaN}', '__import__("os")'):
            with self.assertRaises(ValueError):
                parse_json_loose(text)

    async def test_retry_and_strict_failure(self):
        replies = iter(['{"bad":', '{"ok":true}'])
        async def llm(prompt): return next(replies)
        result, stats = await call_llm_with_retry(llm, "prompt", parser=parse_json_loose, backoff_base=0)
        self.assertTrue(result["ok"])
        self.assertEqual(stats["attempts"], 2)
        with strict_llm_calls(), self.assertRaises(RuntimeError):
            await call_llm_with_retry(llm,"prompt",parser=parse_json_loose,max_retries=1,fallback={"fake":True})

    async def test_boundary_string_boolean_rejected(self):
        async def llm(prompt): return '{"should_end":"false"}'
        s = VideoEpisodeSegmenter(llm, max_retries=1)
        with strict_llm_calls(), self.assertRaises(RuntimeError):
            await s._detect_boundary()

    async def test_fact_cannot_cite_an_unknown_frame(self):
        async def llm(prompt):
            return json.dumps(dict(facts=[dict(content="invented",frame_ids=["frame_99"])]))
        e = VideoEpisode("e","1-00:00:00-00:00:10",86400,86410,1,frame_captions=[
            FrameCaption("DAY1-00:00:00-00:00:10","cup",1,86400,86410,"window","cup")])
        extractor = VideoFactExtractor(llm,max_retries=1)
        with self.assertRaises(RuntimeError):
            await extractor.extract_facts(VideoTopic("t","cup","cup",["e"],86410),[e])

    async def test_unknown_topic_id_and_incomplete_roles_rejected(self):
        async def llm(prompt): return '{"matches":{"invented":{"match":true}}}'
        agg = VideoTopicAggregator(llm,max_retries=1)
        with strict_llm_calls(), self.assertRaises(RuntimeError):
            await agg.process_episode(graph().episodes["e"].to_video_episode(),graph().get_all_topics())

    async def test_real_dedup_updates_links_only_for_same_event(self):
        hg = graph()
        hg.facts['a'].temporal = hg.facts['b'].temporal = 'DAY1-00:00:01'
        async def llm(prompt): return '{"groups":[["a","b"]]}'
        dedup = FactDeduplicator(llm,hg,max_retries=1)
        dedup.schedule(topic_id='t',new_facts=[hg.facts['b'].to_video_fact()],recent_facts=[hg.facts['a'].to_video_fact()])
        await dedup.flush()
        self.assertEqual(set(hg.facts), {'a'})
        self.assertEqual(set(hg.fact_hyperedges['fh'].relation), {'a'})
        self.assertEqual(hg.validate_bidirectional_links(), {})

    async def test_dedup_does_not_collapse_repeated_actions_at_different_times(self):
        hg = graph()
        hg.facts['a'].temporal, hg.facts['b'].temporal = 'DAY1-00:00:01','DAY1-00:00:05'
        async def llm(prompt): return '{"groups":[["a","b"]]}'
        dedup = FactDeduplicator(llm,hg,max_retries=1)
        dedup.schedule(topic_id='t',new_facts=[hg.facts['b'].to_video_fact()],recent_facts=[hg.facts['a'].to_video_fact()])
        with self.assertRaises(RuntimeError): await dedup.flush()
        self.assertEqual(set(hg.facts), {'a','b'})
