import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests.visual_fixtures import graph, indexed, write_index, FixtureTextIndex
from hypervideo.config import HyperVideoConfig
from hypervideo.structure import VideoHypergraph
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever
from hypervideo.retrieval.multimodal import rank_evidence, format_evidence
from hypervideo.mmlifelong_week_retrieval_v2 import _retrieve_day_scoped, answer_one_question_multi_mode


class VisualRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'visual.npz'
        self.vr = write_index(self.path)
        self.hg, self.dual = graph(), FixtureTextIndex()

    async def retrieve(self, day=1, use_visual=True, **kwargs):
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever', return_value=self.vr) as get:
            result = await _retrieve_day_scoped(indexed(self.hg, self.dual), 'query', day, day is None,
                'hierarchical_v2_day', kwargs.pop('top_k', 4), use_visual_channel=use_visual, **kwargs)
            if not use_visual:
                get.assert_not_called()
            return result

    async def test_mixed_types_share_budget_without_comparing_raw_score_scales(self):
        evidence, parents, _ = await self.retrieve(top_k=2)
        self.assertEqual([(r['node_type'],r['id']) for r in evidence], [('fact','f1'),('visual_clip','c2')])
        self.assertEqual(evidence[0]['score'], evidence[1]['score'])
        self.assertLess(evidence[0]['raw_score'], evidence[1]['raw_score'])
        self.assertEqual({p['id'] for p in parents}, {'ep1','ep2'})
        self.assertIn('[visual_clip:c2]', format_evidence(evidence))

    async def test_day_with_only_visuals_does_not_fall_back_to_other_day_facts(self):
        self.hg.facts = {'f3':self.hg.facts['f3']}
        evidence, parents, _ = await self.retrieve()
        self.assertTrue(evidence)
        self.assertTrue(all(r['node_type']=='visual_clip' and r['day']==1 for r in evidence))
        self.assertNotIn('ep3', {p['id'] for p in parents})

    async def test_unknown_day_stays_empty_and_cross_day_can_retrieve_day_two(self):
        self.assertEqual((await self.retrieve(day=9))[0], [])
        evidence, _, _ = await self.retrieve(day=None)
        self.assertIn('c3', [r['id'] for r in evidence])

    async def test_disable_switch_uses_text_only_and_does_not_load_visual_index(self):
        evidence, _, _ = await self.retrieve(use_visual=False)
        self.assertTrue(evidence)
        self.assertTrue(all(r['node_type']=='fact' for r in evidence))

    async def test_clip_from_unselected_episode_cannot_occupy_top_k(self):
        self.dual.hits['episode'] = [('ep1', 1.)]
        evidence, parents, _ = await self.retrieve(top_k=2)
        self.assertIn('c1', [r['id'] for r in evidence])
        self.assertEqual({p['id'] for p in parents}, {'ep1'})

    async def test_old_within_day_timestamps_are_normalized_without_mutating_graph(self):
        self.hg.visual_clips['c2'].start_seconds = 10
        self.hg.visual_clips['c2'].end_seconds = 20
        evidence, _, _ = await self.retrieve()
        row = next(r for r in evidence if r['id']=='c2')
        self.assertEqual(row['start_seconds'], 86410)
        self.assertEqual(self.hg.visual_clips['c2'].start_seconds, 10)

    async def test_index_graph_parent_and_time_mismatches_are_errors(self):
        self.vr.meta[1]['episode_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            await self.retrieve()
        self.vr.meta[1]['episode_id'] = 'ep2'
        self.vr.meta[1]['end_sec_in_day'] = 21
        with self.assertRaisesRegex(ValueError, 'time range'):
            await self.retrieve()

    async def test_hyperedge_cannot_import_clip_from_another_episode(self):
        self.hg.episode_window_hyperedges['wh:ep1'].relation = {'c3':'window_0'}
        with self.assertRaisesRegex(ValueError, 'different parent'):
            await self.retrieve()

    async def test_all_score_propagation_modes_include_visual_nodes(self):
        for mode in ('none','mult','additive','rrf'):
            with self.subTest(mode=mode):
                evidence, _, _ = await self.retrieve(score_prop=mode)
                self.assertEqual(len(evidence), 4)
                self.assertEqual({r['node_type'] for r in evidence}, {'fact','visual_clip'})

    async def test_same_id_in_two_node_types_is_not_deduplicated(self):
        records = [dict(id='same',node_type=kind,episode_ids=['ep1'],channel_rank=0)
                   for kind in ('fact','visual_clip')]
        rows = rank_evidence(records, [('ep1',1)], [], {}, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['score'], rows[1]['score'])

    async def test_core_retriever_has_visual_only_cold_start_and_temporal_cutoff(self):
        self.hg.facts.clear(); self.hg.fact_hyperedges.clear(); self.hg.topics.clear()
        config = HyperVideoConfig(use_visual_channel=True, fact_top_k=4)
        retriever = HierarchicalRetriever(self.hg, self.dual, config, self.vr)
        result = await retriever.retrieve('query', 'DAY1-00:00:00-00:00:10')
        self.assertTrue(result.has_results())
        self.assertEqual([cid for cid,_ in result.visual_clips], ['c1'])
        self.assertEqual(result.facts, [])
        self.assertIn('[visual_clip:c1]', retriever.format_for_llm(result))

    async def test_core_disabled_channel_retains_fact_only_interface(self):
        self.vr.encoder.encode_text = lambda query: self.fail('disabled visual channel was queried')
        retriever = HierarchicalRetriever(self.hg, self.dual, HyperVideoConfig(use_visual_channel=False), self.vr)
        result = await retriever.retrieve('query', mode='fact_only')
        self.assertTrue(result.facts)
        self.assertEqual(result.visual_clips, [])
        self.assertEqual(result.evidence, [])

    async def test_clip_crossing_current_cutoff_is_excluded(self):
        self.hg.visual_clips['c2'].end_seconds = 86425
        config = HyperVideoConfig(use_visual_channel=True, fact_top_k=4)
        result = await HierarchicalRetriever(self.hg,self.dual,config,self.vr).retrieve(
            'query', 'DAY1-00:00:00-00:00:20')
        self.assertNotIn('c2', [cid for cid,_ in result.visual_clips])

    async def test_real_graph_reload_preserves_clip_links_and_retrieval(self):
        path = Path(self.temp.name)/'graph.json'
        self.hg.save(str(path)); self.hg = VideoHypergraph.load(str(path))
        self.assertEqual(self.hg.episodes['ep2'].window_hyperedge_id, 'wh:ep2')
        self.assertEqual(self.hg.episode_window_hyperedges['wh:ep2'].relation, {'c2':'window_0'})
        self.assertIn('c2', [r['id'] for r in (await self.retrieve())[0]])

    async def test_visual_mode_cannot_be_skipped_by_stage_one_answer(self):
        prompts = []
        async def llm(model, prompt, **kwargs):
            prompts.append(prompt)
            if len(prompts)==1:
                return json.dumps(dict(decision='answer', answer='summary answer'))
            self.assertIn('[visual_clip:c2]', prompt)
            self.assertIn('[fact:f1]', prompt)
            return json.dumps(dict(answer='grounded answer', reasoning='visual_clip:c2'))
        cfg = SimpleNamespace(best_model_func_raw=llm,best_model_name='fake')
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever', return_value=self.vr):
            results = await answer_one_question_multi_mode(dict(question='What is visible on Day 1?'),
                indexed(self.hg,self.dual),cfg,[],stage2_top_k=2,
                retrieval_modes=['dual_fact_day','hierarchical_v2_day'],use_visual_channel=True)
        self.assertEqual(results['dual_fact_day']['stage'], 1)
        result = results['hierarchical_v2_day']
        self.assertEqual(result['stage'], 2)
        self.assertEqual(result['answer'], 'grounded answer')
        clip = next(r for r in result['retrieved'] if r['node_type']=='visual_clip')
        self.assertEqual(clip['clip_id'], 'c2')
        self.assertNotIn('fact_id', clip)
        self.assertEqual(set(result['parent_episode_ids']), {'ep1','ep2'})

    async def test_retrieval_failure_does_not_call_answer_model_with_empty_evidence(self):
        calls = []
        async def llm(*args, **kwargs):
            calls.append(args)
            return json.dumps(dict(decision='answer',answer='must not use'))
        cfg=SimpleNamespace(best_model_func_raw=llm,best_model_name='fake')
        results = await answer_one_question_multi_mode(dict(question='Day 1 visual detail?'),
            indexed(self.hg,self.dual),cfg,[],retrieval_modes=['hierarchical_v2_day'],use_visual_channel=True,
            visual_index_path='/no/such/visual.npz')
        self.assertEqual(results['hierarchical_v2_day']['status'], 'failed')
        self.assertEqual(len(calls), 1)
