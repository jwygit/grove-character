import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from lab.identity_state import export_state, review_identity
from lab.reused.identity import IdentityGraph
from lab.grove import load_grove
load_grove()
from tests.character_fixtures import written_graph, revoke
from hypervideo.structure import VideoHypergraph, VideoEpisodeNode
from hypervideo.modules.character_projection import rebuild_characters, character_hits, refresh_characters
from hypervideo.modules.character_cache import ensure_character_text_index


class CharacterProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.hg,self.snapshot=await written_graph(self.root)

    async def test_revoke_rewrites_fact_summary_refs_and_index_but_preserves_raw(self):
        original=self.hg.to_dict()
        state=revoke(self.snapshot)['identity_state']
        updated=rebuild_characters(self.hg,state)
        contents=[f.content for f in updated.facts.values()]
        self.assertIn('<person:ep1:v1> says hello.',contents)
        self.assertIn('<person:ep1:v1> says again.',contents) # direct character token was bound to voice evidence
        self.assertIn('<person:ep1:f1> waves.',contents)
        self.assertEqual(updated.episodes['ep1'].summary,'<person:ep1:f1> watches <person:ep1:v1>.')
        for f in updated.facts.values():
            self.assertEqual(f.identity_evidence['raw_content'],self.hg.facts[f.id].identity_evidence['raw_content'])
        self.assertEqual(self.hg.to_dict(),original)
        self.assertEqual(len(character_hits(updated,['person:ep1:v1'],['speaker'])),4)
        self.assertEqual(refresh_characters(updated).to_dict(),updated.to_dict())

    async def test_split_voice_feature_is_unknown_but_each_utterance_keeps_its_identity(self):
        state=revoke(self.snapshot,split_voice=True)['identity_state']
        updated=rebuild_characters(self.hg,state)
        ep=updated.episodes['ep1']
        self.assertEqual(ep.subject,'<unknown> speaks')
        speakers={r['utterance_id']:r['character_id'] for r in ep.character_refs}
        self.assertNotEqual(speakers['u1'],speakers['u3'])
        self.assertIn('<unknown> says hello.',[f.content for f in updated.facts.values()])

    async def test_reconfirm_merge_and_reload_do_not_accumulate_old_ids(self):
        reviewed=revoke(self.snapshot)
        updated=rebuild_characters(self.hg,reviewed['identity_state'])
        changes={'reviews':[dict(link_id=lid,status='confirmed',reviewer='test',reason='merge again')
            for lid,link in reviewed['graph']['links'].items() if link['status']=='revoked']}
        state=review_identity(reviewed,changes)['identity_state']
        merged=rebuild_characters(updated,state)
        path=self.root/'graph.json'; merged.save(str(path)); restored=VideoHypergraph.load(str(path))
        self.assertEqual(merged.to_dict(),restored.to_dict())
        self.assertEqual(set(restored.character_index['postings']),{'person:ep1:f1'})
        self.assertEqual(len(character_hits(restored,['person:ep1:v1'])),5)

    async def test_clip_spanning_episode_boundary_gets_both_parents_and_only_actual_events(self):
        hg=self.hg.model_copy(deep=True)
        hg.episodes['ep2']=VideoEpisodeNode(id='ep2',source_id='recording',day=1,start_seconds=86410,end_seconds=86420)
        hg.visual_clips['late'].end_seconds=86412
        updated=rebuild_characters(hg,self.snapshot['identity_state'])
        clip=updated.visual_clips['late']
        self.assertEqual(clip.episode_ids,['ep1','ep2'])
        self.assertFalse(clip.identity_evidence['face_timeline_complete'])
        self.assertTrue(all(ref['end']<86410 for ref in clip.character_refs))
        self.assertIn('late',updated.episode_window_hyperedges[updated.episodes['ep2'].window_hyperedge_id].relation)

    async def test_projection_failure_does_not_partially_mutate_original(self):
        state=copy.deepcopy(self.snapshot['identity_state'])
        del state['observations']['ep1:f1']
        from lab.character_links import seal_state
        before=self.hg.to_dict()
        with self.assertRaises(ValueError): rebuild_characters(self.hg,seal_state(state))
        self.assertEqual(before,self.hg.to_dict())

    async def test_detects_stale_index_after_text_or_scope_change(self):
        for kind in ['text','scope','refs']:
            hg=self.hg.model_copy(deep=True)
            if kind=='text': hg.visual_clips['late'].caption_text='new'
            elif kind=='scope': hg.visual_clips['late'].end_seconds+=1
            else: hg.visual_clips['late'].character_refs=[]
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'stale'):
                character_hits(hg,['person:ep1:f1'])

    async def test_content_cache_rebuilds_on_identity_change_even_when_node_counts_same(self):
        calls=[]
        async def rebuild(hg): calls.append('rebuild')
        dual=SimpleNamespace(rebuild=rebuild,save=lambda *a:calls.append('save'),load=lambda *a:calls.append('load'))
        builder=SimpleNamespace(hypergraph=self.hg,mark_index_clean=lambda:None,index_dirty=False)
        owner=SimpleNamespace(hypergraph_builder=builder,dual_index=dual,working_dir=self.temp.name,hyper_config=None)
        await ensure_character_text_index(owner); await ensure_character_text_index(owner)
        self.assertEqual(calls,['rebuild','save'])
        del owner._character_text_version
        await ensure_character_text_index(owner)
        self.assertEqual(calls[-1],'load')
        builder.hypergraph=rebuild_characters(self.hg,revoke(self.snapshot)['identity_state'])
        await ensure_character_text_index(owner)
        self.assertEqual(calls.count('rebuild'),2)
        self.assertTrue((self.root/'character_text_index.json').exists())

    async def test_failed_cache_save_does_not_publish_fresh_marker(self):
        async def rebuild(hg): pass
        def fail(*args): raise OSError('disk full')
        builder=SimpleNamespace(hypergraph=self.hg,mark_index_clean=lambda:None)
        owner=SimpleNamespace(hypergraph_builder=builder,working_dir=self.temp.name,hyper_config=None,
            dual_index=SimpleNamespace(rebuild=rebuild,save=fail))
        with self.assertRaises(OSError): await ensure_character_text_index(owner)
        self.assertFalse((self.root/'character_text_index.json').exists())

    async def test_topic_ids_invalidated_and_clip_caption_rebound_after_split(self):
        from hypervideo.structure import VideoTopicNode
        hg=self.hg.model_copy(deep=True)
        hg.topics['t']=VideoTopicNode(id='t',title='Meeting <person:ep1:f1>',summary='<person:ep1:f1> talks')
        clip=hg.visual_clips['late']
        clip.caption_text='<person:ep1:f1> is present'; clip.identity_evidence.pop('caption_raw')
        prepared=rebuild_characters(hg,self.snapshot['identity_state'])
        updated=rebuild_characters(prepared,revoke(self.snapshot)['identity_state'])
        # Old caption token had face AND voice support: split makes its attribution ambiguous.
        self.assertEqual(updated.visual_clips['late'].caption_text,'<unknown> is present')
        self.assertEqual(updated.topics['t'].summary,'<unknown> talks')
        self.assertEqual(updated.character_catalog['invalidated_topics'],['t'])

    async def test_legacy_identity_refs_require_regeneration_instead_of_silent_migration(self):
        hg=self.hg.model_copy(deep=True)
        hg.episodes['ep1'].character_refs[0].pop('identity_observation_ids')
        with self.assertRaisesRegex(ValueError,'legacy character'):
            rebuild_characters(hg,revoke(self.snapshot)['identity_state'])

    async def test_parent_source_mutation_invalidates_reverse_index(self):
        hg=self.hg.model_copy(deep=True); hg.episodes['ep1'].source_id='wrong'
        with self.assertRaisesRegex(ValueError,'stale'): character_hits(hg,['person:ep1:f1'])

    async def test_conflicting_identity_branch_requires_explicit_selection(self):
        from lab.character_links import seal_state
        hg=self.hg.model_copy(deep=True)
        state=copy.deepcopy(self.snapshot['identity_state']); state['names']={'Alice':['ep1:f1']}
        hg.character_catalog['state']=seal_state(state)
        with self.assertRaisesRegex(ValueError,'conflicting'): refresh_characters(hg)
