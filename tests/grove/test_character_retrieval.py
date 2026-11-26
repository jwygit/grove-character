from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from lab.character_links import bind
from lab.identity_state import export_state
from lab.reused.identity import IdentityGraph
from tests.character_fixtures import written_graph
from tests.visual_fixtures import graph, indexed, write_index, FixtureTextIndex
from hypervideo.config import HyperVideoConfig
from hypervideo.structure import VideoFactNode
from hypervideo.modules.character_projection import rebuild_characters
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever
from hypervideo.mmlifelong_week_retrieval_v2 import _retrieve_day_scoped


class CharacterRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        written,snapshot=await written_graph(root)
        hg=graph(); hg.episodes['ep1']=written.episodes['ep1']
        identity=IdentityGraph(snapshot['graph']); snapshots=[]
        for eid,source in [('ep2','recording'),('ep3','other-camera')]:
            ep=hg.episodes[eid]; ep.source_id=source
            oid=eid+':face'
            identity.observe(dict(id=oid,source=source,modality='face',start=ep.start_seconds+2,
                end=ep.start_seconds+2,embedding=[1.,0.],model='face-fixture'))
            identity.propose(dict(id='link:'+eid,left='ep1:f1',right=oid,method='manual',evidence={'synthetic':True}))
            identity.review('link:'+eid,'confirmed','test','synthetic same person')
            snapshots.append(dict(source=source,start=ep.start_seconds,end=ep.end_seconds,
                face_timeline_complete=True,temporal_evidence=[dict(id=eid+':detection',source=source,
                start=ep.start_seconds+2,end=ep.start_seconds+2,role='visible',identity_observation_ids=[oid])]))
        state=export_state(identity.data,{'Alice':['ep1:f1']})
        for eid in ('ep2','ep3'):
            hg.facts['person:'+eid]=VideoFactNode(id='person:'+eid,content='A person was present.',
                episode_ids=[eid],character_refs=[dict(role='referenced',**bind([eid+':face'],state))])
        self.hg=rebuild_characters(hg,state,snapshots)
        # Make both semantic hierarchy stages miss ep2/ep3 deliberately.
        self.hg.topics['t1'].episode_ids=['ep1']
        self.hg.episode_hyperedges['eh1'].relation={'ep1':'core'}
        self.dual=FixtureTextIndex(); self.dual.hits['episode']=[('ep1',.9)]
        self.vr=write_index(root/'visual.npz')
        self.config=HyperVideoConfig(use_visual_channel=True,fact_top_k=10)

    def retriever(self):
        return HierarchicalRetriever(self.hg,self.dual,self.config,self.vr)

    async def test_person_route_recalls_clips_and_facts_outside_semantic_hierarchy(self):
        plain=await self.retriever().retrieve('unrelated question')
        self.assertNotIn('c2',[r['id'] for r in plain.evidence])
        result=await self.retriever().retrieve('unrelated question',characters=['Alice'])
        for nid in ['c2','c3','person:ep2','person:ep3']:
            hit=next(r for r in result.evidence if r['id']==nid)
            self.assertEqual(hit['retrieval_channels'],['character'])
            self.assertEqual(hit['matched_characters'],['person:ep1:f1'])
        self.assertEqual(set(e for e,_ in result.episodes),{'ep1','ep2','ep3'})

    async def test_source_and_cutoff_still_apply_to_parallel_route(self):
        result=await self.retriever().retrieve('query',characters=['Alice'],source_id='recording')
        self.assertNotIn('c3',[r['id'] for r in result.evidence])
        result=await self.retriever().retrieve('query','DAY1-00:00:00-00:00:10',characters=['Alice'])
        self.assertEqual({p for r in result.evidence for p in r['episode_ids']},{'ep1'})

    async def test_explicit_role_filter_and_visual_disable(self):
        result=await self.retriever().retrieve('query',characters=['Alice'],character_roles=['visible'])
        self.assertNotIn('person:ep2',[r['id'] for r in result.evidence])
        self.assertIn('c2',[r['id'] for r in result.evidence])
        self.config.use_visual_channel=False
        self.vr.encoder.encode_text=lambda q:self.fail('disabled visual encoder')
        result=await self.retriever().retrieve('query',characters=['Alice'])
        self.assertEqual(result.visual_clips,[])
        self.assertIn('person:ep2',[r['id'] for r in result.evidence])

    async def test_low_visual_similarity_does_not_block_explicit_identity_matches(self):
        self.config.visual_min_score=.99
        result=await self.retriever().retrieve('query',characters=['Alice'])
        self.assertIn('c2',[r['id'] for r in result.evidence])
        self.assertTrue(all('visual' not in r['retrieval_channels'] for r in result.evidence))

    async def test_shared_top_k_and_dual_channel_dedup(self):
        result=await self.retriever().retrieve('query',characters=['Alice','person:ep1:v1'])
        clip=next(r for r in result.evidence if r['id']=='c1')
        self.assertEqual(clip['retrieval_channels'],['visual','character'])
        keys=[(r['node_type'],r['id']) for r in result.evidence]
        self.assertEqual(len(keys),len(set(keys)))
        self.config.fact_top_k=2
        self.assertEqual(len((await self.retriever().retrieve('query',characters=['Alice'])).evidence),2)

    async def test_llm_context_preserves_role_labels_without_implying_actor(self):
        from hypervideo.retrieval.multimodal import format_evidence
        result=await self.retriever().retrieve('query',characters=['Alice'])
        context=format_evidence(result.evidence)
        self.assertIn('visible=person:ep1:f1',context)
        self.assertIn('speaker=person:ep1:f1',context)
        self.assertNotIn('actor=',context)

    async def test_day_scoped_entry_uses_parallel_route_and_empty_day_stays_empty(self):
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever',return_value=self.vr):
            for day in (1,9):
                rows,parents,_=await _retrieve_day_scoped(indexed(self.hg,self.dual),'query',day,False,
                    'hierarchical_v2_day',10,use_visual_channel=True,characters=['Alice'])
                if day==9: self.assertEqual(rows,[])
                else:
                    self.assertIn('c2',[r['id'] for r in rows]); self.assertNotIn('c3',[r['id'] for r in rows])
                    self.assertEqual({p['id'] for p in parents},{'ep1','ep2'})

    async def test_day_scoped_source_filter_with_cross_day_enabled(self):
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever',return_value=self.vr):
            rows,_,_=await _retrieve_day_scoped(indexed(self.hg,self.dual),'query',None,True,
                'hierarchical_v2_day',10,use_visual_channel=True,characters=['Alice'],source_id='other-camera')
        self.assertEqual({p for r in rows for p in r['episode_ids']},{'ep3'})
