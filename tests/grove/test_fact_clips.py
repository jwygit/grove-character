import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from lab.grove import load_grove
from tests.evidence_fixtures import written_evidence_graph,grounded_snapshot
from tests.character_fixtures import revoke
from tests.visual_fixtures import FixtureTextIndex,write_index,indexed
load_grove()
from hypervideo.structure import VideoHypergraph,VideoEpisodeNode
from hypervideo.modules.fact_clips import rebuild_fact_clips
from hypervideo.modules.character_projection import rebuild_characters
from hypervideo.modules.character_identity import SnapshotProvider
from hypervideo.retrieval.hierarchical_retriever import HierarchicalRetriever
from hypervideo.config import HyperVideoConfig
from hypervideo.retrieval.multimodal import format_evidence
from hypervideo.mmlifelong_week_retrieval_v2 import _retrieve_day_scoped


class FactClipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.hg,self.snapshot=await written_evidence_graph(self.root)
        self.facts=list(self.hg.facts.values())
        self.dual=FixtureTextIndex(); self.dual.hits['fact']=[(f.id,1.) for f in self.facts]
        self.vr=write_index(self.root/'visual.npz',metadata=[
            dict(id=cid,episode_id='ep1',day=1,start_sec_in_day=a,end_sec_in_day=b)
            for cid,a,b in [('early',0,5),('late',5,10)]],embeddings=[[1,0],[.8,.6]])

    async def test_writer_creates_exact_speech_visual_and_unknown_edges(self):
        speech,unknown,visual,unlocated=self.facts
        self.assertEqual([e['clip_id'] for e in speech.clip_evidence],['early'])
        self.assertEqual([e['clip_id'] for e in unknown.clip_evidence],['late'])
        self.assertIsNone(unknown.character_refs[0]['character_id'])
        self.assertEqual([(e['start'],e['end']) for e in visual.clip_evidence],[(86404,86405),(86405,86406)])
        self.assertEqual(unlocated.clip_evidence,[])
        self.assertEqual(unlocated.evidence_alignment['status'],'unlocalized')
        self.assertEqual(visual.evidence_refs[0]['memory_id'],'description:0')

    async def test_save_reload_and_fact_type_conversion_preserve_evidence(self):
        path=self.root/'graph.json'; self.hg.save(str(path))
        restored=VideoHypergraph.load(str(path))
        self.assertEqual(restored.to_dict(),self.hg.to_dict())
        for fact in restored.facts.values():
            obj=fact.to_video_fact()
            self.assertEqual(obj.evidence_refs,fact.evidence_refs)
            self.assertEqual(obj.to_dict()['clip_evidence'],fact.clip_evidence)

    async def test_identity_revocation_does_not_move_event_evidence(self):
        updated=rebuild_characters(self.hg,revoke(self.snapshot)['identity_state'])
        for fid,fact in updated.facts.items():
            self.assertEqual(fact.clip_evidence,self.hg.facts[fid].clip_evidence)
            self.assertEqual(fact.evidence_refs,self.hg.facts[fid].evidence_refs)
        self.assertIn('person:ep1:v1',updated.facts[self.facts[0].id].content)

    async def test_clip_changes_remove_obsolete_edges_and_are_idempotent(self):
        hg=self.hg.model_copy(deep=True); del hg.visual_clips['late']
        updated=rebuild_fact_clips(hg)
        self.assertEqual(updated.facts[self.facts[1].id].evidence_alignment['status'],'uncovered')
        self.assertEqual(updated.facts[self.facts[2].id].evidence_alignment['status'],'partial')
        self.assertEqual(updated.to_dict(),rebuild_fact_clips(updated).to_dict())
        self.assertEqual(len(self.hg.visual_clips),2)

    async def test_cross_episode_window_and_other_source_do_not_confuse_alignment(self):
        hg=self.hg.model_copy(deep=True)
        hg.episodes['adjacent']=VideoEpisodeNode(id='adjacent',source_id='recording',day=1,start_seconds=86410,end_seconds=86420)
        hg.episodes['other']=VideoEpisodeNode(id='other',source_id='other',day=1,start_seconds=86400,end_seconds=86410)
        hg.add_visual_clip_node('cross',1,86404,86411,'','cross','adjacent',2)
        hg.add_visual_clip_node('wrong-camera',1,86400,86410,'','wrong','other',2)
        result=rebuild_fact_clips(hg)
        ids={e['clip_id'] for e in result.facts[self.facts[2].id].clip_evidence}
        self.assertIn('cross',ids); self.assertNotIn('wrong-camera',ids)

    async def test_bad_memory_scope_rejected_before_writer(self):
        bad=copy.deepcopy(self.snapshot)
        bad['identity_evidence']['memories'][0]['grounding']['scope']['source']='other'
        with self.assertRaisesRegex(ValueError,'source/episode'):
            await SnapshotProvider([bad],'recording').episode_evidence(86400,86410)
        bad=copy.deepcopy(self.snapshot)
        bad['identity_evidence']['memories'][0]['grounding']['anchors'][0]['frame_evidence'][0]['time']=86409
        with self.assertRaisesRegex(ValueError,'frame outside'):
            await SnapshotProvider([bad],'recording').episode_evidence(86400,86410)

    async def test_missing_legacy_evidence_does_not_inherit_episode_cast(self):
        hg=self.hg.model_copy(deep=True)
        fact=hg.facts[self.facts[0].id]; fact.evidence_refs=[]
        fact.clip_evidence=[{'clip_id':'obsolete'}]
        updated=rebuild_fact_clips(hg)
        self.assertEqual(updated.facts[fact.id].clip_evidence,[])
        self.assertEqual(updated.facts[fact.id].evidence_alignment['status'],'no_evidence')

    async def test_invalid_anchor_rebuild_is_transactional(self):
        bad=self.hg.model_copy(deep=True)
        bad.facts[self.facts[0].id].evidence_refs[0]['end']=999999
        before=bad.to_dict()
        with self.assertRaises(ValueError): rebuild_fact_clips(bad)
        self.assertEqual(bad.to_dict(),before)

    async def test_core_retrieval_returns_citations_and_filters_clips_past_cutoff(self):
        config=HyperVideoConfig(use_visual_channel=True,fact_top_k=10)
        result=await HierarchicalRetriever(self.hg,self.dual,config,self.vr).retrieve('query')
        self.assertIn('early@86404-86405',format_evidence(result.evidence))
        self.hg.visual_clips['late'].end_seconds=86412
        self.hg=rebuild_fact_clips(self.hg)
        # The late clip is outside cutoff, so it is not searched or exposed via a fact citation.
        result=await HierarchicalRetriever(self.hg,self.dual,config,self.vr).retrieve('query','DAY1-00:00:00-00:00:10')
        row=next(r for r in result.evidence if r['id']==self.facts[2].id)
        self.assertEqual([e['clip_id'] for e in row['clip_evidence']],['early'])
        self.assertEqual(row['evidence_alignment']['status'],'partial')
        self.assertEqual(len(self.hg.facts[self.facts[2].id].clip_evidence),2)

    async def test_week_entry_keeps_evidence_links_and_stale_links_fail(self):
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever',return_value=self.vr):
            rows,_,_=await _retrieve_day_scoped(indexed(self.hg,self.dual),'query',1,False,
                'hierarchical_v2_day',10,use_visual_channel=True)
        self.assertTrue(any(r.get('clip_evidence') for r in rows))
        self.hg.facts[self.facts[0].id].clip_evidence[0]['start']+=.5
        with self.assertRaisesRegex(ValueError,'stale fact-to-clip'):
            await HierarchicalRetriever(self.hg,self.dual,HyperVideoConfig(use_visual_channel=True),self.vr).retrieve('query')

    async def test_text_only_retrieval_preserves_clip_citations_without_encoder(self):
        self.vr.encoder.encode_text=lambda q:self.fail('visual encoder should not run')
        result=await HierarchicalRetriever(self.hg,self.dual,HyperVideoConfig(use_visual_channel=False),self.vr).retrieve('query')
        self.assertEqual(result.visual_clips,[])
        self.assertTrue(any(r.get('clip_evidence') for r in result.evidence))
        with patch('hypervideo.mmlifelong_week_retrieval_v2._get_visual_retriever',side_effect=AssertionError('disabled')):
            rows,_,_=await _retrieve_day_scoped(indexed(self.hg,self.dual),'query',1,False,
                'hierarchical_v2_day',10,use_visual_channel=False)
        self.assertTrue(any(r.get('clip_evidence') for r in rows))

    async def test_cpu_cli_rebuild_and_independent_alignment_agree(self):
        path=self.root/'graph.json'; self.hg.save(str(path))
        (self.root/'facts.json').write_text(json.dumps([f.model_dump(mode='json') for f in self.facts]))
        from hypervideo.modules.fact_clips import clip_windows
        (self.root/'clips.json').write_text(json.dumps(clip_windows(self.hg)))
        for command in [ ['grove-link-clips','--graph',str(path),'--out',str(self.root/'updated.json')],
            ['align-facts','--facts',str(self.root/'facts.json'),'--clips',str(self.root/'clips.json'),'--out',str(self.root/'aligned.json')]]:
            proc=subprocess.run([sys.executable,'-m','lab',*command],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
        updated=json.loads((self.root/'updated.json').read_text())
        aligned=json.loads((self.root/'aligned.json').read_text())
        for row in aligned:
            self.assertEqual(row['clip_evidence'],updated['facts'][row['id']]['clip_evidence'])
