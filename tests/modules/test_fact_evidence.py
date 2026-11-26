import copy
from pathlib import Path
import unittest
from lab.fact_evidence import align_evidence, select_evidence, validate_grounding
from lab.visual_grounding import apply_locations
from tests.evidence_fixtures import grounded_snapshot


class FactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.snapshot=grounded_snapshot()
        self.scope=dict(source='recording',episode_id='ep1',start=86400,end=86410)
        self.memories={m['id']:m for m in self.snapshot['identity_evidence']['memories']}
        self.utterances={u['utterance_id']:u for u in self.snapshot['character_refs']}
        self.clips=[dict(id='early',source='recording',start=86400,end=86405),
                    dict(id='late',source='recording',start=86405,end=86410)]

    def select(self, **row): return select_evidence(row,self.utterances,self.memories,self.scope)

    def test_speech_uses_utterance_time_even_unknown_speaker(self):
        for uid,clip in [('u1','early'),('u2','late')]:
            refs=self.select(utterance_ids=[uid],start=-100,clip_ids=['invented'])
            result=align_evidence(refs,self.clips)
            self.assertEqual([e['clip_id'] for e in result['clip_evidence']],[clip])
            self.assertEqual(refs[0]['provenance']['semantic_scope'],'spoken_statement')

    def test_visual_interval_crosses_windows_with_exact_intersections(self):
        refs=self.select(memory_ids=['description:0'])
        result=align_evidence(refs,self.clips)
        self.assertEqual([(e['clip_id'],e['start'],e['end']) for e in result['clip_evidence']],
                         [('early',86404,86405),('late',86405,86406)])
        self.assertEqual(result['evidence_alignment']['status'],'complete')

    def test_frame_on_boundary_only_links_next_clip(self):
        memory=self.memories['description:0']
        memory['grounding']['anchors']=[dict(start=86405,end=86405,sample=True,frame_id='f5',
            frame_evidence=[dict(frame_id='f5',time=86405)])]
        result=align_evidence(self.select(memory_ids=['description:0']),self.clips)
        self.assertEqual([e['clip_id'] for e in result['clip_evidence']],['late'])
        self.assertEqual(result['clip_evidence'][0]['frame_id'],'f5')

    def test_unlocalized_description_and_inference_do_not_borrow_face_times(self):
        self.memories['c']=dict(id='c',kind='conclusion',text='A likes coffee',feature_ids=[])
        result=align_evidence(self.select(memory_ids=['description:1','c']),self.clips)
        self.assertEqual(result['clip_evidence'],[])
        self.assertEqual(result['evidence_alignment']['status'],'unlocalized')

    def test_source_isolation_and_same_person_never_create_edges(self):
        clips=[dict(c,source='another-camera',character_refs=[{'character_id':'person:ep1:f1'}]) for c in self.clips]
        self.assertEqual(align_evidence(self.select(utterance_ids=['u1']),clips)['clip_evidence'],[])

    def test_overlapping_clips_do_not_double_count_and_gaps_remain_partial(self):
        refs=self.select(memory_ids=['description:0'])
        clips=[dict(id='a',source='recording',start=86404,end=86404.5),
               dict(id='b',source='recording',start=86404,end=86404.5)]
        result=align_evidence(refs,clips)
        self.assertEqual(result['evidence_alignment']['coverage'][0]['coverage_ratio'],.25)
        self.assertEqual(result['evidence_alignment']['status'],'partial')

    def test_mixed_located_unlocated_and_empty_have_explicit_status(self):
        result=align_evidence(self.select(utterance_ids=['u1'],memory_ids=['description:1']),self.clips)
        self.assertEqual(result['evidence_alignment']['status'],'partial')
        self.assertEqual(align_evidence([],self.clips)['evidence_alignment']['status'],'no_evidence')
        self.assertEqual(align_evidence(self.select(utterance_ids=['u1']),[])['evidence_alignment']['status'],'uncovered')

    def test_rejects_invalid_ids_times_and_stale_memory(self):
        with self.assertRaises(ValueError): self.select(utterance_ids=['u1','u1'])
        with self.assertRaises(ValueError): self.select(memory_ids=['invented'])
        self.utterances['u1']['end']=86420
        with self.assertRaises(ValueError): self.select(utterance_ids=['u1'])
        self.memories['description:0']['text']='changed event'
        with self.assertRaisesRegex(ValueError,'different memory'): self.select(memory_ids=['description:0'])
        for bad in [float('nan'),float('inf'),True]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                align_evidence([], [dict(self.clips[0],start=bad)])

    def test_localization_rejects_unknown_frames_missing_rows_and_unsupported_intervals(self):
        raw=self.snapshot['visual_grounding']['raw']; frames=self.snapshot['visual_grounding']['frames']
        for error in ['missing','duplicate','unknown_frame','out_of_bounds','no_frame']:
            data=copy.deepcopy(raw)
            if error=='missing': data['locations'].pop()
            elif error=='duplicate': data['locations'][1]=data['locations'][0]
            elif error=='unknown_frame': data['locations'][0]['frame_ids']=['invented']
            elif error=='out_of_bounds': data['locations'][0]['intervals'][0]['end']=11
            else: data['locations'][0]['frame_ids']=[]
            with self.subTest(error=error),self.assertRaises(ValueError): apply_locations(self.snapshot,data,frames,{})

    def test_distinct_visual_occurrences_do_not_fill_the_gap(self):
        payload=copy.deepcopy(self.snapshot['visual_grounding']['raw'])
        payload['locations'][0].update(intervals=[dict(start=4,end=4.5),dict(start=8,end=8.5)],frame_ids=['f4','f8'])
        updated=apply_locations(self.snapshot,payload,self.snapshot['visual_grounding']['frames'],{'method':'test'})
        self.memories={m['id']:m for m in updated['identity_evidence']['memories']}
        middle=[dict(id='middle',source='recording',start=86405,end=86407)]
        self.assertEqual(align_evidence(self.select(memory_ids=['description:0']),middle)['clip_evidence'],[])

    def test_contract_copy_matches_grove(self):
        root=Path(__file__).resolve().parents[2]
        self.assertEqual((root/'lab/fact_evidence.py').read_bytes(),
            (root/'vendor/Grove/hypervideo/modules/fact_evidence.py').read_bytes())
