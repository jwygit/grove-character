import copy
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from lab.voices import eligible_turns, slice_pcm, group_voices
from lab.reused.identity import IdentityGraph
from lab.association import prepare_association, associate, normalize_voice_prediction, resolve
from lab.reused.backends import prompt
from tests.fixtures import bundle


def observe(graph,key,vector,model='v1'):
    graph.observe(dict(id=key,modality='voice',source='clip',start=len(graph.data['observations'])*3,
                       end=len(graph.data['observations'])*3+2,model=model,embedding=vector))


def repeated_bundle():
    b=bundle()
    b['utterances'].append(dict(id='u3',start=6.,end=9.,text='again',
                                voice_observation_id='ep1:v3',speaker='different-local-label'))
    b['utterances'][0]['speaker']='local-0'
    b['observations'].append(dict(b['observations'][1],id='ep1:v3',start=86406.,end=86409.))
    return b


class VoiceTests(unittest.TestCase):
    def test_common_duration_gate_ignores_frontend_overlap_labels(self):
        turns=[dict(id='a',start=0,end=1.99),dict(id='b',start=0,end=2,overlap=True),
               dict(id='c',start=0,end=2)]
        self.assertEqual([u['id'] for u in eligible_turns(turns)],['b','c'])

    def test_lossless_pcm_fractional_slice(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.wav';target=Path(d)/'slice.wav'
            data=struct.pack('<16000h',*range(16000))
            with wave.open(str(source),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(data)
            slice_pcm(source,target,.125,.75)
            with wave.open(str(target),'rb') as f:
                self.assertEqual(f.getnframes(),10000)
                self.assertEqual(f.readframes(f.getnframes()),data[4000:24000])
            with self.assertRaises(ValueError): slice_pcm(source,target,0,1.1)

    def test_mean_similarity_not_best_sample(self):
        g=IdentityGraph();observe(g,'a',[1,0]);observe(g,'b',[.6,.8])
        group_voices(g,['a','b'],threshold=.5)
        observe(g,'c',[1,0])
        result=group_voices(g,['c'],threshold=.9)
        self.assertNotEqual(result['c'],g.data['voice_membership']['a'])

    def test_new_voice_and_model_isolation(self):
        g=IdentityGraph();observe(g,'a',[1,0]);observe(g,'b',[-1,0]);observe(g,'c',[1,0],model='v2')
        result=group_voices(g,['a','b','c'])
        self.assertEqual(len(set(result.values())),3)

    def test_twenty_sample_cap_persistence_and_replay(self):
        g=IdentityGraph()
        for i in range(25): observe(g,str(i),[1,0])
        ids=[str(i) for i in range(25)]
        result=group_voices(g,ids,seed=42)
        self.assertEqual(len(set(result.values())),1)
        node=next(iter(g.data['voice_nodes'].values()))
        self.assertEqual(len(node['observation_ids']),20)
        self.assertEqual(len(g.data['observations']),25)
        saved=json.loads(json.dumps(g.data)); restored=IdentityGraph(saved)
        group_voices(restored,ids,seed=42)
        self.assertEqual(restored.data,saved)
        for graph in [g,restored]:
            observe(graph,'next',[1,0]);group_voices(graph,['next'],seed=42)
        self.assertEqual(g.data,restored.data)

    def test_legacy_migration_does_not_group_by_shared_face(self):
        g=IdentityGraph();observe(g,'a',[1,0]);observe(g,'b',[-1,0])
        g.observe(dict(id='f',modality='face',source='clip',start=0,end=1,model='face',embedding=[1,0]))
        for key in ['a','b']:
            g.propose(dict(id=key,left=key,right='f',method='audiovisual',evidence={'test':True}))
            g.review(key,'confirmed','test','test')
        observe(g,'new',[1,0]);group_voices(g,['new'])
        self.assertNotEqual(g.data['voice_membership']['a'],g.data['voice_membership']['b'])
        self.assertEqual(g.data['voice_membership']['a'],g.data['voice_membership']['new'])

    def test_grouping_happens_before_model_and_preserves_offscreen_identity(self):
        b=repeated_bundle()
        outer=self
        class Backend:
            def infer(self,clip,mode):
                outer.assertEqual(mode,'voice_association')
                outer.assertEqual(len(clip['voice_groups']),1)
                group=clip['voice_groups'][0]
                outer.assertEqual([u['id'] for u in group['utterances']],['u1','u3'])
                outer.assertNotIn('speaker',group['utterances'][0])
                return {'voice_links':[dict(voice_id=group['voice_id'],face_id='ep1:f1')],
                        'segments':[dict(utterance_id='u1',face_id='ep1:f1'),dict(utterance_id='u3',face_id=None)]}
        snapshot=associate(b,Backend(),accept_links=True)
        refs={r['utterance_id']:r for r in snapshot['character_refs']}
        self.assertEqual(refs['u1']['voice_id'],refs['u3']['voice_id'])
        self.assertEqual(refs['u1']['character_id'],refs['u3']['character_id'])
        self.assertIsNone(refs['u3']['visible_face_id'])
        self.assertIsNone(refs['u2']['character_id'])
        self.assertEqual(refs['u2']['text'],'yes')
        self.assertIn('u2',snapshot['excluded_voice_turns'])

    def test_short_only_skips_inference_but_preserves_text(self):
        b=bundle();b['utterances']=b['utterances'][1:];b['observations']=b['observations'][:1]
        class Backend:
            def infer(self,*args): raise AssertionError('must not be called')
        result=associate(b,Backend())
        self.assertEqual(result['character_refs'][0]['text'],'yes')
        self.assertEqual(result['voice_groups'],[])

    def test_legacy_short_embedding_is_not_indexed(self):
        b=bundle();b['utterances'][0]['end']=2.9;b['observations'][1]['end']=86402.9
        prepared=prepare_association(b)
        self.assertNotIn('ep1:v1',prepared.graph.data['observations'])
        self.assertEqual(prepared.voice_ids,{})

    def test_voice_observation_timestamp_mismatch_rejected(self):
        b=bundle();b['observations'][1]['start']=86400.
        with self.assertRaises(ValueError): prepare_association(b)

    def test_global_link_requires_visible_support_and_consistency(self):
        p=prepare_association(repeated_bundle());vid=p.clip['voice_groups'][0]['voice_id']
        raw={'voice_links':[dict(voice_id=vid,face_id='ep1:f1')],
             'segments':[dict(utterance_id='u1',face_id=None),dict(utterance_id='u3',face_id=None)]}
        with self.assertRaises(ValueError): normalize_voice_prediction(raw,p)
        raw['segments'][0]['face_id']='ep1:f1';raw['voice_links'][0]['face_id']=None
        with self.assertRaises(ValueError): normalize_voice_prediction(raw,p)

    def test_missing_or_duplicate_voice_links_rejected(self):
        p=prepare_association(bundle());vid=p.clip['voice_groups'][0]['voice_id']
        for links in [[],[dict(voice_id=vid,face_id=None)]*2,[dict(voice_id='invented',face_id=None)]]:
            with self.assertRaises(ValueError):
                normalize_voice_prediction({'voice_links':links,'segments':[dict(utterance_id='u1',face_id=None)]},p)

    def test_prompt_contains_grouped_turns_not_local_speaker_labels(self):
        p=prepare_association(repeated_bundle())
        text=prompt(p.clip,'voice_association')
        self.assertIn('voice:ep1:v1',text)
        self.assertIn('again',text)
        self.assertNotIn('different-local-label',text)
        self.assertNotIn('"u2"',text)

    def test_gemini_adapter_uses_grouped_protocol(self):
        from lab.reused.backends import Backend
        import os
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'video.mp4';image=Path(d)/'face.jpg'
            video.write_bytes(b'test-video');image.write_bytes(b'test-face')
            b=repeated_bundle();b['video']=str(video);b['faces'][0]['image']=str(image)
            def request(url,payload,headers,timeout):
                text=payload['contents'][0]['parts'][0]['text']
                self.assertIn('Voice groups:',text)
                self.assertIn('again',text)
                response={'voice_links':[dict(voice_id='voice:ep1:v1',face_id='ep1:f1')],
                          'segments':[dict(utterance_id='u1',face_id='ep1:f1'),dict(utterance_id='u3',face_id=None)]}
                return {'candidates':[{'content':{'parts':[{'text':json.dumps(response)}]}}]}
            with patch.dict(os.environ,{'GEMINI_API_KEY':'test-only'}),patch('lab.reused.backends.request_json',request):
                result=associate(b,Backend({'backend':'gemini','model':'test-model'},Path(d)),accept_links=True)
            self.assertEqual(result['character_refs'][2]['character_id'],'person:ep1:f1')
