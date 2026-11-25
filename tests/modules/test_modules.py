import copy
import unittest
from lab.metrics import speech_metrics, attribution_metrics, boundary_counts
from lab.speech import words_to_turns
from lab.association import resolve
from lab.common import validate_turns
from lab.reused.core import score_clip, metrics
from tests.fixtures import bundle, predictions


class SpeechTests(unittest.TestCase):
    def test_cross_boundary_keeps_id_without_duplicating_full_transcript(self):
        from lab.perception import episode_turns
        speech={'segments':[dict(id='u1',start=1,end=3,text='hello world')]}
        a,b=episode_turns(speech,0,2,'s')[0],episode_turns(speech,2,4,'s')[0]
        self.assertEqual(a['id'],b['id'])
        self.assertEqual((a['start'],a['end'],b['start'],b['end']),(1,2,0,1))
        self.assertEqual(a['text'],'')
        speech['raw']={'words':[dict(start=1,end=1.5,text='hello'),dict(start=2,end=2.5,text='world')]}
        self.assertEqual(episode_turns(speech,2,4,'s')[0]['text'],'world')

    def test_change_inside_asr_sentence_and_overlap(self):
        words = [dict(start=i, end=i+1, text=t) for i,t in enumerate(['hello ', 'you ', 'both'])]
        diar = [dict(start=0,end=1,speaker='a'), dict(start=1,end=3,speaker='b'),
                dict(start=2,end=3,speaker='a')]
        result = words_to_turns(words, diar)
        self.assertEqual([r['speaker'] for r in result], ['a', 'b', None])
        self.assertTrue(result[2]['overlap'])

    def test_merged_turn_penalized(self):
        truth = [dict(id='a', start=0,end=1,text='hello',speaker='a'),
                 dict(id='b',start=1,end=2,text='you',speaker='b')]
        merged = [dict(id='c',start=0,end=2,text='hello you')]
        result = speech_metrics(truth, merged, 2)
        self.assertEqual(result['wer'], 0)
        self.assertEqual(result['segment_speaker_purity'], .5)
        self.assertLess(result['boundary_f1'], 1)

    def test_silence_and_missed_speech(self):
        result = speech_metrics([dict(id='a',start=0,end=1,text='hi',speaker='a')], [], 2)
        self.assertEqual(result['missed_speech_seconds'], 1)
        self.assertEqual(result['wer'], 1)
        self.assertIsNone(speech_metrics([], [], 2)['wer'])

    def test_chinese_cer(self):
        g = [dict(id='a',start=0,end=1,text='你好！')]
        p = [dict(id='a',start=0,end=1,text='你早')]
        self.assertEqual(speech_metrics(g,p,2)['cer'], .5)

    def test_boundary_one_to_one(self):
        self.assertEqual(boundary_counts([1.], [.9,1.,1.1], .25), (1,2,0))

    def test_invalid_times(self):
        with self.assertRaises(ValueError):
            validate_turns([dict(id='a',start=0,end=float('nan'),text='')], 10)


class IdentityTests(unittest.TestCase):
    def test_candidate_does_not_merge(self):
        r=resolve(bundle(), predictions())
        self.assertEqual(r['character_refs'][0]['character_id'], 'person:ep1:v1')
        self.assertEqual(next(iter(r['graph']['links'].values()))['status'], 'candidate')

    def test_confirmed_id_and_unknown(self):
        r=resolve(bundle(), predictions(), accept_links=True)
        self.assertEqual(r['character_refs'][0]['character_id'], 'person:ep1:f1')
        self.assertIsNone(r['character_refs'][1]['character_id'])
        self.assertEqual(r['character_refs'][1]['text'], 'yes')

    def test_offscreen_voice_reuses_prior_identity(self):
        prior=resolve(bundle(), predictions(), accept_links=True)
        b=bundle(); b.update(time_origin=86410., faces=[])
        b['utterances']=b['utterances'][:1]; b['utterances'][0].update(id='u3',voice_observation_id='ep2:v1')
        b['observations']=[dict(b['observations'][1],id='ep2:v1',start=86411.,end=86413.)]
        r=resolve(b,[dict(utterance_id='u3',face_id=None)],prior['graph'],
                  match_threshold={'face':.9,'voice':.9})
        self.assertEqual(r['character_refs'][0]['character_id'], 'person:ep1:f1')
        self.assertEqual(prior['character_refs'][0]['graph_revision'], prior['graph_revision'])

    def test_future_graph_and_invalid_face_rejected(self):
        b=bundle(); p=predictions(); p[0]['face_id']='fake'
        with self.assertRaises(ValueError): resolve(b,p)
        b['observations'][0]['end']=99999
        with self.assertRaises(ValueError): resolve(b,predictions())

    def test_missing_and_duplicate_predictions(self):
        for p in [predictions()[:1], predictions()+predictions()[:1]]:
            with self.assertRaises(ValueError): resolve(bundle(),p)

    def test_association_wrong_face_and_offscreen_penalty(self):
        g=[dict(start=0,end=1,face_id='f1'),dict(start=1,end=2,face_id=None)]
        p=[dict(start=0,end=2,face_id='f2')]
        r=metrics(score_clip(g,p,2,{'f1','f2'}))
        self.assertEqual(r['association_precision'],0)
        self.assertEqual(r['offscreen_false_link_rate'],1)

    def test_attribution_does_not_reward_abstaining(self):
        g=[dict(unit_id='a',role='speaker',character_id='person1')]
        p=[dict(unit_id='a',role='speaker',character_id=None)]
        self.assertEqual(attribution_metrics(g,p)['character_recall'],0)
        self.assertEqual(attribution_metrics(g,[])['missing_units'],1)

    def test_actor_is_not_speaker(self):
        g=[dict(unit_id='a',role='actor',character_id='person1')]
        p=[dict(unit_id='a',role='speaker',character_id='person1')]
        self.assertEqual(attribution_metrics(g,p)['character_recall'],0)
