import contextlib
import hashlib
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from lab.association import associate, prepare_association
from lab.equivalence import equivalence_metrics
from lab.m3 import build_messages, prepare_features, parse_memories, timestamp, M3Backend
from lab.reused.m3_prompt import M3_PROMPT
from tests.fixtures import bundle
from tests.modules.test_voices import repeated_bundle


class JointBackend:
    mode = 'm3_memorization'

    def infer(self, clip, mode):
        assert mode == self.mode
        self.clip = clip
        return {'video_description': ['<face_0> speaks: hello.', '<voice_0> speaks again.'],
                'high_level_conclusions': ['Equivalence: <face_0>, <voice_0>']}


def snapshot(accept=True):
    return associate(repeated_bundle(), JointBackend(), accept_links=accept)


class M3Tests(unittest.TestCase):
    def setUp(self):
        self.prepared = prepare_association(repeated_bundle())
        self.registry = prepare_features(self.prepared)

    def test_joint_input_uses_grouped_voice_and_exact_prompt(self):
        backend = JointBackend()
        result = associate(repeated_bundle(), backend, accept_links=True)
        backend.clip['video'] = '/local/episode.mp4'
        content = build_messages(backend.clip, {})[0]['content']
        self.assertEqual(content[0]['text'], M3_PROMPT)
        self.assertEqual(content[1], {'type':'video', 'video': backend.clip['video']})
        self.assertEqual(content[3]['text'], '<face_0>:')
        voices = json.loads(content[-1]['text'])
        self.assertEqual(list(voices), ['<voice_0>'])
        self.assertEqual([u['asr'] for u in voices['<voice_0>']], ['hello', 'again'])
        self.assertEqual(voices['<voice_0>'][0]['start_time'], '00:01.000')
        refs = {r['utterance_id']: r for r in result['character_refs']}
        self.assertEqual(refs['u1']['character_id'], refs['u3']['character_id'])
        self.assertIsNone(refs['u2']['character_id'])
        self.assertNotIn('associations', result)
        self.assertEqual(refs['u1']['visibility'], 'not_predicted')

    def test_prompt_matches_recorded_source_hash(self):
        lock = json.loads((Path(__file__).resolve().parents[2]/'sources.lock.json').read_text())
        self.assertEqual(hashlib.sha256(M3_PROMPT.encode()).hexdigest(),
                         lock['m3_agent']['memorization_prompt_sha256'])

    def test_singular_plural_mismatch_is_handled_explicitly(self):
        self.assertEqual(timestamp(59.99999), '01:00.000')
        output = {'video_descriptions':['<face_0> smiles.'], 'high_level_conclusions':[]}
        memories, pairs, keys = parse_memories(output, self.registry)
        self.assertEqual(keys, ['video_descriptions'])
        self.assertEqual(memories[0]['feature_ids'], ['<face_0>'])
        self.assertEqual(pairs, [])
        output['video_description'] = ['different']
        with self.assertRaisesRegex(ValueError, 'conflicting singular/plural'):
            parse_memories(output, self.registry)

    def test_unknown_and_malformed_output_rejected(self):
        invalid = [
            {'video_description':['<face_99> smiles.'], 'high_level_conclusions':[]},
            {'video_description':[], 'high_level_conclusions':['Equivalence: <face_0>, <voice_0> maybe']},
            {'video_description':'not a list', 'high_level_conclusions':[]},
            {'high_level_conclusions':[]},
            {'video_description':['<person:invented> smiles.'], 'high_level_conclusions':[]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_memories(payload, self.registry)

    def test_conflicting_equivalences_rejected(self):
        registry = dict(self.registry, **{'<face_1>': self.registry['<face_0>']})
        payload = {'video_description':[], 'high_level_conclusions':[
            'Equivalence: <face_0>, <voice_0>', 'Equivalence: <face_1>, <voice_0>']}
        with self.assertRaisesRegex(ValueError, 'conflicting faces'):
            parse_memories(payload, registry)

    def test_candidates_do_not_rewrite_two_features_as_same_character(self):
        pending, accepted = snapshot(False), snapshot(True)
        for result, equal in [(pending, False), (accepted, True)]:
            mapping = result['identity_evidence']['feature_mapping']
            self.assertEqual(mapping['<face_0>']['character_id'] ==
                             mapping['<voice_0>']['character_id'], equal)
            self.assertEqual(len(result['identity_evidence']['memories']), 2)
            self.assertEqual(result['equivalences'], [{'face':'<face_0>', 'voice':'<voice_0>'}])

    def test_visual_only_episode_still_calls_model(self):
        data = bundle(); data['utterances'] = []
        data['observations'] = [o for o in data['observations'] if o['modality'] == 'face']
        class VisualBackend:
            mode = 'm3_memorization'
            def infer(self, clip, mode):
                self.called = True
                return {'video_description':['<face_0> waves.'], 'high_level_conclusions':[]}
        backend = VisualBackend()
        result = associate(data, backend)
        self.assertTrue(backend.called)
        self.assertEqual(result['character_refs'], [])
        self.assertEqual(len(result['identity_evidence']['memories']), 1)

    def test_registry_and_graph_are_idempotent_on_replay(self):
        first = snapshot()
        second = associate(repeated_bundle(), JointBackend(), first['graph'], accept_links=True)
        self.assertEqual(first['feature_registry'], second['feature_registry'])
        self.assertEqual(first['graph_revision'], second['graph_revision'])
        self.assertEqual(first['identity_evidence'], second['identity_evidence'])

    def test_identity_score_separates_model_prediction_from_accepted_graph(self):
        truth = {'pairs':[dict(face='<face_0>', voice='<voice_0>', same_person=True)]}
        score = equivalence_metrics(truth, snapshot(False))
        self.assertEqual(score['model_equivalences']['recall'], 1)
        self.assertEqual(score['resolved_character_equivalences']['recall'], 0)
        with self.assertRaisesRegex(ValueError, 'every supplied'):
            equivalence_metrics({'pairs':[]}, snapshot())

    def test_missing_checkpoint_fails_without_download_or_fake_prediction(self):
        backend = M3Backend({'model_path':'/no-such-local-m3-checkpoint'}, '.')
        with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
            backend.infer(self.prepared.clip, backend.mode)

    def test_gpu_adapter_preserves_audio_for_both_processor_interfaces(self):
        class Inputs(dict):
            input_ids = SimpleNamespace(size=lambda axis: 3)
            def to(self, value): return self
        class Tensor:
            def __getitem__(self, index): return self
        generation_calls = []
        class Model:
            device, dtype = 'cuda:0', 'bf16'
            def parameters(self): return [SimpleNamespace(device=SimpleNamespace(index=0, type='cuda'))]
            def generate(self, **kwargs):
                generation_calls.append(kwargs)
                return Tensor()
        class Processor:
            def apply_chat_template(self, messages, **kwargs): return 'template with video/audio'
            def batch_decode(self, value, **kwargs):
                return ['{"video_description":[],"high_level_conclusions":[]}']
        received = []
        class Plural(Processor):
            def __call__(self, text=None, audios=None, **kwargs):
                received.append(audios)
                return Inputs()
        class Singular(Processor):
            def __call__(self, text=None, audio=None, **kwargs):
                received.append(audio)
                return Inputs()
        cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda d: None,
                               max_memory_allocated=lambda d: 123)
        modules = {'torch':SimpleNamespace(cuda=cuda, no_grad=contextlib.nullcontext),
                   'transformers':SimpleNamespace(Qwen2_5OmniProcessor=Processor,
                       Qwen2_5OmniThinkerForConditionalGeneration=Model, GenerationConfig=lambda **kw: kw),
                   'qwen_omni_utils':SimpleNamespace(process_mm_info=lambda *a, **kw: ('soundtrack', [], []))}
        self.prepared.clip['video'] = '/local/episode.mp4'
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            for processor, key in [(Plural(), 'audios'), (Singular(), 'audio')]:
                backend = M3Backend(dict(model_path=directory, model_revision='fixture-revision',
                                         model_id='fixture-model'), '.')
                backend.model, backend.processor = Model(), processor
                result = backend.infer(self.prepared.clip, backend.mode)
                self.assertEqual(result['inference']['audio_parameter'], key)
                self.assertEqual(result['inference']['peak_allocated_bytes'], {'0':123})
        self.assertEqual(received, ['soundtrack', 'soundtrack'])
        self.assertTrue(all(call['use_audio_in_video'] for call in generation_calls))
        self.assertTrue(all(call['do_sample'] and call['temperature'] == 1e-6 for call in generation_calls))
