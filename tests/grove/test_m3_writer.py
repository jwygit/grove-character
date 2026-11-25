import copy
import json
from pathlib import Path
import tempfile
import unittest
from lab.grove import hide_identity, load_grove, write_episode
from tests.fixtures import episode
from tests.modules.test_m3 import snapshot
load_grove()
from hypervideo.types import VideoEpisode
from hypervideo.structure import VideoFactNode, VideoEpisodeNode
from hypervideo.modules.character_identity import (SnapshotProvider, resolve_episode_identity,
    facts_from_response, rewrite_text)


class M3WriterTests(unittest.IsolatedAsyncioTestCase):
    async def make_episode(self, value=None):
        ep = VideoEpisode.from_dict(episode())
        value = value or snapshot()
        ep.character_refs, ep.identity_evidence = await resolve_episode_identity(
            SnapshotProvider([value], 'recording'), ep.start_seconds, ep.end_seconds)
        return ep

    async def test_text_and_refs_share_mapping_through_actual_graph_save_reload(self):
        prompts = []
        async def llm(prompt):
            prompts.append(prompt)
            self.assertIn('<face_0> speaks: hello.', prompt)
            if '"facts"' in prompt:
                return json.dumps({'facts':[dict(content='<face_0> speaks: hello.',
                    utterance_ids=['u1'], memory_ids=['description:0'])]})
            return json.dumps(dict(summary='<face_0> and <voice_0> are the same speaker.',
                                   subject='<face_0> speaks'))
        with tempfile.TemporaryDirectory() as directory:
            graph = await write_episode(episode(), snapshot(), llm, Path(directory)/'graph.json')
        fact = next(iter(graph['facts'].values()))
        ep = next(iter(graph['episodes'].values()))
        self.assertEqual(fact['content'], '<person:ep1:f1> speaks: hello.')
        self.assertTrue(all(r['character_id'] == 'person:ep1:f1' for r in fact['character_refs']))
        self.assertEqual({r['role'] for r in fact['character_refs']}, {'speaker', 'referenced'})
        self.assertNotIn('<face_', ep['summary'])
        self.assertIn('<face_0>', ep['identity_evidence']['writer_summary_raw']['summary'])
        self.assertEqual(fact['identity_evidence']['memories'][0]['text'], '<face_0> speaks: hello.')
        self.assertEqual(len(prompts), 2)
        self.assertEqual(VideoFactNode(**fact).to_video_fact().identity_evidence, fact['identity_evidence'])
        self.assertEqual(VideoEpisodeNode(**ep).to_video_episode().identity_evidence, ep['identity_evidence'])

    async def test_unknown_control_masks_mapping_and_regenerates_text(self):
        value = snapshot()
        ep = await self.make_episode(value)
        value['identity_evidence'] = ep.identity_evidence  # includes previously rendered character_text
        unknown = await self.make_episode(hide_identity(value))
        self.assertIsNotNone(ep.character_refs[0]['character_id'])  # original immutable
        self.assertTrue(all(r['character_id'] is None for r in unknown.character_refs))
        self.assertTrue(all(r['character_id'] is None for r in unknown.identity_evidence['feature_mapping'].values()))
        self.assertEqual(unknown.identity_evidence['memories'][0]['character_text'], '<unknown> speaks: hello.')
        fact = facts_from_response({'facts':[dict(content='<voice_0> speaks.', utterance_ids=['u1'])]}, unknown)[0]
        self.assertEqual(fact.content, '<unknown> speaks.')
        self.assertTrue(all(r['character_id'] is None for r in fact.character_refs))

    async def test_unknown_tag_and_uncited_known_tag_rejected(self):
        ep = await self.make_episode()
        invalid = [dict(content='<face_99> waves.', memory_ids=['description:0']),
                   dict(content='<face_0> waves.', memory_ids=[]),
                   dict(content='<person:invented> waves.', memory_ids=['description:0']),
                   dict(content='waves', memory_ids=['invented'])]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                facts_from_response({'facts':[row]}, ep)

    async def test_reference_is_not_automatically_an_actor(self):
        ep = await self.make_episode()
        fact = facts_from_response({'facts':[dict(content='<face_0> waves.',
                memory_ids=['description:0'])]}, ep)[0]
        self.assertEqual([r['role'] for r in fact.character_refs], ['referenced'])

    async def test_direct_character_token_can_cite_one_equivalent_feature(self):
        ep = await self.make_episode()
        fact = facts_from_response({'facts':[dict(content='<person:ep1:f1> speaks.',
                utterance_ids=['u1'])]}, ep)[0]
        self.assertEqual(fact.content, '<person:ep1:f1> speaks.')
        self.assertEqual(len(fact.character_refs), 2)

    async def test_mapping_revision_and_speaker_mismatch_fail(self):
        for key, value in [('graph_revision', -1), ('character_id', 'person:wrong')]:
            data = snapshot()
            data['identity_evidence']['feature_mapping']['<voice_0>'][key] = value
            with self.assertRaises(ValueError):
                await self.make_episode(data)

    async def test_whole_tokens_avoid_prefix_collision(self):
        ep = await self.make_episode()
        evidence = copy.deepcopy(ep.identity_evidence)
        evidence['feature_mapping']['<face_10>'] = dict(evidence['feature_mapping']['<face_0>'],
                                                      character_id='person:ten')
        self.assertEqual(rewrite_text('<face_0> meets <face_10>.', evidence),
                         '<person:ep1:f1> meets <person:ten>.')
