"""Real segmenter, prompts and retry utility; only model generation is synthetic."""
import unittest
from lab.grove import load_grove
from tests.fixtures import episode, bundle, predictions
from lab.association import resolve
load_grove()
from hypervideo.types import FrameCaption
from hypervideo.modules.character_identity import SnapshotProvider


class BoundaryHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_finalize_waits_for_identity_before_summary(self):
        await self.check_finalize(False)

    async def test_joint_memories_enter_actual_finalize_and_summary_is_rewritten(self):
        await self.check_finalize(True)

    async def check_finalize(self, joint):
        seen=[]
        import json
        from hypervideo.extractors.episode_segmenter import VideoEpisodeSegmenter
        async def llm(prompt):
            seen.append('summary')
            self.assertIn('person:ep1:f1',prompt)
            if joint:
                self.assertIn('<face_0> speaks: hello.',prompt)
            return json.dumps(dict(summary='<face_0> says hello' if joint else 'hello',subject='chat'))
        segmenter=VideoEpisodeSegmenter(llm)
        if joint:
            from tests.modules.test_m3 import snapshot as joint_snapshot
            snapshot=joint_snapshot()
        else:
            snapshot=resolve(bundle(),predictions(),accept_links=True)
        async def provider(a,b):
            seen.append('identity')
            if joint:
                return await SnapshotProvider([snapshot],'recording').episode_evidence(a,b)
            return await SnapshotProvider([snapshot],'recording')(a,b)
        segmenter.identity_provider=provider
        segmenter._buffer=[FrameCaption.from_dict(fc) for fc in episode()['frame_captions']]
        result=await segmenter._finalize_episode()
        self.assertEqual(seen,['identity','summary'])
        self.assertEqual((result.start_seconds,result.end_seconds),(86400,86410))
        self.assertEqual(result.character_refs[0]['character_id'],'person:ep1:f1')
        if joint:
            self.assertEqual(result.summary,'<person:ep1:f1> says hello')
            self.assertEqual(result.identity_evidence['memories'][0]['character_text'],
                             '<person:ep1:f1> speaks: hello.')
        self.assertEqual(segmenter._pending_episode_id,result.episode_id)
