import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from lab.visual_grounding import sample_frames,run_grounding
from lab.common import digest
from tests.character_fixtures import timeline_snapshot


class VisualGroundingAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.video=cls.root/'synthetic.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=4:duration=10',
            '-c:v','libx264','-pix_fmt','yuv420p',str(cls.video)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_real_frame_decode_records_actual_pts_and_rejects_excessive_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            frames=sample_frames(self.video,10,{'sample_fps':.5,'max_frames':5},Path(directory))
            self.assertEqual([f['time'] for f in frames],[0,2,4,6,8])
            self.assertTrue(all(f['timestamp_basis']=='decoded_pts' and Path(f['path']).is_file() for f in frames))
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'max_frames'):
            sample_frames(self.video,10,{'sample_fps':2,'max_frames':5},Path(directory))

    def test_api_receives_frames_references_and_ids_and_returns_bound_localization(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            # A synthetic image supplies the reference; no face model is simulated as having run.
            refdir=out/'reference'; refdir.mkdir()
            ref=sample_frames(self.video,10,{'sample_fps':.1},refdir)[0]['path']
            snapshot=timeline_snapshot()
            snapshot['feature_registry']['<face_0>']['image']=ref
            config=dict(endpoint='http://127.0.0.1:8000/v1/chat/completions',model='fixture-vlm',
                model_revision='fixture-only',sample_fps=.5,max_frames=5)
            bundle=dict(id='ep1',source='recording',time_origin=86400,duration=10,video=str(self.video),episode_video_sha256=digest(self.video))
            payload={'locations':[
                dict(memory_id='description:0',intervals=[dict(start=4,end=6)],frame_ids=['frame:2'],reason='synthetic'),
                dict(memory_id='description:1',intervals=[],frame_ids=[],reason='not visible')]}
            def api(endpoint,body,headers,timeout):
                content=body['messages'][0]['content']
                self.assertEqual(sum(c['type']=='image_url' for c in content),6)
                self.assertIn('frame:2 at 4.000000',json.dumps(content))
                self.assertIn('description:0',content[-1]['text'])
                self.assertIn(['<face_0>','<voice_0>'],json.loads(content[-1]['text'])['confirmed_identity_groups'])
                return {'choices':[{'message':{'content':json.dumps(payload)}}]}
            with patch('lab.visual_grounding.request_json',side_effect=api):
                result=run_grounding(bundle,snapshot,config,out)
            memory=result['identity_evidence']['memories'][0]
            self.assertEqual(memory['grounding']['anchors'][0]['start'],86404)
            self.assertNotIn('grounding',snapshot['identity_evidence']['memories'][0])
            self.assertEqual(result['identity_state'],snapshot['identity_state'])

    def test_failed_endpoint_and_mismatched_source_do_not_produce_success(self):
        snapshot=timeline_snapshot()
        config=dict(endpoint='http://localhost',model='fixture',model_revision='fixture')
        bundle=dict(id='ep1',source='wrong',time_origin=86400,duration=10,video=str(self.video),episode_video_sha256=digest(self.video))
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'scope mismatch'):
            run_grounding(bundle,snapshot,config,Path(directory))

        bundle['source']='recording'
        with tempfile.TemporaryDirectory() as directory,patch('lab.visual_grounding.sample_frames',return_value=[]), \
             patch('lab.visual_grounding.build_content',return_value=[]), \
             patch('lab.visual_grounding.request_json',side_effect=RuntimeError('endpoint failed')),self.assertRaises(RuntimeError):
            run_grounding(bundle,snapshot,config,Path(directory))

    def test_changed_video_is_rejected_before_sending_to_model(self):
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'video hash'):
            run_grounding(dict(id='ep1',source='recording',time_origin=86400,duration=10,
                video=str(self.video),episode_video_sha256='wrong'),timeline_snapshot(),{},Path(directory))
