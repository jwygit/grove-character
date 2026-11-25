import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from lab.speech import infer


class AdapterContractTests(unittest.TestCase):
    def test_gemini_response_is_validated_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.wav'
            with wave.open(str(path),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0'*64000)
            def fake_request(url,payload,headers,timeout):
                self.assertEqual(payload['contents'][0]['parts'][1]['inline_data']['mime_type'],'audio/wav')
                return {'candidates':[{'content':{'parts':[{'text':json.dumps({'segments':[
                    {'start':.1,'end':1.8,'text':'hello'}]})}]}}]}
            with patch.dict(os.environ,{'GEMINI_API_KEY':'test-only'}),patch('lab.speech.request_json',fake_request):
                result=infer(path,{'backend':'gemini','model':'test-model'})
            self.assertEqual(result['segments'][0]['start'],.1)
            self.assertIsNone(result['segments'][0]['speaker'])

    def test_missing_key_does_not_fall_back_to_fixture(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):
                infer(Path('not-read.wav'),{'backend':'gemini','model':'test-model'})
