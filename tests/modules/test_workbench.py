import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from lab.workbench.app import create_app
from lab.workbench.runtime import validate_answer
from lab.workbench.store import Store
from tests.http_models import config, create_video


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root/'data')
        self.store.add_video(dict(id='v',path=str(self.root/'source.mp4'),has_audio=True))
        self.run = self.store.create_run('v','core',config('http://127.0.0.1:1'))['id']

    def test_artifact_authorization_immutability_and_symlink(self):
        path = self.store.directory(self.run)/'evidence.json'
        path.write_text('{}')
        item = self.store.artifact(self.run,path)
        self.assertEqual(item,self.store.artifact(self.run,path))
        with self.assertRaises(KeyError):
            self.store.artifact_path('another-run',item['id'])
        outside = self.root/'outside.txt'
        outside.write_text('private')
        link = path.parent/'link'
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.store.artifact(self.run,link)
        path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.store.artifact_path(self.run,item['id'])

    def test_cursor_dedup_and_resume_contract(self):
        self.store.event(self.run,'memory','committed',unique_key='v1')
        self.store.event(self.run,'memory','committed',unique_key='v1')
        events = self.store.events(self.run)
        self.assertEqual(len(events),2)
        self.assertEqual(self.store.events(self.run,events[0]['seq']),events[1:])
        with self.assertRaises(ValueError): self.store.resume(self.run)
        self.store.request_stop(self.run)
        self.store.state(self.run,'stopped')
        self.store.resume(self.run)
        self.assertEqual(self.store.run(self.run)['status'],'queued')
        self.assertFalse(self.store.run(self.run)['stop'])

    def test_history_uses_input_frontier_and_ready_not_evidence_time(self):
        with self.store.connect() as db:
            for version,frontier,coverage,ready in [(1,6,4,1),(2,10,8,0),(3,12,12,1)]:
                db.execute('INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?)',
                    (self.run,version,str(version),frontier,coverage,'/unused','hash',ready))
        self.assertIsNone(self.store.choose_snapshot(self.run,5))
        self.assertEqual(self.store.choose_snapshot(self.run,10)['version'],1)
        self.assertEqual(self.store.choose_snapshot(self.run,None)['version'],3)
        with self.assertRaises(ValueError): self.store.choose_snapshot(self.run,float('nan'))

    def test_citations_reject_fabricated_or_unmarked_claim(self):
        citations = [dict(citation_id='C1')]
        good = dict(answer='A fact [C1]',citation_ids=['C1'],insufficient_evidence=False)
        self.assertEqual(validate_answer(good,citations)['citations'],citations)
        for change in [dict(citation_ids=['C99']),dict(answer='No inline marker'),dict(citation_ids=[]),
                       dict(insufficient_evidence='false')]:
            with self.assertRaises(ValueError):validate_answer(good|change,citations)

    def test_upload_media_ranges_profile_validation_and_sse_resume(self):
        app = create_app(self.root/'api',{'core':dict(config=config('http://127.0.0.1:1'))},start_workers=False)
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.post('/api/videos',content=b'not video').status_code,400)
            video = self.root/'valid.mp4'
            create_video(video,1)
            response = client.post('/api/videos?filename=../../movie.mp4',content=video.read_bytes())
            self.assertEqual(response.status_code,200,response.text)
            uploaded=response.json()
            self.assertNotIn('path',uploaded)
            self.assertEqual(uploaded['name'],'movie.mp4')
            response=client.get(f'/api/videos/{uploaded["id"]}/media',headers={'Range':'bytes=0-31'})
            self.assertEqual(response.status_code,206)
            self.assertEqual(response.content,video.read_bytes()[:32])
            body=dict(video_id=uploaded['id'],profile='core')
            self.assertEqual(client.post('/api/runs',json=body|dict(config={})).status_code,422)
            run=client.post('/api/runs',json=body).json()
            rid=run['id']
            first=app.state.store.events(rid)[0]
            app.state.store.event(rid,'frames','completed')
            stream=client.get(f'/api/runs/{rid}/events?follow=false',headers={'Last-Event-ID':str(first['seq'])})
            self.assertNotIn('"queued"',stream.text)
            self.assertIn('"frames"',stream.text)
            self.assertEqual(client.post(f'/api/runs/{rid}/questions',json=dict(question='hello',playback_time=0)).json()['citations'],[])
            self.assertEqual(client.get(f'/api/runs/{rid}/artifacts/nope').status_code,404)

    def test_streamed_upload_limit_cleans_partial_file(self):
        app=create_app(self.root/'limited',{},start_workers=False,max_upload_bytes=10)
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/videos',content=b'x'*11).status_code,413)
        self.assertEqual(list((self.root/'limited'/'uploads').iterdir()),[])
