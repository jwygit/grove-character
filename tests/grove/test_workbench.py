"""Actual uploaded media, worker process, Grove graph/indices and HTTP contracts.

Only model predictions are synthetic. These tests establish wiring and recovery,
not ASR, recognition, retrieval relevance or answer accuracy on real programs.
"""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import threading
import unittest
from fastapi.testclient import TestClient
from lab.common import audio, digest, write
from lab.full_loop import run_video
from lab.grove import load_grove
from lab.stream_identity import identity_provider_factory
from lab.workbench.app import create_app
from lab.workbench.runtime import Observer, ask
from lab.workbench.store import Store
from tests.http_models import config, create_video, model_server
from tests.grove.test_full_loop_characters import InferenceFixture


class WorkbenchGroveTests(unittest.TestCase):
    def test_queued_stop_then_resume_uses_real_worker(self):
        from lab.workbench.worker import execute
        with tempfile.TemporaryDirectory() as directory, model_server() as (models,base):
            root=Path(directory);video=root/'test.mp4';create_video(video)
            app=create_app(root/'service',{'core':dict(config=config(base))},start_workers=False)
            with TestClient(app) as client:
                vid=client.post('/api/videos',content=video.read_bytes()).json()['id']
                rid=client.post('/api/runs',json=dict(video_id=vid,profile='core')).json()['id']
                self.assertEqual(client.post(f'/api/runs/{rid}/stop').status_code,200)
                execute(app.state.store,rid)
                self.assertEqual(app.state.store.run(rid)['status'],'stopped')
                self.assertEqual(models.calls['caption_frames'],0)
                self.assertEqual(client.post(f'/api/runs/{rid}/resume').status_code,200)
                execute(app.state.store,rid)
                self.assertEqual(app.state.store.run(rid)['status'],'complete')
                self.assertEqual(len(app.state.store.snapshots(rid)),3)

    def test_upload_real_worker_events_history_and_cited_answer(self):
        with tempfile.TemporaryDirectory() as directory, model_server() as (models,base):
            root=Path(directory)
            video=root/'test.mp4'
            create_video(video)
            release = threading.Event()
            original = models.respond
            def gated(path, payload):
                if payload and 'caption_frames' in json.dumps(payload) and models.calls['caption_frames']==4:
                    release.wait(15)
                return original(path,payload)
            models.respond = gated
            self.addCleanup(release.set)
            app=create_app(root/'service',{'core':dict(config=config(base))})
            with TestClient(app) as client:
                vid=client.post('/api/videos',content=video.read_bytes()).json()['id']
                rid=client.post('/api/runs',json=dict(video_id=vid,profile='core')).json()['id']
                end=time.monotonic()+70
                observed_while_running=False
                queried_while_running=False
                while time.monotonic()<end:
                    run=client.get(f'/api/runs/{rid}').json()
                    if run['status']=='running' and app.state.store.events(rid,limit=1000):
                        observed_while_running=True
                        if run['snapshots'] and run['snapshots'][0]['ready'] and not queried_while_running:
                            live=client.get(f'/api/runs/{rid}/events?follow=false').text
                            self.assertIn('"ready"',live)
                            response=client.post(f'/api/runs/{rid}/questions',json=dict(question='red cup',playback_time=6))
                            self.assertEqual(response.status_code,200,response.text)
                            self.assertEqual(response.json()['memory_version'],1)
                            queried_while_running=True
                            release.set()
                    if run['status'] in ('failed','complete','interrupted'):break
                    time.sleep(.1)
                log=(app.state.store.directory(rid)/'worker.log').read_text()
                self.assertEqual(run['status'],'complete',log)
                self.assertTrue(observed_while_running)
                self.assertTrue(queried_while_running)
                snapshots=app.state.store.snapshots(rid)
                self.assertEqual([(s['frontier'],s['coverage']) for s in snapshots],[(6,4),(10,8),(12,12)])
                events=client.get(f'/api/runs/{rid}/events?follow=false').text
                for stage in ('boundary_stream','episode_summary','fact_extract','topic_create','topic_update','committed','ready'):
                    self.assertIn('"'+stage+'"',events)
                self.assertIsNone(client.get(f'/api/runs/{rid}/snapshot?playback_time=5').json()['version'])
                early=client.get(f'/api/runs/{rid}/snapshot?playback_time=6').json()
                self.assertEqual(len(early['graph']['episodes']),1)
                early_topic=json.dumps(early['graph']['topics'])
                self.assertNotIn('green',early_topic)
                late=client.get(f'/api/runs/{rid}/snapshot').json()
                self.assertEqual(len(late['graph']['episodes']),3)
                answer=client.post(f'/api/runs/{rid}/questions',json=dict(question='red cup',playback_time=6)).json()
                self.assertEqual(answer['memory_version'],1,answer)
                self.assertTrue(answer['citations'])
                self.assertTrue(all(c['spans'] for c in answer['citations']))
                self.assertEqual({e['node_type'] for e in answer['retrieval']['evidence']},{'fact','visual_clip'})
                self.assertEqual(client.get(f'/api/runs/{rid}/questions').json()[-1],answer)
                artifact=next(iter(late['frames'].values()))
                self.assertEqual(client.get(artifact['url']).status_code,200)
                # Error responses do not append a fabricated replacement answer.
                models.invalid.add('evidence_answer')
                self.assertEqual(client.post(f'/api/runs/{rid}/questions',json=dict(question='cup')).status_code,502)
                self.assertEqual(len(client.get(f'/api/runs/{rid}/questions').json()),2)

    def test_failed_window_is_not_published_and_resume_is_idempotent(self):
        load_grove()
        with tempfile.TemporaryDirectory() as directory, model_server() as (models,base):
            root=Path(directory); video=root/'video.mp4'; create_video(video)
            store=Store(root/'service');store.add_video(dict(id='video',path=str(video)))
            cfg=config(base); rid=store.create_run('video','core',cfg)['id'];out=store.directory(rid)/'pipeline'
            observer=Observer(store,rid)
            models.invalid.add('episode_summary')
            with self.assertRaises(Exception):asyncio.run(run_video(video,cfg,out,'video',observer=observer))
            self.assertEqual(store.snapshots(rid),[])
            self.assertTrue(any(e['state']=='failed' for e in store.events(rid,limit=1000)))
            models.invalid.clear()
            memory,_=asyncio.run(run_video(video,cfg,out,'video',observer=observer))
            count=len(store.snapshots(rid))
            asyncio.run(run_video(video,cfg,out,'video',observer=observer))
            self.assertEqual(len(store.snapshots(rid)),count)
            committed=[e for e in store.events(rid,limit=1000) if e['state']=='committed']
            self.assertEqual(len(committed),count)
            # Simulate crash after checkpoint publication but before index-ready.
            last=store.snapshots(rid)[-1]
            with store.connect() as db:db.execute('UPDATE snapshots SET ready=0 WHERE run_id=? AND version=?',(rid,last['version']))
            self.assertEqual(store.choose_snapshot(rid)['version'],count-1)
            asyncio.run(observer.committed(memory,12))
            self.assertEqual(store.choose_snapshot(rid)['version'],count)

    def test_character_snapshot_spoken_timing_and_citations(self):
        load_grove()
        with tempfile.TemporaryDirectory() as directory, model_server() as (models,base):
            root=Path(directory);video=root/'video.mp4';create_video(video)
            wav=root/'audio.wav';audio(video,wav);speech=root/'speech.json'
            write(speech,dict(audio_sha256=digest(wav),segments=[dict(id=f'u{i}',start=i*4+.5,end=i*4+3,text='hello') for i in range(3)],raw={}))
            cfg=config(base)
            cfg['boundary_use_transcript']=True
            cfg['characters']=dict(speech_input=str(speech),perception={},association={},accept_links=True,
                grounding=dict(endpoint=base+'/v1/chat/completions',model='synthetic-grounding',model_revision='fixture-v1',sample_fps=1,max_frames=20))
            app=create_app(root/'service',{'characters':dict(config=cfg)},start_workers=False)
            with TestClient(app) as client:
                vid=client.post('/api/videos',content=video.read_bytes()).json()['id']
                rid=client.post('/api/runs',json=dict(video_id=vid,profile='characters')).json()['id']
                store=app.state.store;out=store.directory(rid)/'pipeline';out.mkdir()
                observer=Observer(store,rid);inference=InferenceFixture()
                factory=identity_provider_factory(video,out,cfg['characters'],vid,86400,
                    perception_extract=inference.perceive,association_backend=inference,observer=observer)
                memory,_=asyncio.run(run_video(video,cfg,out,vid,observer=observer,identity_factory=factory))
                response=client.get(f'/api/runs/{rid}/snapshot')
                self.assertEqual(response.status_code,200,response.text)
                snapshot=response.json()
                self.assertEqual(len(snapshot['people']),3)
                self.assertTrue(snapshot['people'][0]['faces'])
                self.assertEqual(client.get(snapshot['people'][0]['faces'][0]['media']['url']).status_code,200)
                speakers={ref['character_id'] for f in memory.graph.facts.values() for ref in f.character_refs if ref['role']=='speaker'}
                self.assertEqual(len(speakers),1)
                answer=asyncio.run(ask(store,rid,'hello',None,list(speakers)))
                spoken=[c for c in answer['retrieval']['evidence'] if c['node_type']=='fact' and 'says' in c['content']]
                self.assertTrue(spoken)
                self.assertEqual(spoken[0]['evidence_refs'][0]['start']%4,.5)
                self.assertEqual(inference.scopes,[(0,4),(4,8),(8,12)])
                stages={e['stage'] for e in store.events(rid,limit=1000)}
                self.assertTrue({'speech','face_voice','association','grounding','identity','character_facts'}<=stages)
