"""Local/intranet workbench. Profiles are server-owned, never uploaded code/config."""
import asyncio
from contextlib import asynccontextmanager
import copy
import fcntl
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from lab.common import digest
from lab.full_loop import validate_model
from .store import Store, dumps
from .runtime import ask

TERMINAL = {'complete','failed','stopped','interrupted'}


def is_locked(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


async def supervise(store):
    process = None
    active_id = None
    while True:
        if process is not None and process.poll() is not None:
            if store.run(active_id)['status'] in ('queued','running') and not is_locked(store.directory(active_id)/'worker.lock'):
                store.state(active_id, 'interrupted', f'worker exited with code {process.returncode}; see worker.log')
            process = None
        runs = store.runs()
        for run in runs:
            if run['status'] == 'running' and not is_locked(store.directory(run['id'])/'worker.lock'):
                # A process may have exited without writing its terminal status.
                store.state(run['id'], 'interrupted', 'worker exited; resume from checkpoint')
        if process is None and not is_locked(store.root/'compute.lock'):
            queued = sorted((r for r in store.runs() if r['status']=='queued'), key=lambda r:r['created'])
            if queued:
                run = queued[0]
                active_id = run['id']
                try:
                    with (store.directory(run['id'])/'worker.log').open('ab') as log:
                        process = subprocess.Popen([sys.executable,'-m','lab.workbench.worker',
                            '--data',str(store.root),'--run',run['id']], stdout=log,stderr=log,
                            cwd=str(Path(__file__).resolve().parents[2]))
                except OSError as exc:
                    store.state(run['id'], 'failed', f'cannot start worker: {exc}')
        await asyncio.sleep(.5)


class RunInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    video_id: str
    profile: str


class QuestionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1,max_length=4000)
    playback_time: float | None = Field(default=None,ge=0,allow_inf_nan=False)
    characters: list[str] = Field(default_factory=list,max_length=20)


def create_app(data_dir, profiles, *, speech_python=None, start_workers=True,
               max_upload_bytes=4*1024**3):
    store = Store(data_dir)
    # Resolve and validate server-owned profile files once; each run gets a copy.
    prepared = {}
    for name, entry in profiles.items():
        entry = {'config':entry} if isinstance(entry,str) else entry
        config = json.loads(Path(entry['config']).read_text()) if isinstance(entry['config'],str) else copy.deepcopy(entry['config'])
        for field in ('caption','writer','embedding'):
            validate_model(config[field])
        speech = entry.get('speech_config')
        if isinstance(speech,str):
            speech = json.loads(Path(speech).read_text())
        prepared[name] = dict(config=config,speech=speech,description=entry.get('description',name))

    @asynccontextmanager
    async def lifespan(app):
        lock = (store.root/'server.lock').open('a')
        task = None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if start_workers:
                task = asyncio.create_task(supervise(store))
            yield
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            lock.close()
            # Running child workers retain their locks and may finish after a
            # web restart. They are not killed halfway through a GPU operation.

    app = FastAPI(title='Grove 视频记忆工作台', lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=409)

    @app.get('/api/profiles')
    def get_profiles():
        return [dict(id=name,description=row['description'],characters=bool(row['config'].get('characters')),
                     speech=bool(row['speech'] or row['config'].get('characters')),
                     transcript_boundary=bool(row['config'].get('boundary_use_transcript'))) for name,row in prepared.items()]

    @app.post('/api/videos')
    async def upload(request: Request, filename: str='video.mp4'):
        # Stream raw bytes, limiting disk use before body parsing/spooling.
        vid = uuid4().hex
        directory = store.root/'uploads'/vid
        directory.mkdir(parents=True)
        path = directory/'video.mp4'
        size = 0
        sha = hashlib.sha256()
        try:
            with path.open('wb') as stream:
                async for block in request.stream():
                    size += len(block)
                    if size > max_upload_bytes:
                        raise HTTPException(413,'video exceeds configured upload limit')
                    sha.update(block)
                    await asyncio.to_thread(stream.write, block)
            command = ['ffprobe','-v','error','-show_entries','format=duration,format_name:stream=codec_type,codec_name','-of','json',str(path)]
            probe = json.loads(await asyncio.to_thread(subprocess.check_output,command,stderr=subprocess.PIPE,timeout=30))
            duration = float(probe['format']['duration'])
            streams = probe['streams']
            if not math.isfinite(duration) or duration<=0 or not any(s['codec_type']=='video' for s in streams):
                raise ValueError('upload must contain a finite, nonempty video stream')
            if 'mp4' not in probe['format']['format_name'] or any(
                s['codec_name'] not in ('h264','aac','mp3') for s in streams if s['codec_type'] in ('video','audio')):
                raise HTTPException(415,'Use MP4 with H.264 video and AAC/MP3 audio; see deployment conversion command')
            record = dict(id=vid,name=Path(filename).name[:255],path=str(path),sha256=sha.hexdigest(),
                          bytes=size,duration=duration,has_audio=any(s['codec_type']=='audio' for s in streams))
            store.add_video(record)
            return {k:v for k,v in record.items() if k!='path'}
        except BaseException as exc:
            path.unlink(missing_ok=True)
            directory.rmdir()
            if isinstance(exc, (subprocess.SubprocessError, KeyError, ValueError)):
                raise HTTPException(400,'file cannot be decoded as a supported video') from exc
            raise

    @app.get('/api/videos/{video_id}/media')
    def media(video_id: str):
        return FileResponse(store.video(video_id)['path'], media_type='video/mp4')

    @app.post('/api/runs')
    def run(body: RunInput):
        if body.profile not in prepared:
            raise HTTPException(400,'unknown server profile')
        profile = prepared[body.profile]
        video = store.video(body.video_id)
        if (profile['config'].get('characters') or profile['speech']) and not video['has_audio']:
            raise HTTPException(400,'this profile requires an audio track')
        return store.create_run(body.video_id,body.profile,copy.deepcopy(profile['config']),
                                profile['speech'],speech_python)

    @app.get('/api/runs')
    def runs():
        return store.runs()

    @app.get('/api/runs/{run_id}')
    def run_state(run_id: str):
        row = store.run(run_id)
        video = store.video(row['video_id'])
        row['video'] = {k:v for k,v in video.items() if k!='path'}
        row['snapshots'] = [{k:v for k,v in s.items() if k not in ('path','checkpoint_hash')} for s in store.snapshots(run_id)]
        saved_config = json.loads((store.directory(run_id)/'config.json').read_text())
        row['offline_audio_context'] = bool(saved_config.get('speech') or saved_config.get('characters'))
        return row

    @app.post('/api/runs/{run_id}/stop')
    def stop(run_id: str):
        row = store.run(run_id)
        if row['status'] not in TERMINAL:
            store.request_stop(run_id)
        return store.run(run_id)

    @app.post('/api/runs/{run_id}/resume')
    def resume(run_id: str):
        store.resume(run_id)
        return store.run(run_id)

    @app.get('/api/runs/{run_id}/events')
    async def events(request: Request, run_id: str, after: int=0, follow: bool=True):
        store.run(run_id)
        try:
            cursor = max(0,after,int(request.headers.get('last-event-id','0')))
        except ValueError:
            raise HTTPException(400,'invalid event cursor')
        async def stream():
            nonlocal cursor
            yield 'retry: 1000\n\n'
            while not await request.is_disconnected():
                batch = store.events(run_id,cursor)
                for row in batch:
                    cursor=row['seq']
                    yield f'id: {cursor}\ndata: {dumps(row)}\n\n'
                if len(batch)==200:
                    continue
                if not follow or store.run(run_id)['status'] in TERMINAL:
                    break
                yield ': heartbeat\n\n'
                await asyncio.sleep(.5)
        return StreamingResponse(stream(),media_type='text/event-stream',
            headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get('/api/runs/{run_id}/artifacts/{artifact_id}')
    def artifact(run_id: str, artifact_id: str):
        path = store.artifact_path(run_id,artifact_id)
        return FileResponse(path,media_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    @app.get('/api/runs/{run_id}/snapshot')
    def snapshot(run_id: str, playback_time: float | None=None, version: int | None=None):
        store.run(run_id)
        selected = store.choose_snapshot(run_id,playback_time) if version is None else next(
            (s for s in store.snapshots(run_id) if s['version']==version and s['ready']),None)
        if selected is None:
            return dict(version=None,graph=None,coverage=0)
        path = Path(selected['path'])/'checkpoint.json'
        if digest(path)!=selected['checkpoint_hash']:
            raise ValueError('immutable snapshot changed')
        state = json.loads(path.read_text())
        graph = state['graph']
        # Register actual media evidence for clicks. Never let the browser request
        # arbitrary filesystem paths supplied in model output or query strings.
        images = {}
        for key, frame in state['frame_catalog'].items():
            images[key] = store.artifact(run_id,frame['path'])
        people = []
        for episode in graph['episodes'].values():
            provider = episode.get('identity_evidence',{}).get('provider_snapshot_path')
            if provider:
                saved=json.loads(Path(provider).read_text())['snapshot']
                faces=[]
                for tag, feature in saved.get('feature_registry',{}).items():
                    if feature.get('image'):
                        faces.append(dict(label=tag,media=store.artifact(run_id,feature['image'])))
                people.append(dict(episode_id=episode['id'], faces=faces,
                    mapping=saved.get('identity_evidence',{}).get('feature_mapping',{}),
                    equivalences=saved.get('equivalences',[]),
                    equivalences_accepted=saved.get('equivalences_accepted',False),
                    graph_revision=saved.get('graph_revision'),
                    refs=saved.get('character_refs',[]),
                    artifact=store.artifact(run_id,provider)))
        return dict(version=selected['version'],coverage=selected['coverage'],input_frontier=selected['frontier'],
                    origin=state['contract']['scope']['time_origin'], graph=graph, frames=images,people=people)

    @app.post('/api/runs/{run_id}/questions')
    async def question(run_id: str, body: QuestionInput):
        store.run(run_id)
        try:
            return await asyncio.to_thread(lambda: asyncio.run(ask(store,run_id,body.question,
                                                                   body.playback_time,body.characters)))
        except Exception as exc:
            store.event(run_id,'question','failed',error=f'{type(exc).__name__}: {exc}')
            raise HTTPException(502,f'query failed: {exc}') from exc

    @app.get('/api/runs/{run_id}/questions')
    def questions(run_id: str):
        store.run(run_id)
        return store.questions(run_id)

    static = Path(__file__).parent/'static'
    app.mount('/',StaticFiles(directory=static,html=True),name='workbench')
    return app
