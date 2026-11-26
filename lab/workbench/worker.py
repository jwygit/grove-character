"""One durable run per process. The web server does not execute GPU inference."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from lab.full_loop import run_video
from lab.grove import load_grove
from lab.observability import StopRequested
from .store import Store
from .runtime import Observer


def prepare_asr(store, run_id, observer, video):
    root = store.directory(run_id)
    config = root/'speech-config.json'
    result = root/'speech-result.json'
    if not config.exists():
        return
    if result.exists():
        observer.event('speech', 'artifact', files=[str(result)], cached=True, mode='offline_full_audio')
        return
    observer.event('speech', 'started', mode='offline_full_audio')
    launcher = json.loads((root/'launcher.json').read_text())
    executable = launcher.get('speech_python') or sys.executable
    output = root/('speech-attempt-'+uuid4().hex)
    env = dict(os.environ)
    # A separate venv must load its own CUDA/cuDNN, not those of the M3 parent.
    libraries = subprocess.check_output([executable, '-c',
        'import sysconfig,pathlib; p=pathlib.Path(sysconfig.get_paths()["purelib"]); '
        'print(":".join(str(x) for x in (p/"nvidia").glob("*/lib")))'], text=True).strip()
    if libraries:
        env['LD_LIBRARY_PATH'] = libraries + ':' + env.get('LD_LIBRARY_PATH','')
    log = root/'speech-worker.log'
    with log.open('ab') as stream:
        subprocess.run([executable, '-m', 'lab', 'speech', '--video', video['path'],
            '--config', str(config), '--out', str(output)], env=env, stdout=stream, stderr=stream, check=True)
    from hypervideo.streaming_memory import atomic_json
    record = json.loads((output/'result.json').read_text())
    if record.get('status') != 'completed':
        raise ValueError('ASR did not complete')
    atomic_json(result, record)
    observer.event('speech', 'artifact', files=[str(result),str(output/'audio.wav'),str(log)],
                   count=len(record['segments']), mode='offline_full_audio')
    observer.event('speech', 'completed')


def execute(store, run_id):
    load_grove()
    root = store.directory(run_id)
    # Repeated starts/restarts never run two writers for one task or one GPU queue.
    with (root/'worker.lock').open('a') as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with (store.root/'compute.lock').open('a') as compute_lock:
            fcntl.flock(compute_lock, fcntl.LOCK_EX)
            observer = Observer(store, run_id)
            try:
                if observer.stop_requested():
                    raise StopRequested()
                store.state(run_id, 'running')
                run = store.run(run_id)
                video = store.video(run['video_id'])
                prepare_asr(store, run_id, observer, video)
                if observer.stop_requested():
                    raise StopRequested()
                config = json.loads((root/'config.json').read_text())
                asyncio.run(run_video(video['path'], config, root/'pipeline', video['id'], observer=observer))
                store.state(run_id, 'complete')
            except StopRequested:
                store.state(run_id, 'stopped')
            except Exception as exc:
                store.state(run_id, 'failed', f'{type(exc).__name__}: {exc}')
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    execute(Store(args.data), args.run)


if __name__ == '__main__':
    main()
