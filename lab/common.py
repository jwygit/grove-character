import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audio(video, output):
    # AAC decoder padding can extend a nominal 60s clip to 60.032s. Bound the
    # decoded WAV to the declared media duration so all frontends share its clock.
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                     'format=duration', '-of', 'json', str(video)], text=True)
    length = float(json.loads(probe)['format']['duration'])
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video),
                    '-vn', '-t', str(length), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                    '-n', str(output)], check=True)


def duration(wav):
    with wave.open(str(wav)) as f:
        return f.getnframes() / f.getframerate()


def validate_turns(turns, length):
    ids = set()
    for u in turns:
        if not isinstance(u['id'], str) or not u['id'] or u['id'] in ids:
            raise ValueError('utterance IDs must be unique nonempty strings')
        ids.add(u['id'])
        a, b = u['start'], u['end']
        if not all(type(t) in (int, float) and math.isfinite(t) for t in (a, b)):
            raise ValueError('finite numeric timestamps required')
        if not 0 <= a < b <= length:
            raise ValueError('utterance outside media interval')
        if not isinstance(u['text'], str):
            raise ValueError('text must be a string')
    return turns


def overlap(a, b):
    return max(0., min(a['end'], b['end']) - max(a['start'], b['start']))
