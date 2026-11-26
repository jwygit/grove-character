"""Durable task/event metadata. Model artifacts and immutable checkpoints stay on disk."""
import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'workbench.sqlite3'
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
                    profile TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                    stop INTEGER DEFAULT 0, error TEXT DEFAULT '');
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, data TEXT NOT NULL, unique_key TEXT UNIQUE);
                CREATE INDEX IF NOT EXISTS event_run ON events(run_id,seq);
                CREATE TABLE IF NOT EXISTS snapshots(run_id TEXT NOT NULL, version INTEGER NOT NULL,
                    graph_hash TEXT NOT NULL, frontier REAL NOT NULL, coverage REAL NOT NULL,
                    path TEXT NOT NULL, checkpoint_hash TEXT NOT NULL, ready INTEGER DEFAULT 0,
                    PRIMARY KEY(run_id,version), UNIQUE(run_id,graph_hash));
                CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    path TEXT NOT NULL, sha256 TEXT NOT NULL, UNIQUE(run_id,path,sha256));
                CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    created REAL NOT NULL, data TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def directory(self, run_id):
        self.run(run_id)
        return self.root / 'runs' / run_id

    def video(self, video_id):
        with self.connect() as db:
            row = db.execute('SELECT data FROM videos WHERE id=?', (video_id,)).fetchone()
        if row is None:
            raise KeyError('video not found')
        return json.loads(row['data'])

    def add_video(self, data):
        with self.connect() as db:
            db.execute('INSERT INTO videos VALUES (?,?)', (data['id'], dumps(data)))

    def create_run(self, video_id, profile, config, speech_config=None, speech_python=None):
        self.video(video_id)
        from lab.common import write
        rid = uuid4().hex
        directory = self.root / 'runs' / rid
        directory.mkdir(parents=True)
        if speech_config is not None:
            target = str(directory/'speech-result.json')
            if config.get('characters'):
                config['characters']['speech_input'] = target
                config['characters'].pop('speech', None)
            config['speech'] = {'speech_input': target}
            write(directory/'speech-config.json', speech_config)
            write(directory/'launcher.json', {'speech_python': speech_python})
        write(directory/'config.json', config)
        with self.connect() as db:
            db.execute('INSERT INTO runs(id,video_id,profile,status,created) VALUES (?,?,?,?,?)',
                       (rid, video_id, profile, 'queued', time.time()))
        self.event(rid, 'run', 'queued')
        return self.run(rid)

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError('run not found')
        return dict(row)

    def runs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM runs ORDER BY created DESC')]

    def state(self, run_id, status, error=''):
        with self.connect() as db:
            db.execute('UPDATE runs SET status=?,error=? WHERE id=?', (status, error, run_id))
        self.event(run_id, 'run', status, error=error)

    def request_stop(self, run_id):
        self.run(run_id)
        with self.connect() as db:
            db.execute('UPDATE runs SET stop=1 WHERE id=?', (run_id,))

    def resume(self, run_id):
        with self.connect() as db:
            result = db.execute("UPDATE runs SET status='queued',stop=0,error='' WHERE id=? AND status IN ('failed','stopped','interrupted')", (run_id,))
            if result.rowcount != 1:
                raise ValueError('only failed, stopped or interrupted runs can resume')
        self.event(run_id, 'run', 'queued')

    def event(self, run_id, stage, state, *, unique_key=None, **data):
        record = dict(schema_version=1, stage=stage, state=state, wall_time=time.time(), **data)
        with self.connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO events(run_id,data,unique_key) VALUES (?,?,?)',
                                (run_id, dumps(record), unique_key))
            return cursor.lastrowid

    def events(self, run_id, after=0, limit=200):
        with self.connect() as db:
            rows = db.execute('SELECT seq,data FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?',
                              (run_id, after, limit)).fetchall()
        return [dict(json.loads(row['data']), seq=row['seq']) for row in rows]

    def artifact(self, run_id, path):
        path = Path(path).resolve(strict=True)
        allowed = self.directory(run_id).resolve()
        if not path.is_relative_to(allowed) or not path.is_file():
            raise ValueError('artifact must be a file in its run directory')
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        aid = uuid4().hex
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?)', (aid, run_id, str(path), sha))
            row = db.execute('SELECT id FROM artifacts WHERE run_id=? AND path=? AND sha256=?',
                             (run_id, str(path), sha)).fetchone()
        return dict(id=row['id'], name=path.name, url=f'/api/runs/{run_id}/artifacts/{row["id"]}')

    def artifact_path(self, run_id, artifact_id):
        with self.connect() as db:
            row = db.execute('SELECT path,sha256 FROM artifacts WHERE run_id=? AND id=?', (run_id, artifact_id)).fetchone()
        if row is None:
            raise KeyError('artifact not found')
        path = Path(row['path']).resolve(strict=True)
        if not path.is_relative_to(self.directory(run_id).resolve()):
            raise ValueError('artifact escaped its run directory')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError('artifact changed after registration')
        return path

    def snapshots(self, run_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM snapshots WHERE run_id=? ORDER BY version', (run_id,))]

    def choose_snapshot(self, run_id, playback=None):
        if playback is not None and (not math.isfinite(playback) or playback < 0):
            raise ValueError('invalid playback time')
        candidates = [s for s in self.snapshots(run_id) if s['ready'] and
                      (playback is None or s['frontier'] <= playback)]
        return candidates[-1] if candidates else None

    def save_question(self, run_id, data):
        qid = uuid4().hex
        data = dict(data, id=qid)
        with self.connect() as db:
            db.execute('INSERT INTO questions VALUES (?,?,?,?)', (qid, run_id, time.time(), dumps(data)))
        return data

    def questions(self, run_id):
        with self.connect() as db:
            return [json.loads(r['data']) for r in db.execute('SELECT data FROM questions WHERE run_id=? ORDER BY created', (run_id,))]
