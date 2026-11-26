import argparse
import asyncio
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from .common import audio, digest, duration, read, write


def metadata():
    packages = {}
    for name in ('numpy', 'pydantic', 'torch', 'faster-whisper', 'pyannote.audio', 'funasr',
                 'transformers', 'qwen-omni-utils', 'accelerate', 'insightface'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[1]
    return {'python': platform.python_version(), 'packages': packages,
            'code_sha256': {str(p.relative_to(root)): digest(p) for p in sorted((root/'lab').rglob('*.py'))}}


def main():
    parser = argparse.ArgumentParser(description='Independent speech / AV / Grove character experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    from .character_commands import add_commands
    add_commands(sub)
    p = sub.add_parser('speech')
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('evaluate-speech')
    p.add_argument('--truth', required=True, type=Path)
    p.add_argument('--prediction', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('perceive')
    for key in ('video', 'speech', 'config', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    for key in ('start', 'end', 'time-origin'):
        p.add_argument('--'+key, required=True, type=float)
    p.add_argument('--source', required=True)
    p.add_argument('--episode-id', required=True)
    p = sub.add_parser('associate')
    for key in ('bundle', 'config', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--accept-model-links', action='store_true')
    p = sub.add_parser('grove-write')
    for key in ('episode', 'snapshot', 'config', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--identity-mode', choices=['predicted', 'unknown'], default='predicted')
    p = sub.add_parser('evaluate-characters')
    for key in ('truth', 'prediction', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p = sub.add_parser('evaluate-association')
    for key in ('truth', 'snapshot', 'bundle', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p = sub.add_parser('evaluate-equivalence')
    for key in ('truth', 'snapshot', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    if args.command in ('identity-review', 'project-clips', 'grove-project', 'character-lookup'):
        from .character_commands import run
        run(args)
        return
    if args.command == 'speech':
        from .speech import infer
        args.out.mkdir(parents=True, exist_ok=False)
        config = read(args.config)
        record = {'status': 'running', 'kind': 'real_inference', 'media_sha256': digest(args.video),
                  'config': config, 'environment': metadata()}
        started = time.perf_counter()
        try:
            wav = args.out/'audio.wav'
            audio(args.video, wav)
            record.update(audio_sha256=digest(wav), duration=duration(wav))
            record.update(infer(wav, config))
            record['status'] = 'completed'
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['wall_seconds'] = time.perf_counter()-started
            record['rtf_including_load'] = record['wall_seconds']/record['duration'] if record.get('duration') else None
            write(args.out/'result.json', record)
    elif args.command == 'evaluate-speech':
        from .metrics import speech_metrics
        truth, pred = read(args.truth), read(args.prediction)
        if not truth.get('reviewed') or truth.get('media_sha256') != pred.get('media_sha256'):
            raise ValueError('reviewed ground truth and matching media hash required')
        if abs(truth['duration']-pred['duration']) > .01:
            raise ValueError('ground truth and prediction durations differ')
        if pred['status'] != 'completed':
            raise ValueError('failed inference must be rerun, not scored as an empty successful result')
        result = speech_metrics(truth['segments'], pred['segments'], truth['duration'])
        write(args.out, dict(metrics=result, eligible_for_selection=not truth.get('synthetic', False),
                             truth_sha256=digest(args.truth), prediction_sha256=digest(args.prediction)))
    elif args.command == 'perceive':
        from .perception import extract
        speech = read(args.speech)
        if speech.get('status') != 'completed' or speech['media_sha256'] != digest(args.video):
            raise ValueError('speech result must match video')
        if not 0 <= args.start < args.end <= speech['duration']:
            raise ValueError('episode outside source media')
        bundle = extract(args.video, speech, args.start, args.end, args.time_origin,
                         args.source, args.episode_id, read(args.config), args.out,
                         source_wav=(args.speech.parent/'audio.wav') if (args.speech.parent/'audio.wav').is_file() else None)
        bundle['speech_sha256'] = digest(args.speech)
        bundle['source_media_sha256'] = digest(args.video)
        bundle['config'] = read(args.config)
        write(args.out/'bundle.json', bundle)
    elif args.command == 'associate':
        from .association import associate
        from .reused.backends import Backend
        args.out.mkdir(parents=True, exist_ok=False)
        bundle, config = read(args.bundle), read(args.config)
        if bundle.get('voice_pipeline_version') != 2:
            raise ValueError('rerun perceive: grouped voice association requires v2 common-PCM evidence')
        if 'reid_margin' in config:
            raise ValueError('remove reid_margin: M3 voice matching uses node mean similarity without margin')
        if config['backend'] == 'replay':
            raise ValueError('replay is for unit tests only, not this real inference command')
        if config['backend'] == 'm3_memorization':
            from .m3 import M3Backend
            backend = M3Backend(config, args.config.resolve().parent)
        else:
            backend = Backend(config, args.config.resolve().parent)
        started = time.perf_counter()
        try:
            prior = read(args.prior)['graph'] if args.prior else None
            snapshot = associate(bundle, backend, prior, args.accept_model_links,
                                 config.get('reid_thresholds'))
            snapshot.update(bundle_sha256=digest(args.bundle), config=config,
                            prior_sha256=digest(args.prior) if args.prior else None,
                            environment=metadata(), wall_seconds=time.perf_counter()-started)
            write(args.out/'snapshot.json', snapshot)
        except Exception as error:
            write(args.out/'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                            'error': str(error), 'bundle_sha256': digest(args.bundle)})
            raise
    elif args.command == 'grove-write':
        from .grove import write_episode, make_llm
        args.out.mkdir(parents=True, exist_ok=False)
        snapshot = read(args.snapshot)
        if args.identity_mode == 'unknown':
            from .grove import hide_identity
            snapshot = hide_identity(snapshot)
        graph = asyncio.run(write_episode(read(args.episode), snapshot,
                                          make_llm(read(args.config)), args.out/'hypergraph.json'))
        write(args.out/'provenance.json', {'episode_sha256': digest(args.episode),
            'snapshot_sha256': digest(args.snapshot), 'identity_mode': args.identity_mode,
            'config': read(args.config), 'environment': metadata(),
            'scope': 'fixed boundary + real writer + real Grove graph; excludes topic/retrieval'})
    elif args.command == 'evaluate-equivalence':
        from .equivalence import equivalence_metrics
        truth, snapshot = read(args.truth), read(args.snapshot)
        if (not truth.get('reviewed') or truth.get('bundle_sha256') != snapshot['bundle_sha256'] or
                truth.get('feature_registry_sha256') != snapshot['feature_registry_sha256']):
            raise ValueError('reviewed truth tied to exact bundle and feature registry required')
        write(args.out, dict(metrics=equivalence_metrics(truth, snapshot),
            eligible_for_selection=not truth.get('synthetic', False),
            truth_sha256=digest(args.truth), snapshot_sha256=digest(args.snapshot)))
    elif args.command == 'evaluate-association':
        from .reused.core import score_clip, metrics
        truth, snapshot, bundle = read(args.truth), read(args.snapshot), read(args.bundle)
        if snapshot.get('association_schema') == 'm3_equivalence_v1':
            raise ValueError('M3 does not predict visibility; use evaluate-equivalence for identity links')
        if not truth.get('reviewed') or truth.get('bundle_sha256') != digest(args.bundle):
            raise ValueError('reviewed association truth tied to exact bundle required')
        if snapshot['bundle_sha256'] != digest(args.bundle):
            raise ValueError('prediction bundle mismatch')
        scores = metrics(score_clip(truth['segments'], snapshot['associations'], bundle['duration'],
                                     {f['id'] for f in bundle['faces']}))
        write(args.out, {'metrics': scores, 'eligible_for_selection': not truth.get('synthetic', False),
                         'truth_sha256': digest(args.truth), 'snapshot_sha256': digest(args.snapshot)})
    else:
        from .metrics import attribution_metrics
        truth, prediction = read(args.truth), read(args.prediction)
        if not truth.get('reviewed') or truth.get('evaluation_set_id') != prediction.get('evaluation_set_id'):
            raise ValueError('reviewed, manually aligned units with matching evaluation_set_id required')
        write(args.out, {'metrics': attribution_metrics(truth['units'], prediction['units']),
                         'eligible_for_selection': not truth.get('synthetic', False),
                         'truth_sha256': digest(args.truth), 'prediction_sha256': digest(args.prediction)})


if __name__ == '__main__':
    main()
