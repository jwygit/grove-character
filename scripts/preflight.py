"""Check compatible core separately from optional legacy benchmarks and model setup."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root/'vendor/Grove'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path)
args = parser.parse_args()
core = ['hypervideo.extractors.fact_extractor','hypervideo.extractors.fact_deduplicator',
        'hypervideo.extractors.llm_utils','hypervideo.prompts.episode_prompts',
        'hypervideo.prompts.topic_prompts','hypervideo.index.bm25_index',
        'hypervideo.index.embedding_index','hypervideo.retrieval.rrf_fusion',
        'hypervideo.streaming_memory']
errors = {}
for module in core:
    try:
        importlib.import_module(module)
    except Exception as exc:
        errors[module] = f'{type(exc).__name__}: {exc}'
config_errors = []
if args.config:
    from lab.full_loop import validate_model
    try:
        config = json.loads(args.config.read_text())
        for key in ('caption','writer','embedding'):
            validate_model(config[key])
        if config.get('grove',{}).get('use_visual_channel'):
            visual = config['visual_encoder']
            if not visual.get('server_url') or not visual.get('dim') or 'SET_' in visual.get('model_revision','SET_'):
                raise ValueError('configure visual encoder URL, dimension and revision')
    except Exception as exc:
        config_errors.append(str(exc))
else:
    config_errors.append('pass --config to check model configuration')
ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
legacy = ['hypervideo/modules/async_pipeline/pipeline.py','hypervideo/modules/audio_cues/asr.py',
          'hypervideo/modules/audio_cues/diarization.py','hypervideo/modules/entity_id/entity_timeline.py',
          'hypervideo/modules/profile_aggregator/aggregator.py','videorag/_storage/kv_json.py',
          'videorag/_storage/gdb_networkx.py']
print(json.dumps(dict(compatible_pipeline_code_ready=not errors, core_import_errors=errors,
    ffmpeg=ffmpeg,ffprobe=ffprobe,model_config_errors=config_errors,
    full_grove_pipeline_ready=not errors and bool(ffmpeg and ffprobe) and not config_errors,
    model_deployment_verified=False,
    optional_legacy_missing=[p for p in legacy if not (root/'vendor/Grove'/p).exists()],
    note='Readiness means the compatible API loop imports and has configured prerequisites; it does not probe model servers or validate GPU weights/accuracy. Optional legacy benchmark entrypoints remain outside this loop.'),indent=2))
