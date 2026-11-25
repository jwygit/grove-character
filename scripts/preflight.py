"""Reports actual prerequisites, never declares full Grove ready just because tests pass."""
import importlib.util
import json
import shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
grove=root/'vendor/Grove'
missing=[p for p in ['hypervideo/extractors/fact_extractor.py','hypervideo/extractors/fact_deduplicator.py',
                    'hypervideo/extractors/llm_utils.py','hypervideo/prompts/episode_prompts.py',
                    'hypervideo/prompts/topic_prompts.py','hypervideo/modules/audio_cues/asr.py',
                    'hypervideo/modules/audio_cues/diarization.py',
                    'hypervideo/index/bm25_index.py','hypervideo/index/embedding_index.py',
                    'hypervideo/retrieval/rrf_fusion.py'] if not (grove/p).is_file()]
print(json.dumps({'ffmpeg':shutil.which('ffmpeg'),'grove_checkout':grove.is_dir(),
                  'missing_upstream_files':missing,
                  'test_dependencies':{p:importlib.util.find_spec(p) is not None for p in ['numpy','pydantic','hdbscan']},
                  'full_grove_pipeline_ready':False,
                  'note':'Full pipeline requires missing upstream modules and model/runtime verification.'},indent=2))
