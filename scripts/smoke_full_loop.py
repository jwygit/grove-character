"""Persist a SYNTHETIC model-service run of the real video-to-retrieval Grove loop."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.full_loop import run_video
from tests.http_models import model_server, config, create_video


async def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    video = root/'synthetic-colors.mp4'
    create_video(video)
    with model_server() as (fixture, base):
        _, report = await run_video(video, config(base), root/'memory', 'synthetic-camera', queries=['red cup'])
        report.update(synthetic=True, eligible_for_selection=False, model_calls=dict(fixture.calls),
                      note='Real ffmpeg, HTTP, Grove modules, BM25, dense/visual cosine and persistence; all model outputs synthetic.')
    (root/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=report['status'],stats=report['stats'],synthetic=True,
                         report=str(root/'validation_report.json')),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    asyncio.run(main())
