"""Browser test harness only. Serves real Grove with explicit synthetic model APIs."""
import argparse
import json
from pathlib import Path
import uvicorn
from lab.workbench.app import create_app
from tests.http_models import config, create_video, model_server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8088)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    video=args.out/'browser-test.mp4'
    if not video.exists():create_video(video)
    with model_server() as (_,base):
        app=create_app(args.out/'service',{'contract-test':dict(config=config(base),description='测试：合成模型协议')})
        print(json.dumps(dict(url=f'http://127.0.0.1:{args.port}',video=str(video.resolve()))),flush=True)
        uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')


if __name__=='__main__':main()
