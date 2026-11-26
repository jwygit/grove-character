import argparse
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Serve the real Grove video memory workbench')
    parser.add_argument('--profiles',type=Path,required=True,help='server-owned JSON profile registry')
    parser.add_argument('--data',type=Path,default=Path('runs/workbench'))
    parser.add_argument('--speech-python',help='Python executable in the separate ASR environment')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--max-upload-gb',type=float,default=4)
    args=parser.parse_args()
    if not math.isfinite(args.max_upload_gb) or args.max_upload_gb<=0:
        parser.error('--max-upload-gb must be finite and positive')
    from .app import create_app
    import uvicorn
    app=create_app(args.data,json.loads(args.profiles.read_text()),speech_python=args.speech_python,
                   max_upload_bytes=int(args.max_upload_gb*1024**3))
    uvicorn.run(app,host=args.host,port=args.port)


if __name__=='__main__':
    main()
