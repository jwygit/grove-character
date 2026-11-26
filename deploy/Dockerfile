# Build online on Linux x86_64, save images, then load them in the intranet.
# Candidate environments, not a claim of GPU/model validation. See OFFLINE_DEPLOYMENT.md.
FROM python:3.11-slim-bookworm AS core
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates libsndfile1 libgomp1 libgl1 curl unzip \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements-test.txt requirements-workbench.txt /opt/
COPY deploy/entrypoint.sh /opt/grove-entrypoint.sh
RUN pip install --no-cache-dir -r /opt/requirements-workbench.txt huggingface-hub==0.34.4 \
    && pip check && pip freeze > /opt/python-packages.txt
ENV HF_HOME=/tmp/grove-hf XDG_CACHE_HOME=/tmp/grove-cache
WORKDIR /work
ENTRYPOINT ["bash", "/opt/grove-entrypoint.sh"]

FROM core AS visual
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    transformers==4.51.3 sentencepiece==0.2.0 protobuf==5.29.5 Pillow==11.2.1 \
    && pip check && pip freeze > /opt/python-packages.txt

FROM core AS asr
# pyannote 4.0.3 pins torch/torchaudio 2.8.0 and torchcodec 0.7.0.
# Keep this separate from M3's earlier Transformers/Torch environment.
RUN pip install --no-cache-dir faster-whisper==1.2.1 pyannote.audio==4.0.3 \
    && pip check && pip freeze > /opt/python-packages.txt
ENV PYANNOTE_METRICS_ENABLED=0

FROM core AS character
ARG SPEAKERLAB_REVISION
RUN test -n "$SPEAKERLAB_REVISION" \
    && git clone https://github.com/modelscope/3D-Speaker.git /opt/3D-Speaker \
    && git -C /opt/3D-Speaker checkout --detach "$SPEAKERLAB_REVISION" \
    && git -C /opt/3D-Speaker rev-parse HEAD > /opt/speakerlab-revision.txt
RUN pip install --no-cache-dir numpy==1.26.4 cython wheel setuptools \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    opencv-python-headless==4.11.0.86 albumentations==1.4.3 onnxruntime-gpu==1.21.1 \
    && pip install --no-cache-dir --no-build-isolation insightface==0.7.3
# Memorization setup pinned by M3 upstream; not its separate control/vLLM setup.
RUN pip install --no-cache-dir \
    'git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8' \
    qwen-omni-utils==0.0.4 'accelerate>=1,<2' 'librosa>=0.10.2,<0.12' \
    soundfile sentencepiece protobuf av decord pyyaml \
    && pip check && pip freeze > /opt/python-packages.txt
ENV PYTHONPATH=/opt/3D-Speaker:/work

FROM character AS workbench
# One service queues the main Grove/M3 process, but launches ASR using its own
# Python/CUDA libraries. Do not mix pyannote 4's Torch with M3's pinned stack.
RUN python -m venv /opt/asr-env \
    && /opt/asr-env/bin/pip install --no-cache-dir -r /opt/requirements-test.txt \
       faster-whisper==1.2.1 pyannote.audio==4.0.3 \
    && /opt/asr-env/bin/pip check \
    && /opt/asr-env/bin/pip freeze > /opt/asr-python-packages.txt
ENV PYANNOTE_METRICS_ENABLED=0
