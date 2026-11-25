"""Lazy local face/voice extraction, borrowing M3's model choices, not its CUDA globals.

ERes2NetV2 initialization/FBank convention follows ByteDance-Seed/m3-agent
mmagent/voice_processing.py (Apache-2.0). See docs/PROVENANCE.md.
"""
from pathlib import Path
import subprocess
import wave
from .common import overlap


def episode_turns(speech, start, end, source):
    turns = []
    for u in speech['segments']:
        if not overlap(u, dict(start=start, end=end)):
            continue
        partial = u['start'] < start or u['end'] > end
        # Without word alignment, never duplicate a full sentence into both episodes.
        text = u['text'] if not partial else ''
        if partial and speech.get('raw', {}).get('words'):
            text = ''.join(w['text'] for w in speech['raw']['words']
                           if max(start,u['start']) <= w['start'] and w['end'] <= min(end,u['end']))
        turns.append(dict(u, id=source+':'+u['id'], start=max(u['start'],start)-start,
                          end=min(u['end'],end)-start, text=text, partial=partial))
    return turns


def extract(video, speech, start, end, origin, source, episode_id, config, out):
    if any(config[k].startswith('SET_') for k in ('face_model_id','voice_model_id')):
        raise ValueError('replace model IDs with exact weight fingerprints before extraction')
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    clip = out/'episode.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video), '-ss', str(start),
                    '-t', str(end-start), '-c:v', 'libx264', '-c:a', 'aac', '-n', str(clip)], check=True)
    wav = out/'audio.wav'
    from .common import audio
    audio(clip, wav)
    app = FaceAnalysis(name=config.get('face_model', 'buffalo_l'),
                       root=config['insightface_root'],
                       providers=config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider']))
    app.prepare(ctx_id=config.get('ctx_id', 0), det_size=(640, 640))
    cap = cv2.VideoCapture(str(clip))
    faces, observations, prototypes = [], [], []
    try:
        for t in np.arange(0, end-start, 1/config.get('face_fps', 2)):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            for face in app.get(frame):
                if float(face.det_score) < config.get('detection_threshold', 0.7):
                    continue
                v = face.normed_embedding
                scores = sorted([(float(v @ p), i) for i, p in enumerate(prototypes)], reverse=True)
                if scores and scores[0][0] >= config['face_cluster_threshold']:
                    # This is sampled face clustering, not a tracker or active-speaker detector.
                    if len(scores) == 1 or scores[0][0]-scores[1][0] >= config.get('face_cluster_margin', 0.08):
                        continue
                face_id = f'{episode_id}:f{len(faces):04d}'
                x1, y1, x2, y2 = face.bbox.astype(int)
                crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                if not crop.size:
                    continue
                image = out/f'face_{len(faces):04d}.jpg'
                if not cv2.imwrite(str(image), crop):
                    raise RuntimeError('face crop write failed')
                faces.append(dict(id=face_id, observation_id=face_id, image=str(image.resolve())))
                prototypes.append(v)
                observations.append(dict(id=face_id, modality='face', source=source,
                    start=origin+start+float(t), end=origin+start+float(t),
                    model=config['face_model_id'], embedding=v.tolist()))
    finally:
        cap.release()
    if len(faces) > config.get('max_face_references', 24):
        raise ValueError('too many face references; review clustering or shorten common episodes')
    turns = episode_turns(speech, start, end, source)
    # Transcript is retained for short turns; only voice embedding can be skipped.
    eligible = [u for u in turns if u['end']-u['start'] >= config.get('min_voice_seconds', 2)
                and not u.get('overlap', False)]
    if eligible:
        import torch
        from speakerlab.process.processor import FBank
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
        device = config.get('device', 'cuda')
        model = ERes2NetV2(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(config['voice_checkpoint'], map_location='cpu', weights_only=True))
        model.to(device).eval()
        fbank = FBank(80, sample_rate=16000, mean_nor=True)
        with wave.open(str(wav)) as f:
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)/32768.
        for u in eligible:
            signal = torch.from_numpy(samples[int(u['start']*16000):int(u['end']*16000)])
            with torch.inference_mode():
                vector = model(fbank(signal.unsqueeze(0)).unsqueeze(0).to(device)).squeeze(0).cpu().tolist()
            obs_id = episode_id+':voice:'+u['id']
            observations.append(dict(id=obs_id, modality='voice', source=source,
                start=origin+start+u['start'], end=origin+start+u['end'],
                model=config['voice_model_id'], embedding=vector))
            u['voice_observation_id'] = obs_id
    return dict(id=episode_id, source=source, time_origin=origin+start, duration=end-start,
                video=str(clip.resolve()), faces=faces, utterances=turns, observations=observations)
