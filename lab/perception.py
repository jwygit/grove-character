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


def extract(video, speech, start, end, origin, source, episode_id, config, out, source_wav=None):
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
    from .common import audio, digest
    from .voices import slice_pcm, eligible_turns
    if source_wav is None:
        source_wav = out/'source_audio.wav'
        audio(video, source_wav)
    # Verify the same decoded audio as the ASR stage when provenance is available.
    if speech.get('audio_sha256') and digest(source_wav) != speech['audio_sha256']:
        raise ValueError('voice audio differs from the ASR input')
    slice_pcm(source_wav, wav, start, end)
    app = FaceAnalysis(name=config.get('face_model', 'buffalo_l'),
                       root=config['insightface_root'],
                       providers=config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider']))
    app.prepare(ctx_id=config.get('ctx_id', 0), det_size=(640, 640))
    cap = cv2.VideoCapture(str(clip))
    from .faces import select_faces
    from .common import write
    detections, faces, observations = [], [], []
    try:
        for frame_id, t in enumerate(np.arange(0, end-start, 1/config.get('face_fps', 5))):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            for face in app.get(frame):
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0,int(x1)), max(0,int(y1))
                x2, y2 = min(frame.shape[1],int(x2)), min(frame.shape[0],int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                ok, jpeg = cv2.imencode('.jpg', frame[y1:y2,x1:x2])
                if not ok:
                    continue
                aspect = (y2-y1)/(x2-x1)
                detections.append(dict(frame_id=frame_id, timestamp=float(t),
                    bounding_box=[x1,y1,x2,y2], face_emb=face.normed_embedding.tolist(),
                    cluster_id=-1, crop_jpeg=jpeg.tobytes(), extra_data={
                        'face_type':'ortho' if 1 < aspect < 1.5 else 'side',
                        'face_detection_score':float(face.det_score),
                        'face_quality_score':float(np.linalg.norm(face.embedding,ord=2))}))
    finally:
        cap.release()
    labeled, groups = select_faces(detections, config)
    write(out/'face_detections.json', [dict(
        {k:v for k,v in row.items() if k != 'crop_jpeg'}, cluster_id=int(row['cluster_id']))
        for row in labeled])
    for cluster_id, representatives in groups.items():
        face_id = f'{episode_id}:f{cluster_id:04d}'
        ids, images = [], []
        for rank, row in enumerate(representatives):
            obs_id = f'{face_id}:r{rank}'
            image = out/f'face_{cluster_id:04d}_{rank}.jpg'
            image.write_bytes(row['crop_jpeg'])
            ids.append(obs_id)
            images.append(str(image.resolve()))
            observations.append(dict(id=obs_id, modality='face', source=source,
                start=origin+start+row['timestamp'], end=origin+start+row['timestamp'],
                model=config['face_model_id'], embedding=row['face_emb'],
                detection_score=row['extra_data']['face_detection_score'],
                quality_score=row['extra_data']['face_quality_score'],
                frame_id=row['frame_id'], bounding_box=row['bounding_box'], cluster_id=cluster_id))
        faces.append(dict(id=face_id, cluster_id=int(cluster_id), observation_id=ids[0], observation_ids=ids,
                          image=images[0], images=images))
    if len(faces) > config.get('max_face_references', 24):
        raise ValueError('too many face clusters; shorten common episodes')
    turns = episode_turns(speech, start, end, source)
    # Transcript is retained for short turns; only voice embedding can be skipped.
    eligible = eligible_turns(turns, config.get('min_voice_seconds', 2))
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
    from .character_links import face_occurrences
    timeline = face_occurrences(labeled, faces, episode_id=episode_id, source=source,
                                origin=origin+start, duration=end-start)
    return dict(id=episode_id, source=source, time_origin=origin+start, duration=end-start,
                face_occurrences=timeline, face_timeline_version=1,
                video=str(clip.resolve()), episode_video_sha256=digest(clip), faces=faces, utterances=turns, observations=observations,
                voice_audio_sha256=digest(wav), source_audio_sha256=digest(source_wav), voice_pipeline_version=2)
