"""Real model adapters, imported lazily. All main comparisons consume the same WAV."""
import os
from pathlib import Path
from urllib.parse import quote

from .common import overlap, validate_turns, duration
from .reused.backends import request_json, parse_json
import base64


def words_to_turns(words, diarization, max_gap=0.5):
    """Split on word-level speaker changes; ambiguous/overlapping words stay unknown.

    This does not separate overlapping voices. Preserve diarization separately for audit.
    """
    result = []
    for w in words:
        if w['end'] <= w['start']:
            continue
        speakers = {d['speaker'] for d in diarization if overlap(w, d) > 0}
        speaker = next(iter(speakers)) if len(speakers) == 1 else None
        ambiguous = len(speakers) > 1
        if (result and result[-1]['speaker'] == speaker and
                result[-1]['overlap'] == ambiguous and
                0 <= w['start'] - result[-1]['end'] <= max_gap):
            result[-1]['end'] = w['end']
            result[-1]['text'] += w['text']
        else:
            result.append(dict(id=f'u{len(result):05d}', start=w['start'], end=w['end'],
                               text=w['text'], speaker=speaker, overlap=ambiguous))
    return result


def infer(wav, config):
    backend = config['backend']
    raw = {}
    if backend == 'gemini':
        key = os.environ.get(config.get('api_key_env', 'GEMINI_API_KEY'))
        model = config['model']
        if not key or model.startswith('SET_'):
            raise ValueError('set Gemini model ID and API key environment variable')
        payload = {'contents': [{'role': 'user', 'parts': [
            {'text': 'Transcribe all intelligible speech verbatim in the original language. '
             'Split at speaker changes and pauses, not just sentences. Retain short utterances. '
             'Do not assign speaker identities. Return JSON {"segments": [{"start": 0.1, '
             '"end": 1.5, "text": "..."}]}. Times are seconds from the start of this audio. '
             'Allow overlapping intervals for simultaneous speech; do not invent words.'},
            {'inline_data': {'mime_type': 'audio/wav',
                             'data': base64.b64encode(Path(wav).read_bytes()).decode()}}
        ]}], 'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json'}}
        import json
        if len(json.dumps(payload).encode()) > 19_000_000:
            raise ValueError('audio exceeds inline experiment limit; use common shorter windows')
        raw = request_json('https://generativelanguage.googleapis.com/v1beta/models/' +
                           quote(model, safe='') + ':generateContent', payload,
                           {'x-goog-api-key': key}, config.get('timeout', 300))
        parts = raw['candidates'][0]['content']['parts']
        parsed = parse_json(''.join(p.get('text', '') for p in parts if not p.get('thought')))
        turns = [dict(s, id=f'u{i:05d}', speaker=None) for i, s in enumerate(parsed['segments'])]
    elif backend == 'whisper_pyannote':
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline
        import torch
        model = WhisperModel(config['asr_model'], device=config.get('device', 'cuda'),
                             compute_type=config.get('compute_type', 'float16'),
                             local_files_only=config.get('local_files_only', True))
        segments, info = model.transcribe(str(wav), language=config.get('language'),
                                         word_timestamps=True, vad_filter=True)
        words = [dict(start=w.start, end=w.end, text=w.word)
                 for segment in segments for w in (segment.words or [])]
        pipeline = Pipeline.from_pretrained(config['diarization_model'],
                                            token=os.environ.get('HF_TOKEN'))
        pipeline.to(torch.device(config.get('device', 'cuda')))
        output = pipeline(str(wav))
        annotation = getattr(output, 'speaker_diarization', output)
        diarization = [dict(start=t.start, end=t.end, speaker=s)
                       for t, _, s in annotation.itertracks(yield_label=True)]
        raw = {'words': words, 'diarization': diarization, 'language': info.language}
        turns = words_to_turns(words, diarization)
    elif backend == 'funasr':
        from funasr import AutoModel
        model = AutoModel(model=config['asr_model'], vad_model=config['vad_model'],
                          punc_model=config['punc_model'], spk_model=config['speaker_model'],
                          device=config.get('device', 'cuda:0'), disable_update=True)
        raw = model.generate(input=str(wav), batch_size_s=config.get('batch_size_s', 60))
        sentences = raw[0].get('sentence_info')
        if sentences is None:
            raise ValueError('FunASR must return sentence_info with millisecond start/end and spk')
        turns = [dict(id=f'u{i:05d}', start=s['start']/1000, end=s['end']/1000,
                      text=s['text'], speaker=str(s['spk']) if 'spk' in s else None)
                 for i, s in enumerate(sentences)]
        # Retain serializable audit fields, not tensor-valued embeddings.
        raw = {'sentence_info': sentences, 'text': raw[0].get('text')}
    else:
        raise ValueError('unsupported speech backend')
    return {'segments': validate_turns(turns, duration(wav)), 'raw': raw}
