"""Auditable metrics. Not DER: Gemini deliberately emits no speaker labels."""
import unicodedata
from collections import Counter
from .common import validate_turns


def ratio(a, b):
    return a / b if b else None


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if not unicodedata.category(c).startswith(('P', 'S')))


def boundary_counts(truth, predicted, collar):
    # Earliest feasible match maximizes cardinality for sorted 1-D tolerance intervals.
    a, b = sorted(truth), sorted(predicted)
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= collar:
            matched += 1
            i += 1
            j += 1
        elif b[j] < a[i] - collar:
            j += 1
        else:
            i += 1
    return matched, len(b) - matched, len(a) - matched


def speech_metrics(truth, predicted, length, collar=0.25):
    validate_turns(truth, length)
    validate_turns(predicted, length)
    reference = normalize(' '.join(s['text'] for s in sorted(truth, key=lambda x: x['start'])))
    hypothesis = normalize(' '.join(s['text'] for s in sorted(predicted, key=lambda x: x['start'])))
    ref_chars, hyp_chars = ''.join(reference.split()), ''.join(hypothesis.split())
    char_errors, word_errors = edits(ref_chars, hyp_chars), edits(reference.split(), hypothesis.split())
    points = sorted({0., length, *(s[k] for s in truth + predicted for k in ('start', 'end'))})
    tp = fp = fn = overlap_seconds = 0.
    per_segment = [Counter() for _ in predicted]
    for a, b in zip(points, points[1:]):
        t, dt = (a+b)/2, b-a
        g = [s for s in truth if s['start'] <= t < s['end']]
        p = [i for i, s in enumerate(predicted) if s['start'] <= t < s['end']]
        tp += dt * bool(g and p)
        fp += dt * bool(p and not g)
        fn += dt * bool(g and not p)
        speakers = {s.get('speaker') for s in g} - {None}
        overlap_seconds += dt * (len(speakers) > 1)
        if len(speakers) == 1:
            for i in p:
                per_segment[i][next(iter(speakers))] += dt
    total = sum(sum(c.values()) for c in per_segment)
    dominant = sum(max(c.values(), default=0) for c in per_segment)
    boundary = [boundary_counts([s[k] for s in truth], [s[k] for s in predicted], collar)
                for k in ('start', 'end')]
    bt, bf, bn = (sum(x[i] for x in boundary) for i in range(3))
    return {'cer': ratio(char_errors, len(ref_chars)), 'wer': ratio(word_errors, len(reference.split())),
            'char_errors': char_errors, 'reference_chars': len(ref_chars),
            'word_errors': word_errors, 'reference_words': len(reference.split()),
            'boundary_f1': ratio(2*bt, 2*bt+bf+bn), 'boundary_collar_seconds': collar,
            'speech_f1': ratio(2*tp, 2*tp+fp+fn), 'missed_speech_seconds': fn,
            'false_speech_seconds': fp, 'segment_speaker_purity': ratio(dominant, total),
            'mixed_speaker_seconds': total-dominant, 'reference_overlap_seconds': overlap_seconds}


def attribution_metrics(truth, predicted):
    """Aligned semantic units: (unit_id, role) -> canonical character_id or null.

    Human adjudication must align generated facts to reference units beforehand.
    Extra output units count as false positives; omissions cannot improve recall.
    """
    def index(rows):
        result = {}
        for r in rows:
            key = (r['unit_id'], r['role'])
            if key in result:
                raise ValueError('duplicate attribution unit/role')
            result[key] = r['character_id']
        return result
    g, p = index(truth), index(predicted)
    tp = sum(v is not None and p.get(k) == v for k, v in g.items())
    assigned = sum(v is not None for v in p.values())
    known = sum(v is not None for v in g.values())
    unknown = sum(v is None for v in g.values())
    return {'character_precision': ratio(tp, assigned), 'character_recall': ratio(tp, known),
            'character_f1': ratio(2*tp, assigned+known),
            'unit_accuracy': ratio(sum(k in p and p[k] == v for k, v in g.items()), len(g)),
            'coverage': ratio(sum(k in p and p[k] is not None for k in g), len(g)),
            'unknown_false_binding_rate': ratio(sum(v is None and p.get(k) is not None for k,v in g.items()), unknown),
            'missing_units': len(g.keys()-p.keys()), 'extra_units': len(p.keys()-g.keys())}
