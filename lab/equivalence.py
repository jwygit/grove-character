"""Episode face/voice identity-pair scoring, independent of screen visibility."""


def equivalence_metrics(truth, snapshot):
    if snapshot.get('association_schema') != 'm3_equivalence_v1':
        raise ValueError('M3 identity equivalence snapshot required')
    registry = snapshot['feature_registry']
    faces = [tag for tag, row in registry.items() if row['modality'] == 'face']
    voices = [tag for tag, row in registry.items() if row['modality'] == 'voice']
    expected = {(face, voice) for face in faces for voice in voices}
    rows = truth['pairs']
    pairs = [(r['face'], r['voice']) for r in rows]
    if len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise ValueError('truth must label every supplied face/voice pair exactly once')
    if any(type(r['same_person']) is not bool for r in rows):
        raise ValueError('same_person must be boolean')
    positives = {(r['face'], r['voice']) for r in rows if r['same_person']}
    predicted = {(r['face'], r['voice']) for r in snapshot['equivalences']}
    mapping = snapshot['identity_evidence']['feature_mapping']
    resolved = {(f, v) for f, v in expected if mapping[f]['character_id'] is not None
                and mapping[f]['character_id'] == mapping[v]['character_id']}
    def score(found):
        tp, fp, fn = len(found & positives), len(found-positives), len(positives-found)
        return dict(true_positive=tp, false_positive=fp, false_negative=fn,
                    precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                    pair_accuracy=(len(expected)-fp-fn)/len(expected) if expected else None)
    return dict(pair_count=len(expected), model_equivalences=score(predicted),
                resolved_character_equivalences=score(resolved))
