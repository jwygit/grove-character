"""Episode-scoped matching using the existing AV adapter and identity graph."""
import copy
from .common import overlap, validate_turns
from .reused.identity import IdentityGraph


def resolve(bundle, predictions, prior=None, accept_links=False, match_threshold=None, margin=0.08):
    """Only observations from this closed episode enter this graph snapshot.

    No embedding is compared across modalities. Local speaker labels never become IDs.
    accept_links is an experimental policy, NOT a claim of ground-truth certainty.
    """
    validate_turns(bundle['utterances'], bundle['duration'])
    faces = {f['id']: f for f in bundle['faces']}
    utterances = {u['id']: u for u in bundle['utterances']}
    if len(faces) != len(bundle['faces']):
        raise ValueError('duplicate face IDs')
    found = [p['utterance_id'] for p in predictions]
    if len(found) != len(set(found)) or set(found) != set(utterances):
        raise ValueError('association must cover each utterance exactly once')
    graph = IdentityGraph(prior)
    origin = bundle['time_origin']
    source = bundle['source']
    if any(o['end'] > origin + bundle['duration'] for o in graph.data['observations'].values()):
        raise ValueError('prior graph contains future observations; replay chronologically')
    # Prefix every input ID with a recording/episode namespace before constructing bundle.
    for o in bundle['observations']:
        if o['source'] != source or not origin <= o['start'] <= o['end'] <= origin+bundle['duration']:
            raise ValueError('observation outside closed episode/source')
        graph.observe(o)
        if 'voice' in (match_threshold or {}) and o['modality'] == 'voice':
            ranked = graph.candidates(o['id'])
            # Compare different components; duplicate samples of a person are not competitors.
            mapping = {x: p['person_id'] for p in graph.people() for x in p['faces']+p['voices']}
            groups = {}
            for item in ranked:
                if mapping[item['observation_id']] != mapping[o['id']]:
                    groups.setdefault(mapping[item['observation_id']], item)
            candidates = list(groups.values())
            threshold = match_threshold[o['modality']]
            if candidates and candidates[0]['cosine'] >= threshold and (
                    len(candidates) == 1 or candidates[0]['cosine']-candidates[1]['cosine'] >= margin):
                link_id = 'reid:' + o['id']
                if link_id not in graph.data['links']:
                    graph.propose(dict(id=link_id, left=o['id'], right=candidates[0]['observation_id'],
                                       method='embedding', evidence={'cosine': candidates[0]['cosine']}))
                    graph.review(link_id, 'confirmed', 'experiment-policy', 'threshold and margin')
    from .faces import link_face_groups
    face_config = bundle.get('config', {})
    face_threshold = (match_threshold or {}).get('face', face_config.get('img_matching_threshold', .3))
    link_face_groups(graph, list(faces.values()), threshold=face_threshold,
                     max_embeddings=face_config.get('max_img_embeddings',10),
                     seed=face_config.get('face_sampling_seed',0))
    obs = graph.data['observations']
    for f in faces.values():
        if f['observation_id'] not in obs or obs[f['observation_id']]['modality'] != 'face':
            raise ValueError('invalid face observation reference')
    for u in utterances.values():
        v = u.get('voice_observation_id')
        if v and (v not in obs or obs[v]['modality'] != 'voice' or obs[v]['source'] != source):
            raise ValueError('invalid voice observation reference')
    selected = {}
    for p in predictions:
        u, face_id = utterances[p['utterance_id']], p['face_id']
        if face_id is not None and face_id not in faces:
            raise ValueError('hallucinated face ID')
        selected[u['id']] = face_id
        voice = u.get('voice_observation_id')
        if face_id is None or not voice:
            continue
        link_id = 'av:' + u['id'] + ':' + face_id
        if link_id not in graph.data['links']:
            graph.propose(dict(id=link_id, left=voice, right=faces[face_id]['observation_id'],
                               method='audiovisual', evidence={'utterance_id': u['id'],
                               'source': source, 'policy_accept': accept_links}))
            if accept_links:
                graph.review(link_id, 'confirmed', 'experiment-policy', 'accept model association baseline')
    exported = graph.export()
    mapping = {e['observation_id']: e['person_id'] for e in exported['events']}
    refs = []
    for u in utterances.values():
        face_id = selected[u['id']]
        voice = u.get('voice_observation_id')
        evidence = [voice] if voice else []
        character = mapping.get(voice)
        if face_id and accept_links:
            face_obs = faces[face_id]['observation_id']
            evidence.append(face_obs)
            character = mapping[face_obs]
        refs.append({'utterance_id': u['id'], 'start': origin+u['start'], 'end': origin+u['end'],
                     'text': u['text'], 'character_id': character, 'role': 'speaker',
                     'evidence_ids': evidence, 'graph_revision': exported['revision']})
    return {'source': source, 'start': origin, 'end': origin+bundle['duration'],
            'graph_revision': exported['revision'], 'character_refs': refs,
            'graph': copy.deepcopy(graph.data), 'associations': predictions}
