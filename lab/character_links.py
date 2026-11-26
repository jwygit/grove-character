"""Pure identity projection/index contracts; also vendored into Grove unchanged."""
import copy
import hashlib
import json
import math


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal_state(payload):
    state = copy.deepcopy(payload)
    state.pop('state_id', None)
    state['state_id'] = fingerprint(state)
    return state


def validate_state(state):
    if state.get('schema_version') != 1 or seal_state(state)['state_id'] != state.get('state_id'):
        raise ValueError('invalid identity state fingerprint/schema')


def bind(ids, state):
    ids = list(dict.fromkeys(ids))
    if any(oid not in state['observations'] for oid in ids):
        raise ValueError('identity state is missing original observations')
    people = sorted({state['observations'][oid]['character_id'] for oid in ids})
    return dict(identity_observation_ids=ids, character_id=people[0] if len(people)==1 else None,
                candidate_character_ids=people, binding_status='resolved' if len(people)==1 else
                ('ambiguous' if people else 'unknown'), graph_revision=state['revision'],
                identity_state_id=state['state_id'])


def rebind_ref(ref, state):
    row = copy.deepcopy(ref)
    ids = row.get('identity_observation_ids')
    if ids is None:
        if row.get('character_id') is not None:
            raise ValueError('legacy character reference lacks immutable identity observation IDs')
        ids = []
    row.update(bind(ids, state))
    return row


def face_occurrences(rows, faces, *, episode_id, source, origin, duration):
    """All clustered samples, not just gallery representatives; noise stays unknown."""
    by_cluster = {f['cluster_id']: f for f in faces if 'cluster_id' in f}
    result = []
    for index, row in enumerate(rows):
        time = float(row['timestamp'])
        if not math.isfinite(time) or not 0 <= time < duration:
            raise ValueError('face detection outside episode')
        face = by_cluster.get(int(row['cluster_id']))
        result.append(dict(id=f'{episode_id}:detection:{index}', source=source, role='visible',
            start=origin+time, end=origin+time, sample=True, frame_id=row['frame_id'],
            bounding_box=row['bounding_box'], cluster_id=int(row['cluster_id']),
            identity_observation_ids=list(face.get('observation_ids', [face['observation_id']])) if face else [],
            evidence_ids=[f'{episode_id}:detection:{index}'],
            detection_score=row['extra_data']['face_detection_score']))
    return result


def validate_event(row, state):
    if row['role'] not in ('visible', 'speaker') or not row['id'] or not row['source']:
        raise ValueError('invalid temporal evidence identity/role')
    if not all(math.isfinite(row[k]) for k in ('start', 'end')) or not 0 <= row['start'] <= row['end']:
        raise ValueError('invalid temporal evidence range')
    if (row['role']=='visible') != (row['start']==row['end']):
        raise ValueError('visible evidence must be sampled points; speaker evidence needs duration')
    for oid in row['identity_observation_ids']:
        observation = state['observations'].get(oid)
        if observation is None or observation['source'] != row['source']:
            raise ValueError('temporal evidence observation/source mismatch')
        if observation['modality'] != ('face' if row['role']=='visible' else 'voice'):
            raise ValueError('temporal evidence modality mismatch')


def project_clip(clip, events, state):
    validate_state(state)
    if (not clip.get('source') or not all(math.isfinite(clip[k]) for k in ('start','end'))
            or not 0 <= clip['start'] < clip['end']):
        raise ValueError('clip requires source and a positive time interval')
    refs, seen = [], set()
    for event in events:
        if event['id'] in seen:
            raise ValueError('duplicate temporal evidence ID')
        seen.add(event['id'])
        validate_event(event, state)
        if event['source'] != clip['source']:
            continue
        a, b = event['start'], event['end']
        if a == b:
            if not clip['start'] <= a < clip['end']:
                continue
        else:
            a, b = max(a, clip['start']), min(b, clip['end'])
            if a >= b:
                continue
        ref = dict(event, start=a, end=b, event_id=event['id'])
        ref.update(bind(event['identity_observation_ids'], state))
        refs.append(ref)
    return sorted(refs, key=lambda r: (r['start'], r['role'], r['event_id']))


def covered(clip, spans):
    cursor = clip['start']
    for a, b in sorted((s['start'], s['end']) for s in spans if s['source']==clip['source']):
        if b <= cursor: continue
        if a > cursor+1e-6: break
        cursor = max(cursor, b)
    return cursor >= clip['end']-1e-6


def build_index(records, state):
    validate_state(state)
    postings, seen = {}, set()
    for record in records:
        key = record['node_type'], record['id']
        if key in seen: raise ValueError('duplicate typed evidence node')
        seen.add(key)
        people = {}
        for ref in record.get('character_refs', []):
            rebound = rebind_ref(ref, state)
            if any(ref.get(k) != rebound[k] for k in ('character_id', 'graph_revision', 'identity_state_id')):
                raise ValueError('stale character reference; rebuild before indexing')
            if ref['character_id']:
                people.setdefault(ref['character_id'], set()).add(ref['role'])
        for person, roles in people.items():
            postings.setdefault(person, []).append(dict(node_type=key[0], id=key[1], roles=sorted(roles)))
    return dict(schema_version=1, identity_state_id=state['state_id'], graph_revision=state['revision'],
                records_fingerprint=fingerprint(records), postings=postings)


def lookup(index, state, selectors, roles=None):
    validate_state(state)
    if index.get('identity_state_id') != state['state_id']:
        raise ValueError('character index belongs to a different identity state')
    resolved = set()
    for selector in selectors:
        if selector in state['aliases']:
            resolved.add(state['aliases'][selector])
        elif selector in state.get('names', {}):
            resolved.update(bind(state['names'][selector], state)['candidate_character_ids'])
        else:
            raise ValueError(f'unknown character selector: {selector}')
    matches = {}
    for person in sorted(resolved):
        for row in index['postings'].get(person, []):
            if roles and not set(roles).intersection(row['roles']): continue
            key = row['node_type'], row['id']
            item = matches.setdefault(key, dict(row, matched_characters=[]))
            item['matched_characters'].append(person)
    return sorted(matches.values(), key=lambda r: (r['node_type'], r['id']))
