"""Prepare face/voice nodes BEFORE audiovisual inference; resolve results afterward."""
import copy
from dataclasses import dataclass
from .common import validate_turns
from .faces import link_face_groups
from .voices import eligible_turns, group_voices
from .reused.identity import IdentityGraph


@dataclass
class PreparedAssociation:
    bundle: dict
    graph: IdentityGraph
    clip: dict
    voice_ids: dict
    exclusions: dict


def prepare_association(bundle, prior=None, match_threshold=None):
    bundle = copy.deepcopy(bundle)
    validate_turns(bundle['utterances'], bundle['duration'])
    if len({f['id'] for f in bundle['faces']}) != len(bundle['faces']):
        raise ValueError('duplicate face IDs')
    graph = IdentityGraph(prior)
    origin, source = bundle['time_origin'], bundle['source']
    if any(o['end'] > origin+bundle['duration'] for o in graph.data['observations'].values()):
        raise ValueError('prior graph contains future observations; replay chronologically')
    config = bundle.get('config', {})
    eligible = {u['id'] for u in eligible_turns(bundle['utterances'], config.get('min_voice_seconds',2))}
    supplied = {o['id']:o for o in bundle['observations']}
    if len(supplied) != len(bundle['observations']):
        raise ValueError('duplicate observation IDs')
    usable, exclusions = {}, {}
    for u in bundle['utterances']:
        voice = u.get('voice_observation_id')
        if u['id'] not in eligible:
            exclusions[u['id']] = 'shorter_than_min_voice_seconds'
            continue
        if not voice:
            exclusions[u['id']] = 'no_voice_embedding'
            continue
        o = supplied.get(voice)
        if (o is None or o['modality']!='voice' or o['source']!=source or
                abs(o['start']-(origin+u['start'])) > 1/16000 or
                abs(o['end']-(origin+u['end'])) > 1/16000):
            raise ValueError('voice observation must match its utterance, source and time interval')
        if voice in usable.values():
            raise ValueError('each utterance requires its own voice observation')
        usable[u['id']] = voice
    for o in bundle['observations']:
        if o['source']!=source or not origin <= o['start'] <= o['end'] <= origin+bundle['duration']:
            raise ValueError('observation outside closed episode/source')
        if o['modality']=='voice' and o['id'] not in usable.values():
            continue
        graph.observe(o)
    thresholds = match_threshold or {}
    link_face_groups(graph, bundle['faces'], threshold=thresholds.get('face',config.get('img_matching_threshold',.3)),
                     max_embeddings=config.get('max_img_embeddings',10),seed=config.get('face_sampling_seed',0))
    membership = group_voices(graph,list(usable.values()),
                              threshold=thresholds.get('voice',config.get('audio_matching_threshold',.6)),
                              max_embeddings=config.get('max_audio_embeddings',20),
                              seed=config.get('voice_sampling_seed',0))
    voice_ids = {uid:membership[oid] for uid,oid in usable.items()}
    groups = {}
    for u in sorted(bundle['utterances'],key=lambda u:(u['start'],u['end'],u['id'])):
        if u['id'] in voice_ids:
            vid = voice_ids[u['id']]
            groups.setdefault(vid,{'voice_id':vid,'utterances':[]})['utterances'].append(
                {k:u[k] for k in ('id','start','end','text')})
    clip = copy.deepcopy(bundle)
    clip['voice_groups'] = list(groups.values())
    # Excluded turns remain in the bundle for Grove, not in AV identity inference.
    clip['utterances'] = [u for group in groups.values() for u in group['utterances']]
    return PreparedAssociation(bundle,graph,clip,voice_ids,exclusions)


def normalize_voice_prediction(payload, prepared):
    """Separate voice identity equivalence from per-turn on-screen visibility."""
    faces = {f['id'] for f in prepared.bundle['faces']}
    expected = {g['voice_id'] for g in prepared.clip['voice_groups']}
    links = payload['voice_links']
    found = [r['voice_id'] for r in links]
    if len(found)!=len(set(found)) or set(found)!=expected:
        raise ValueError('voice_links must cover every voice_id exactly once')
    linked = {r['voice_id']:r['face_id'] for r in links}
    segments = payload['segments']
    found = [r['utterance_id'] for r in segments]
    if len(found)!=len(set(found)) or set(found)!=set(prepared.voice_ids):
        raise ValueError('segments must cover every grouped utterance exactly once')
    for row in links+segments:
        if row['face_id'] is not None and row['face_id'] not in faces:
            raise ValueError('hallucinated face ID')
    visible = {r['utterance_id']:r['face_id'] for r in segments}
    for uid,face in visible.items():
        if face is not None and linked[prepared.voice_ids[uid]]!=face:
            raise ValueError('visible face contradicts its voice identity link')
    for vid,face in linked.items():
        if face is not None and not any(prepared.voice_ids[uid]==vid and fid==face for uid,fid in visible.items()):
            raise ValueError('voice-face link requires a visible supporting utterance in this episode')
    expanded = [dict(utterance_id=u['id'],start=u['start'],end=u['end'],
                     face_id=visible.get(u['id']),voice_id=prepared.voice_ids.get(u['id']))
                for u in prepared.bundle['utterances']]
    return links, expanded


def finish_association(prepared, payload, accept_links=False):
    links, predictions = normalize_voice_prediction(payload,prepared)
    bundle, graph = prepared.bundle, prepared.graph
    faces = {f['id']:f for f in bundle['faces']}
    linked = {r['voice_id']:r['face_id'] for r in links}
    for row in links:
        vid,face = row['voice_id'],row['face_id']
        if face is None:
            continue
        voice_anchor = graph.data['voice_nodes'][vid]['anchor_id']
        evidence_uids = [u['id'] for u in bundle['utterances'] if prepared.voice_ids.get(u['id'])==vid]
        link_id = 'av:'+bundle['id']+':'+vid+':'+face
        if link_id not in graph.data['links']:
            graph.propose(dict(id=link_id,left=voice_anchor,right=faces[face]['observation_id'],
                               method='audiovisual',evidence={'voice_id':vid,'utterance_ids':evidence_uids,
                               'source':bundle['source'],'policy_accept':accept_links}))
        if accept_links and graph.data['links'][link_id]['status']=='candidate':
            graph.review(link_id,'confirmed','experiment-policy','accept grouped voice association')
    exported = graph.export()
    mapping = {e['observation_id']:e['person_id'] for e in exported['events']}
    visible = {r['utterance_id']:r['face_id'] for r in predictions}
    refs = []
    for u in bundle['utterances']:
        vid = prepared.voice_ids.get(u['id'])
        voice = u.get('voice_observation_id') if vid else None
        evidence = [voice] if voice else []
        if vid and linked[vid] and accept_links:
            evidence.append(faces[linked[vid]]['observation_id'])
        refs.append({'utterance_id':u['id'],'start':bundle['time_origin']+u['start'],
                     'end':bundle['time_origin']+u['end'],'text':u['text'],
                     'character_id':mapping.get(voice),'voice_id':vid,'role':'speaker',
                     'visible_face_id':visible[u['id']],'evidence_ids':evidence,
                     'graph_revision':exported['revision']})
    return {'source':bundle['source'],'start':bundle['time_origin'],
            'end':bundle['time_origin']+bundle['duration'],'graph_revision':exported['revision'],
            'character_refs':refs,'graph':copy.deepcopy(graph.data),'associations':predictions,
            'voice_links':links,'voice_groups':copy.deepcopy(prepared.clip['voice_groups']),
            'excluded_voice_turns':prepared.exclusions}


def associate(bundle, backend, prior=None, accept_links=False, match_threshold=None):
    """Public orchestration: grouping is complete before the model is called."""
    prepared = prepare_association(bundle,prior,match_threshold)
    from .identity_state import attach_temporal_evidence
    if getattr(backend, 'mode', None) == 'm3_memorization':
        from .m3 import prepare_features, finish_memorization
        prepare_features(prepared)
        # Visual memories are required even with no eligible voice groups.
        raw = backend.infer(prepared.clip, 'm3_memorization')
        return attach_temporal_evidence(finish_memorization(prepared, raw, accept_links), bundle)
    if prepared.clip['voice_groups']:
        raw = backend.infer(prepared.clip,'voice_association')
    else:
        raw = {'voice_links':[],'segments':[],'inference_skipped':'no_eligible_voice_groups'}
    snapshot = finish_association(prepared,raw,accept_links)
    snapshot['raw'] = raw
    return attach_temporal_evidence(snapshot, bundle)


def resolve(bundle, predictions, prior=None, accept_links=False, match_threshold=None):
    """Compatibility for old fixture callers. Real CLI uses grouped `associate`.

    Inconsistent face assignments within one voice group are rejected, not averaged.
    """
    found = [r['utterance_id'] for r in predictions]
    if len(found)!=len(set(found)) or set(found)!={u['id'] for u in bundle['utterances']}:
        raise ValueError('association must cover each utterance exactly once')
    prepared = prepare_association(bundle,prior,match_threshold)
    grouped = {vid:set() for vid in prepared.voice_ids.values()}
    segments = []
    for row in predictions:
        vid = prepared.voice_ids.get(row['utterance_id'])
        if vid is None:
            if row['face_id'] is not None:
                raise ValueError('cannot bind an excluded utterance to a face')
            continue
        if row['face_id'] is not None: grouped[vid].add(row['face_id'])
        segments.append(row)
    if any(len(faces)>1 for faces in grouped.values()):
        raise ValueError('conflicting faces for the same voice ID')
    payload = {'voice_links':[dict(voice_id=vid,face_id=next(iter(faces)) if faces else None)
                             for vid,faces in grouped.items()], 'segments':segments}
    return finish_association(prepared,payload,accept_links)
