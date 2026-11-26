"""Versioned projection of the existing identity graph; no model inference."""
import copy
from .reused.identity import IdentityGraph
from .character_links import seal_state, fingerprint, bind


def review_identity(snapshot, changes):
    """Explicit audited edits; returns a new state without changing inference artifacts."""
    graph = IdentityGraph(snapshot['graph'])
    for link in changes.get('links', []): graph.propose(link)
    for review in changes.get('reviews', []):
        graph.review(review['link_id'], review['status'], review['reviewer'], review['reason'])
    names = changes.get('names', snapshot.get('identity_state', {}).get('names', {}))
    return dict(graph=graph.data, identity_state=export_state(graph.data, names))


def export_state(graph_data, names=None):
    graph = IdentityGraph(graph_data)
    view = graph.export()
    observations = {e['observation_id']: dict(character_id=e['person_id'],
        source=e['source'], modality=e['modality'], start=e['start'], end=e['end']) for e in view['events']}
    aliases = {alias: p['person_id'] for p in view['people'] for alias in [p['person_id']]+p['aliases']}
    state = seal_state(dict(schema_version=1, revision=view['revision'],
        graph_fingerprint=fingerprint(graph_data), observations=observations,
        aliases=aliases, names=copy.deepcopy(names or {})))
    for ids in state['names'].values(): bind(ids, state)
    return state


def attach_temporal_evidence(snapshot, bundle):
    state = export_state(snapshot['graph'])
    events = copy.deepcopy(bundle.get('face_occurrences', []))
    for u in bundle['utterances']:
        ref = next(r for r in snapshot['character_refs'] if r['utterance_id']==u['id'])
        ids = [u['voice_observation_id']] if ref.get('voice_id') else []
        ref.update(bind(ids, state), source=bundle['source'])
        events.append(dict(id=bundle['id']+':speech:'+u['id'], source=bundle['source'], role='speaker',
            start=ref['start'], end=ref['end'], utterance_id=u['id'], text=u['text'],
            identity_observation_ids=ids, evidence_ids=list(ids)))
    snapshot.update(identity_state=state, temporal_evidence=events,
        temporal_evidence_version=1, face_timeline_complete=bundle.get('face_timeline_version')==1)
    evidence = snapshot.get('identity_evidence')
    if evidence:
        for tag, row in evidence['feature_mapping'].items():
            # Current observations determine the feature; historical anchor remains provenance only.
            ids = snapshot['feature_registry'][tag]['observation_ids']
            row.update(bind(ids, state))
        evidence.update(source=bundle['source'], identity_state_id=state['state_id'])
    return snapshot
