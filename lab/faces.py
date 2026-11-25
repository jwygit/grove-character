"""M3 face clustering/selection and face-node matching adapted to evidence graphs.

Clustering is the unchanged upstream function. Selection follows face_processing.py;
matching follows videograph.search_img_nodes/update_node (Apache-2.0, see provenance).
"""
import math
import random


def select_faces(detections, config):
    from .reused.m3_face_clustering import cluster_faces
    if not detections:
        return [], {}
    minimum = config.get('face_min_cluster_size', 20)
    limit = config.get('max_faces_per_character', 3)
    if type(minimum) is not int or minimum < 2 or type(limit) is not int or limit < 1:
        raise ValueError('invalid face cluster size or representative count')
    labeled = cluster_faces(detections, min_cluster_size=minimum)
    groups = {}
    for face in labeled:
        extra = face['extra_data']
        if (face['cluster_id'] >= 0 and
                float(extra['face_detection_score']) > config.get('face_detection_score_threshold', .85) and
                float(extra['face_quality_score']) > config.get('face_quality_score_threshold', 22)):
            groups.setdefault(int(face['cluster_id']), []).append(face)
    for label, members in groups.items():
        groups[label] = sorted(members, key=lambda f: (
            float(f['extra_data']['face_detection_score']),
            float(f['extra_data']['face_quality_score'])), reverse=True)[:limit]
    return labeled, groups


def mean_similarity(query, references):
    """All-pairs cosine mean, not maximum or cosine between averaged vectors."""
    if not query or not references:
        raise ValueError('nonempty embedding groups required')
    def unit(v):
        norm = math.hypot(*v)
        if not norm or not math.isfinite(norm):
            raise ValueError('invalid face embedding')
        return [x/norm for x in v]
    vectors = [unit(v) for v in query+references]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError('embedding dimension mismatch')
    q, r = vectors[:len(query)], vectors[len(query):]
    return sum(sum(x*y for x,y in zip(a,b)) for a in q for b in r)/(len(q)*len(r))


def link_face_groups(graph, faces, threshold=.3, max_embeddings=10, seed=0):
    """Keep multiple representatives in a face node, then link its observations.

    M3 caps image embeddings using random.sample. We seed that sampling by node
    and update IDs for reproducibility. Full observations remain in the audit graph.
    Historical nodes are separate from cross-modal person components, as in M3.
    """
    if not -1 <= threshold <= 1 or type(max_embeddings) is not int or max_embeddings < 1:
        raise ValueError('invalid face matching configuration')
    obs = graph.data['observations']
    groups = []
    all_ids = set()
    for face in faces:
        ids = face.get('observation_ids', [face['observation_id']])
        if not ids or face['observation_id'] != ids[0] or len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError('invalid/duplicate face group references')
        if any(key not in obs or obs[key]['modality'] != 'face' for key in ids):
            raise ValueError('invalid face observation')
        spaces = {(obs[key]['model'], len(obs[key]['embedding'])) for key in ids}
        if len(spaces) != 1:
            raise ValueError('a face group cannot mix embedding models')
        all_ids.update(ids)
        groups.append(ids)
    nodes = graph.data.setdefault('face_nodes', {})
    # Read old snapshots without face-node metadata. Do not index current observations twice.
    indexed = {key for node in nodes.values() for key in node['observation_ids']} | {n['anchor_id'] for n in nodes.values()}
    for person in graph.people():
        legacy = [key for key in person['faces'] if key not in all_ids and key not in indexed]
        # All old observations remain auditable, but capped-out samples must not become new nodes.
        if 'face_nodes_initialized' in graph.data:
            break
        by_space = {}
        for key in legacy:
            by_space.setdefault((obs[key]['model'],len(obs[key]['embedding'])), []).append(key)
        for ids in by_space.values():
            nodes[ids[0]] = {'anchor_id':ids[0], 'observation_ids':ids[:max_embeddings]}
    graph.data['face_nodes_initialized'] = True
    processed = graph.data.setdefault('face_groups', {})

    def connect(left, right, method, evidence):
        if left == right:
            return
        link_id = method+':'+left+':'+right
        if link_id not in graph.data['links']:
            graph.propose(dict(id=link_id,left=left,right=right,method=method,evidence=evidence))
            graph.review(link_id,'confirmed','m3-face-policy',method)

    for ids in groups:
        group_key = ids[0]
        if group_key in processed:
            if processed[group_key]['observation_ids'] != ids:
                raise ValueError('face group changed; use a new episode/run ID')
            continue
        space = obs[ids[0]]['model'], len(obs[ids[0]]['embedding'])
        matches = []
        for key, node in nodes.items():
            refs = node['observation_ids']
            if any((obs[x]['model'],len(obs[x]['embedding'])) != space for x in refs):
                continue
            score = mean_similarity([obs[x]['embedding'] for x in ids], [obs[x]['embedding'] for x in refs])
            if score >= threshold:
                matches.append((score,key))
        if matches:
            score, key = max(matches, key=lambda pair:pair[0])
            node = nodes[key]
            connect(ids[0],node['anchor_id'],'m3_face_reid',{'mean_cosine':score,'threshold':threshold})
        else:
            key = ids[0]
            node = nodes.setdefault(key, {'anchor_id':key,'observation_ids':[]})
        for other in ids[1:]:
            connect(ids[0],other,'m3_face_cluster',{'representatives':ids})
        combined = list(dict.fromkeys(node['observation_ids']+ids))
        if len(combined)>max_embeddings:
            combined = random.Random(str(seed)+':'+key+':'+','.join(ids)).sample(combined,max_embeddings)
        if node['observation_ids'] != combined:
            node['observation_ids'] = combined
            graph.data['audit'].append({'action':'face_node_update','id':key,'observation_ids':list(combined)})
        processed[group_key] = {'anchor_id':node['anchor_id'],'observation_ids':list(ids)}
