"""Synthetic identity evidence, never a recognition accuracy benchmark."""
import json
from pathlib import Path
from lab.association import associate
from lab.character_links import face_occurrences
from lab.grove import load_grove, write_episode
from lab.identity_state import export_state, review_identity
from tests.fixtures import episode
from tests.modules.test_m3 import JointBackend
from tests.modules.test_voices import repeated_bundle


def timeline_snapshot():
    data = repeated_bundle()
    data['faces'][0]['cluster_id'] = 0
    rows = [dict(timestamp=t, cluster_id=0 if t!=7 else -1, frame_id=str(t),
                 bounding_box=[0,0,40,40], extra_data={'face_detection_score':.95})
            for t in [0,1,2,4,7,8,9]]
    data['face_occurrences'] = face_occurrences(rows, data['faces'], episode_id='ep1',
        source='recording', origin=86400., duration=10.)
    data['face_timeline_version'] = 1
    return associate(data, JointBackend(), accept_links=True)


def revoke(snapshot, *, split_voice=False):
    links = snapshot['graph']['links']
    targets = [lid for lid,row in links.items() if row['left'].startswith('ep1:f') != row['right'].startswith('ep1:f')]
    if split_voice:
        targets += [lid for lid,row in links.items() if row['left'].startswith('ep1:v') and row['right'].startswith('ep1:v')]
    return review_identity(snapshot, dict(reviews=[dict(link_id=lid, status='revoked',
        reviewer='test', reason='synthetic wrong match') for lid in targets]))


async def written_graph(directory, snapshot=None):
    load_grove()
    from hypervideo.structure import VideoHypergraph
    from hypervideo.modules.character_projection import rebuild_characters
    snapshot = snapshot or timeline_snapshot()
    async def llm(prompt):
        if '"facts"' in prompt:
            return json.dumps({'facts':[
                dict(content='<voice_0> says hello.', utterance_ids=['u1']),
                dict(content='<face_0> waves.', memory_ids=['description:0']),
                dict(content='<person:ep1:f1> says again.', utterance_ids=['u3'])]})
        return json.dumps(dict(summary='<face_0> watches <voice_0>.', subject='<voice_0> speaks'))
    graph = VideoHypergraph.from_dict(await write_episode(episode(), snapshot, llm, Path(directory)/'raw.json'))
    for cid,start,end in [('early',86400,86405),('late',86405,86410)]:
        graph.add_visual_clip_node(cid,1,start,end,'',cid,'ep1',2)
    return rebuild_characters(graph,snapshot['identity_state']), snapshot
