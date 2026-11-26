from tests.character_fixtures import timeline_snapshot
from lab.visual_grounding import apply_locations


def grounded_snapshot():
    snapshot=timeline_snapshot()
    memory=snapshot['identity_evidence']['memories'][0]
    memory['text']='<face_0> lifts a cup.'
    frames=[dict(id='f4',time=4.),dict(id='f5',time=5.),dict(id='f8',time=8.)]
    payload={'locations':[
        dict(memory_id='description:0',reason='synthetic visible action',intervals=[dict(start=4.,end=6.)],frame_ids=['f4','f5']),
        dict(memory_id='description:1',reason='speech only',intervals=[],frame_ids=[])]}
    return apply_locations(snapshot,payload,frames,dict(method='synthetic',episode_id='ep1',reviewed=False))


async def written_evidence_graph(directory):
    import json
    from pathlib import Path
    from lab.grove import write_episode,load_grove
    from tests.fixtures import episode
    load_grove()
    from hypervideo.structure import VideoHypergraph
    from hypervideo.modules.character_projection import rebuild_characters
    snapshot=grounded_snapshot()
    async def llm(prompt):
        if '"facts"' in prompt:
            return json.dumps({'facts':[
                dict(content='<voice_0> says hello.',utterance_ids=['u1']),
                dict(content='An unidentified speaker says yes.',utterance_ids=['u2']),
                dict(content='<face_0> lifts a cup.',memory_ids=['description:0']),
                dict(content='<voice_0> speaks again.',memory_ids=['description:1'])]})
        return json.dumps(dict(summary='<face_0> lifts a cup.',subject='Conversation'))
    hg=VideoHypergraph.from_dict(await write_episode(episode(),snapshot,llm,Path(directory)/'initial.json'))
    for cid,start,end in [('early',86400,86405),('late',86405,86410)]:
        hg.add_visual_clip_node(cid,1,start,end,'',cid,'ep1',2)
    return rebuild_characters(hg,snapshot['identity_state']),snapshot
