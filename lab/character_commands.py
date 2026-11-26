"""CPU-only replay/projection commands, usable with real GPU-generated observations."""
import json
import os
import tempfile
from pathlib import Path
from .common import read
from .character_links import project_clip, build_index, lookup
from .identity_state import review_identity


def add_commands(sub):
    p = sub.add_parser('identity-review')
    for name in ('snapshot', 'changes', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('project-clips')
    for name in ('state', 'snapshot', 'clips', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('grove-project')
    for name in ('graph', 'state', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--snapshot', type=Path, action='append', default=[])
    p = sub.add_parser('character-lookup')
    for name in ('graph', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--character', required=True, action='append')
    p.add_argument('--role', action='append', choices=['visible', 'speaker', 'referenced'])


def run(args):
    if args.out.exists(): raise FileExistsError('choose a new output; original graphs are never overwritten')
    if args.command == 'identity-review':
        result = review_identity(read(args.snapshot), read(args.changes))
    elif args.command == 'project-clips':
        state = read(args.state)['identity_state']
        snapshot, clips = read(args.snapshot), read(args.clips)
        if not snapshot.get('face_timeline_complete'):
            raise ValueError('rerun perceive/associate for a complete face timeline')
        records = []
        for clip in clips:
            records.append(dict(clip, node_type='visual_clip',
                character_refs=project_clip(clip, snapshot['temporal_evidence'], state)))
        result = dict(clips=records, character_index=build_index(records, state), identity_state=state)
    else:
        from .grove import load_grove
        load_grove()
        from hypervideo.structure import VideoHypergraph
        from hypervideo.modules.character_projection import rebuild_characters, character_hits
        graph = VideoHypergraph.from_dict(read(args.graph))
        if args.command == 'grove-project':
            result = rebuild_characters(graph, read(args.state)['identity_state'],
                                        [read(p) for p in args.snapshot]).to_dict()
            if VideoHypergraph.from_dict(result).to_dict() != result:
                raise AssertionError('graph round-trip changed character projection')
        else:
            result = dict(matches=character_hits(graph, args.character, args.role),
                identity_state_id=graph.character_catalog['state']['state_id'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a fully validated result. Separate output permits comparison/rollback.
    fd, temp = tempfile.mkstemp(prefix='.character-', dir=args.out.parent)
    try:
        with os.fdopen(fd, 'w') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
        os.link(temp, args.out)  # atomic create, fails if another process claimed the path
    finally:
        os.unlink(temp)
