"""Pure, source-qualified fact→evidence→clip contracts; copied unchanged into Grove."""
import copy
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_interval(start, end, *, point=False):
    if not (number(start) and number(end) and 0 <= start <= end) or (start==end) != point:
        raise ValueError('invalid evidence interval/point')


def validate_grounding(grounding, memory, scope):
    if grounding.get('memory_sha256') != digest({k:memory[k] for k in ('id','kind','text','feature_ids')}):
        raise ValueError('visual grounding belongs to different memory text/features')
    if grounding.get('scope') != {k:scope[k] for k in ('source','start','end')} or grounding.get('schema_version') != 1:
        raise ValueError('visual grounding source/episode mismatch')
    anchors = grounding['anchors']
    if grounding['status'] not in ('localized','unlocalized') or bool(anchors) != (grounding['status']=='localized'):
        raise ValueError('visual grounding status/anchors disagree')
    if memory['kind'] != 'description' and anchors:
        raise ValueError('inference is not a directly observed visual event')
    if not grounding.get('provenance') or not grounding.get('reason'):
        raise ValueError('grounding requires reason and provenance')
    for anchor in anchors:
        check_interval(anchor['start'], anchor['end'], point=anchor.get('sample',False))
        if not scope['start'] <= anchor['start'] <= anchor['end'] <= scope['end']:
            raise ValueError('visual grounding outside episode')
        if anchor.get('sample') and (not anchor.get('frame_id') or anchor['start']==scope['end']):
            raise ValueError('frame evidence requires an in-episode frame ID')
        frames=anchor.get('frame_evidence', [])
        expected=anchor.get('frame_ids',[anchor.get('frame_id')])
        if (not frames or len({f['frame_id'] for f in frames})!=len(frames) or
                [f['frame_id'] for f in frames]!=expected):
            raise ValueError('visual anchor requires its original supporting frames')
        for frame in frames:
            t=frame['time']
            if not number(t) or (t!=anchor['start'] if anchor.get('sample') else not anchor['start']<=t<anchor['end']):
                raise ValueError('supporting frame outside visual anchor')


def select_evidence(row, utterances, memories, scope):
    """Writer selects IDs; only trusted upstream evidence supplies time and source."""
    result = []
    for kind, field, items in [('speech','utterance_ids',utterances),('visual','memory_ids',memories)]:
        ids = row.get(field, [])
        if not isinstance(ids,list) or len(set(ids))!=len(ids) or any(i not in items for i in ids):
            raise ValueError('invalid fact evidence selection')
        for key in ids:
            item = items[key]
            base = dict(kind=kind, source=scope['source'], episode_id=scope['episode_id'],
                        text=item['text'], **{('utterance_id' if kind=='speech' else 'memory_id'):key})
            anchors = []
            reason = 'missing_visual_localization'
            provenance = {}
            if kind=='speech':
                check_interval(item['start'],item['end'])
                if not scope['start'] <= item['start'] < item['end'] <= scope['end']:
                    raise ValueError('utterance outside episode')
                if item.get('source',scope['source']) != scope['source']:
                    raise ValueError('utterance source mismatch')
                if scope['source']:
                    anchors=[dict(start=item['start'],end=item['end'],sample=False)]
                reason='source_missing' if not anchors else 'utterance_time'
                provenance={'method':'utterance_time','semantic_scope':'spoken_statement'}
            elif item.get('grounding'):
                g=item['grounding']; validate_grounding(g,item,scope)
                anchors=g['anchors']; reason=g['reason']; provenance=g['provenance']
            elif item['kind']=='conclusion':
                base['kind']='inference'; reason='inference_not_direct_observation'
            for index, anchor in enumerate(anchors or [None]):
                ref=dict(base, id=digest([scope,kind,key,index]), status='localized' if anchor else 'unlocalized',
                         reason=reason, provenance=copy.deepcopy(provenance))
                if anchor: ref.update(copy.deepcopy(anchor))
                result.append(ref)
    return result


def align_evidence(refs, clips):
    """Temporal intersection proves where to inspect evidence, not semantic entailment."""
    clips=sorted(copy.deepcopy(clips), key=lambda c:c['id'])
    if len({c['id'] for c in clips})!=len(clips): raise ValueError('duplicate clip ID')
    for clip in clips:
        check_interval(clip['start'],clip['end'])
        if not clip.get('source'): raise ValueError('clip requires source')
    edges, coverage, seen = [], [], set()
    for ref in refs:
        if ref['id'] in seen: raise ValueError('duplicate evidence ID')
        seen.add(ref['id'])
        if ref['status']=='unlocalized':
            coverage.append(dict(evidence_id=ref['id'],status='unlocalized',coverage_ratio=None))
            continue
        if ref['status']!='localized' or not ref.get('source'):
            raise ValueError('invalid localized evidence')
        point=ref.get('sample',False)
        check_interval(ref['start'],ref['end'],point=point)
        intersections=[]
        for clip in clips:
            if clip['source']!=ref['source']: continue
            a,b=max(ref['start'],clip['start']),min(ref['end'],clip['end'])
            if (point and not clip['start']<=ref['start']<clip['end']) or (not point and a>=b): continue
            intersections.append((a,b))
            edges.append(dict(evidence_id=ref['id'],clip_id=clip['id'],kind=ref['kind'],source=ref['source'],
                start=a,end=b,sample=point,frame_id=ref.get('frame_id'),
                frame_ids=[f['frame_id'] for f in ref.get('frame_evidence',[]) if clip['start']<=f['time']<clip['end']],
                relation='temporal_overlap'))
        if point:
            ratio=float(bool(intersections))
        else:
            total=0.; cursor=ref['start']
            for a,b in sorted(intersections):
                total+=max(0,b-max(a,cursor)); cursor=max(cursor,b)
            ratio=min(1.,total/(ref['end']-ref['start']))
        coverage.append(dict(evidence_id=ref['id'],coverage_ratio=ratio,
            status='covered' if ratio>=1-1e-9 else ('partial' if ratio else 'uncovered')))
    status=('no_evidence' if not refs else 'complete' if all(c['status']=='covered' for c in coverage)
            else 'partial' if edges else 'unlocalized' if all(c['status']=='unlocalized' for c in coverage)
            else 'uncovered')
    return dict(clip_evidence=edges,evidence_alignment=dict(schema_version=1,status=status,coverage=coverage,
                inputs_sha256=digest(dict(evidence_refs=refs,clips=clips))))
