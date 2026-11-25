"""M3-style duration gate, PCM slicing and sequential multi-embedding voice nodes.

Matching/update rules follow ByteDance-Seed/m3-agent voice_processing.py and
videograph.py (Apache-2.0). Fractional seconds and seeded sampling are adaptations.
"""
import math
import random
import wave
from .faces import mean_similarity


def eligible_turns(turns, minimum=2.):
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError('positive finite minimum voice duration required')
    # M3 uses duration only. Do not selectively apply one frontend's overlap flag.
    return [u for u in turns if u['end']-u['start'] >= minimum]


def slice_pcm(source, destination, start, end):
    """Lossless slice from the common decoded WAV, never from AAC-reencoded video."""
    with wave.open(str(source),'rb') as reader:
        rate = reader.getframerate()
        if (rate,reader.getnchannels(),reader.getsampwidth()) != (16000,1,2):
            raise ValueError('expected mono 16kHz signed 16-bit PCM')
        if not all(math.isfinite(t) for t in (start,end)) or not 0 <= start < end <= reader.getnframes()/rate:
            raise ValueError('PCM window outside source audio')
        first,last = int(start*rate),int(end*rate)
        reader.setpos(first)
        data = reader.readframes(last-first)
        params = reader.getparams()
    with wave.open(str(destination),'wb') as writer:
        writer.setparams(params)
        writer.writeframes(data)


def group_voices(graph, observations, threshold=.6, max_embeddings=20, seed=0):
    """Process turns chronologically: match best node by ALL-pairs mean, else create.

    Local ASR speaker labels are not used. Voice nodes remain distinct from person
    components, even if audiovisual edges connect two voices to the same face.
    """
    if not math.isfinite(threshold) or not -1 <= threshold <= 1 or type(max_embeddings) is not int or max_embeddings < 1:
        raise ValueError('invalid voice matching configuration')
    obs = graph.data['observations']
    if len(observations) != len(set(observations)) or any(k not in obs or obs[k]['modality']!='voice' for k in observations):
        raise ValueError('expected unique existing voice observations')
    nodes = graph.data.setdefault('voice_nodes',{})
    membership = graph.data.setdefault('voice_membership',{})
    if not graph.data.get('voice_nodes_initialized'):
        # Migrate old snapshots using voice-only confirmed edges, never face equivalences.
        historical = [k for k,v in obs.items() if v['modality']=='voice' and k not in observations]
        parent = {k:k for k in historical}
        def root(k):
            while parent[k]!=k: k=parent[k]
            return k
        for link in graph.data['links'].values():
            a,b=link['left'],link['right']
            if link['status']=='confirmed' and a in parent and b in parent:
                if (obs[a]['model'],len(obs[a]['embedding']))==(obs[b]['model'],len(obs[b]['embedding'])):
                    parent[root(b)]=root(a)
        components={}
        for k in historical: components.setdefault(root(k),[]).append(k)
        for ids in components.values():
            anchor=ids[0]; vid='voice:'+anchor
            pool = ids if len(ids)<=max_embeddings else random.Random(str(seed)+':'+vid).sample(ids,max_embeddings)
            nodes[vid]={'anchor_id':anchor,'observation_ids':list(pool)}
            membership.update({k:vid for k in ids})
        graph.data['voice_nodes_initialized']=True
    for key in sorted(observations,key=lambda k:(obs[k]['start'],obs[k]['end'],k)):
        if key in membership:
            continue
        source=obs[key]
        space=source['model'],len(source['embedding'])
        candidates=[]
        for vid,node in nodes.items():
            ids=node['observation_ids']
            if any((obs[k]['model'],len(obs[k]['embedding']))!=space for k in ids):
                continue
            score=mean_similarity([source['embedding']],[obs[k]['embedding'] for k in ids])
            if score>=threshold: candidates.append((score,vid))
        if candidates:
            score,vid=max(candidates,key=lambda pair:pair[0])
            node=nodes[vid]
            link_id='m3_voice_reid:'+key+':'+node['anchor_id']
            graph.propose(dict(id=link_id,left=key,right=node['anchor_id'],method='m3_voice_reid',
                               evidence={'mean_cosine':score,'threshold':threshold,'voice_id':vid}))
            graph.review(link_id,'confirmed','m3-voice-policy','best node by mean similarity')
        else:
            vid='voice:'+key
            node=nodes.setdefault(vid,{'anchor_id':key,'observation_ids':[]})
        pool=node['observation_ids']+[key]
        if len(pool)>max_embeddings:
            pool=random.Random(str(seed)+':'+vid+':'+key).sample(pool,max_embeddings)
        node['observation_ids']=pool
        membership[key]=vid
        graph.data['audit'].append({'action':'voice_node_update','voice_id':vid,'observation_id':key,
                                    'observation_ids':list(pool)})
    return {k:membership[k] for k in observations}
