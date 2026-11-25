"""Synthetic index fixtures; real Grove nodes and real NumPy cosine calculations."""
import json
from types import SimpleNamespace
import numpy as np
from lab.grove import load_grove
load_grove()
from hypervideo.structure import VideoHypergraph, VideoEpisodeNode, VideoFactNode, VideoTopicNode
from hypervideo.mmlifelong_week_retrieval_v2 import IndexedHypergraph
from hypervideo.visual_retriever import VisualRetriever


class FixtureEncoder:
    backbone, dim = 'fixture-visual', 2
    def encode_text(self, query):
        return np.array([1., 0.], dtype=np.float32)


class FixtureTextIndex:
    def __init__(self):
        self.hits = dict(topic=[('t1', .8)], episode=[('ep1', .9), ('ep2', .8), ('ep3', .7)],
                         fact=[('f1', .0001), ('f2', .00009), ('f3', .00008)])
        self.calls = []

    async def search(self, query, doc_type, top_n=10, **kwargs):
        self.calls.append((doc_type, None))
        return self.hits[doc_type][:top_n]

    async def search_within(self, query, candidate_ids, doc_type, top_n=10, **kwargs):
        self.calls.append((doc_type, set(candidate_ids)))
        return [row for row in self.hits[doc_type] if row[0] in candidate_ids][:top_n]


def graph():
    hg = VideoHypergraph()
    for eid, day, start, end in [('ep1', 1, 0, 10), ('ep2', 1, 10, 20), ('ep3', 2, 0, 10)]:
        hg.episodes[eid] = VideoEpisodeNode(id=eid, day=day, start_seconds=day*86400+start,
            end_seconds=day*86400+end, time_span=f'{day}-00:00:{start:02d}-00:00:{end:02d}',
            summary='Fixture summary', subject='Fixture event')
    for fid, eid in [('f1', 'ep1'), ('f2', 'ep1'), ('f3', 'ep3')]:
        hg.facts[fid] = VideoFactNode(id=fid, content='Synthetic textual fact '+fid, episode_ids=[eid])
    for eid, ids in [('ep1', ['f1', 'f2']), ('ep3', ['f3'])]:
        hg.add_fact_hyperedge('fh:'+eid, {fid:'core' for fid in ids}, None, eid)
        hg.update_episode_fact_hyperedge_link(eid, 'fh:'+eid)
    hg.topics['t1'] = VideoTopicNode(id='t1', title='Fixture topic', episode_ids=list(hg.episodes))
    hg.add_episode_hyperedge('eh1', {eid:'core' for eid in hg.episodes}, None, 't1')
    hg.update_topic_episode_hyperedge_link('t1', 'eh1')
    for cid, eid in [('c1', 'ep1'), ('c2', 'ep2'), ('c3', 'ep3')]:
        ep = hg.episodes[eid]
        hg.add_visual_clip_node(cid, ep.day, ep.start_seconds, ep.end_seconds,
            ep.time_span, 'Synthetic visual caption '+cid, eid, 2)
        hg.add_episode_window_hyperedge('wh:'+eid, eid, [cid])
    return hg


def write_index(path, *, metadata=None, embeddings=None):
    if metadata is None:
        metadata = [{key:cid, 'episode_id':eid, 'day':day,
                     'start_sec_in_day':start, 'end_sec_in_day':end}
                    for key,cid,eid,day,start,end in [('visual_clip_id','c1','ep1',1,0,10),
                        ('id','c2','ep2',1,10,20), ('clip_id','c3','ep3',2,0,10)]]
    if embeddings is None:
        embeddings = [[.6, .8], [.8, .6], [1., 0.]]
    np.savez_compressed(path, embeddings=np.asarray(embeddings, dtype=np.float32),
        meta=np.array([json.dumps(row) for row in metadata]), backbone=np.array(['fixture-visual']), dim=np.array([2]))
    return VisualRetriever(str(path), encoder=FixtureEncoder())


def indexed(hg, dual):
    async def ensure(): pass
    vg = SimpleNamespace(hypergraph_builder=SimpleNamespace(hypergraph=hg), dual_index=dual,
                         ensure_retrieval_index=ensure)
    return IndexedHypergraph(vg)
