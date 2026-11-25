import copy
import unittest
import numpy as np
from lab.faces import select_faces, mean_similarity, link_face_groups
from lab.reused.identity import IdentityGraph
from lab.association import resolve
from tests.fixtures import bundle, predictions


def detections():
    rng = np.random.default_rng(17)
    rows = []
    for label, center in enumerate(([1.,0.,0.,0.],[-1.,0.,0.,0.])):
        for i in range(40):
            v = np.array(center)+rng.normal(0,.015,4)
            v /= np.linalg.norm(v)
            rows.append(dict(frame_id=label*40+i, face_emb=v.tolist(), cluster_id=-1,
                extra_data=dict(face_type='ortho', face_detection_score=.9+i*.001,face_quality_score=25+i*.1)))
    return rows


def observe(graph,key,v,model='face-v1'):
    graph.observe(dict(id=key,modality='face',source='clip',start=0,end=1,model=model,embedding=v))


def face(ids):
    return dict(id=ids[0],observation_id=ids[0],observation_ids=ids)


class FaceClusteringTests(unittest.TestCase):
    def test_real_hdbscan_and_best_representatives(self):
        labeled, groups = select_faces(detections(), {})
        self.assertEqual(len(groups),2)
        self.assertEqual({tuple(r['frame_id'] for r in rows) for rows in groups.values()},
                         {(39,38,37),(79,78,77)})
        self.assertEqual(len(labeled),80)

    def test_quality_gate_and_no_face_or_too_few_samples(self):
        self.assertEqual(select_faces([],{}),([],{}))
        labeled, groups = select_faces(detections()[:19],{})
        self.assertEqual(groups,{})
        self.assertTrue(all(row['cluster_id']==-1 for row in labeled))
        rows=detections()
        for r in rows:
            r['extra_data']['face_quality_score']=19
        self.assertEqual(select_faces(rows,{})[1],{})

    def test_second_quality_filter_strict_threshold(self):
        rows=detections()
        # They pass pre-clustering (>=.8, >=20), but fail final selection (> .85, >22).
        for row in rows:
            row['extra_data']['face_quality_score']=22
        labeled, groups=select_faces(rows,{})
        self.assertEqual(len({r['cluster_id'] for r in labeled}),2)
        self.assertEqual(groups,{})

    def test_isolated_low_quality_detection_stays_noise(self):
        rows=detections();bad=copy.deepcopy(rows[0]);bad['extra_data']['face_detection_score']=.7
        rows.append(bad)
        labeled, groups=select_faces(rows,{})
        self.assertEqual(labeled[-1]['cluster_id'],-1)
        self.assertEqual(len(groups),2)

    def test_average_not_nearest_neighbor(self):
        self.assertEqual(mean_similarity([[1,0]],[[1,0],[0,1]]),.5)
        graph=IdentityGraph()
        observe(graph,'a',[1,0]);observe(graph,'b',[0,1])
        link_face_groups(graph,[face(['a','b'])],threshold=.75)
        observe(graph,'c',[1,0])
        link_face_groups(graph,[face(['c'])],threshold=.75)
        self.assertEqual(len(graph.people()),2)

    def test_representatives_merge_and_historical_match(self):
        graph=IdentityGraph()
        for k in ['a','b','c']: observe(graph,k,[1,0])
        link_face_groups(graph,[face(['a','b','c'])])
        observe(graph,'later',[1,0])
        link_face_groups(graph,[face(['later'])])
        self.assertEqual(len(graph.people()),1)
        self.assertEqual(graph.people()[0]['person_id'],'person:a')
        self.assertEqual(len(graph.data['face_nodes']),1)
        before=copy.deepcopy(graph.data)
        link_face_groups(graph,[face(['later'])])
        self.assertEqual(graph.data,before)

    def test_different_model_not_matched(self):
        graph=IdentityGraph();observe(graph,'a',[1,0]);link_face_groups(graph,[face(['a'])])
        observe(graph,'b',[1,0],model='new-model');link_face_groups(graph,[face(['b'])])
        self.assertEqual(len(graph.people()),2)

    def test_memory_cap_and_reload_are_reproducible(self):
        graph=IdentityGraph()
        for i in range(3): observe(graph,str(i),[1,0])
        ids=[str(i) for i in range(15)]
        link_face_groups(graph,[face(ids[:3])])
        for i in range(3,15): observe(graph,str(i),[1,0])
        copy_graph=IdentityGraph(graph.data)
        for g in [graph,copy_graph]: link_face_groups(g,[face(ids[3:])],seed=42)
        self.assertEqual(graph.data,copy_graph.data)
        self.assertEqual(len(graph.data['face_nodes']['0']['observation_ids']),10)
        self.assertEqual(len(graph.people()[0]['faces']),15)

    def test_multiple_face_representatives_feed_character_mapping(self):
        b=bundle();o=copy.deepcopy(b['observations'][0]);o['id']='ep1:f1:second'
        b['observations'].append(o)
        b['faces'][0]['observation_ids']=['ep1:f1','ep1:f1:second']
        snapshot=resolve(b,predictions(),accept_links=True)
        self.assertEqual(snapshot['character_refs'][0]['character_id'],'person:ep1:f1')
        self.assertEqual(len(IdentityGraph(snapshot['graph']).people()),1)

    def test_duplicate_observation_across_groups_rejected(self):
        g=IdentityGraph();observe(g,'a',[1,0])
        with self.assertRaises(ValueError): link_face_groups(g,[face(['a']),face(['a'])])
