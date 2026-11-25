import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from lab.grove import load_grove, write_episode
from lab.association import resolve
from tests.fixtures import bundle, predictions, episode
load_grove()
from hypervideo.types import VideoEpisode, VideoFact
from hypervideo.structure import VideoHypergraph, VideoEpisodeNode, VideoFactNode
from hypervideo.modules.character_identity import SnapshotProvider, facts_from_response


class GroveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.snapshot=resolve(bundle(),predictions(),accept_links=True)
        self.ep=VideoEpisode.from_dict(episode())
        self.ep.character_refs=await SnapshotProvider([self.snapshot],'recording')(86400,86410)

    async def test_snapshot_requires_correct_time_and_source(self):
        for source,a,b in [('other',86400,86410),('recording',86400,86411)]:
            with self.assertRaises(ValueError):
                await SnapshotProvider([self.snapshot],source)(a,b)

    async def test_revision_mismatch(self):
        s=copy.deepcopy(self.snapshot); s['character_refs'][0]['graph_revision']+=1
        with self.assertRaises(ValueError):
            await SnapshotProvider([s],'recording')(86400,86410)

    async def test_facts_cite_only_their_utterance(self):
        facts=facts_from_response({'facts':[
            dict(content='A person says hello',utterance_ids=['u1']),
            dict(content='Someone nods',utterance_ids=[]),
            dict(content='An unknown person says yes',utterance_ids=['u2'])]},self.ep)
        self.assertEqual(facts[0].character_refs[0]['character_id'],'person:ep1:f1')
        self.assertEqual(facts[1].character_refs,[])
        self.assertIsNone(facts[2].character_refs[0]['character_id'])
        self.assertEqual(facts[0].character_refs[0]['role'],'speaker')

    async def test_hallucinated_evidence_rejected(self):
        with self.assertRaises(ValueError):
            facts_from_response({'facts':[dict(content='bad',utterance_ids=['imaginary'])]},self.ep)

    async def test_graph_and_dataclass_roundtrip(self):
        fact=facts_from_response({'facts':[dict(content='hello',utterance_ids=['u1'])]},self.ep)[0]
        self.assertEqual(VideoFact.from_dict(fact.to_dict()).character_refs,fact.character_refs)
        self.assertEqual(VideoEpisodeNode.from_video_episode(self.ep).to_video_episode().character_refs,self.ep.character_refs)
        self.assertEqual(VideoFactNode.from_video_fact(fact).to_video_fact().character_refs,fact.character_refs)
        graph=VideoHypergraph(); graph.add_episode_node(self.ep); graph.add_fact_node(fact)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'hypergraph.json'; graph.save(str(path))
            restored=VideoHypergraph.load(str(path))
            self.assertEqual(restored.to_dict(),graph.to_dict())

    async def test_old_graph_compatible(self):
        self.assertEqual(VideoFactNode(id='old',content='old fact').character_refs,[])
        self.assertEqual(VideoEpisode.from_dict(episode()).character_refs,[])

    async def test_isolated_writer_order_with_test_llm(self):
        prompts=[]
        async def llm(prompt):
            prompts.append(prompt)
            self.assertIn('person:ep1:f1',prompt)
            if len(prompts)==1:
                return json.dumps(dict(summary='person:ep1:f1 says hello',subject='Conversation'))
            return json.dumps({'facts':[dict(content='person:ep1:f1 says hello',utterance_ids=['u1'])]})
        with tempfile.TemporaryDirectory() as d:
            graph=await write_episode(episode(),self.snapshot,llm,Path(d)/'graph.json')
            fact=next(iter(graph['facts'].values()))
            self.assertEqual(fact['character_refs'][0]['character_id'],'person:ep1:f1')
            self.assertEqual(len(prompts),2)
