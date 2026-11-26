"""Real dynamic provider/association/grounding/writer; synthetic GPU predictions.

Only face/speaker embedding inference and M3 generation are substituted at model
boundaries. No Grove module, graph, association algorithm or index is replaced.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from lab.common import audio, digest as file_digest, write
from lab.full_loop import run_video
from lab.grove import load_grove
from lab.stream_identity import identity_provider_factory
from lab.perception import episode_turns
from lab.character_links import face_occurrences
from tests.http_models import model_server, config, create_video


class InferenceFixture:
    mode = "m3_memorization"
    def __init__(self):
        self.scopes = []
        self.inferences = 0

    def infer(self, clip, mode):
        self.inferences += 1
        assert mode == self.mode
        return dict(video_description=["<face_0> holds a cup."],
                    high_level_conclusions=["Equivalence: <face_0>, <voice_0>"])

    def perceive(self, video, speech, start, end, origin, source, episode_id, config, out, source_wav=None):
        self.scopes.append((start, end))
        out.mkdir()
        clip, image = out / "episode.mp4", out / "face.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video), "-t", str(end-start),
                        "-c:v", "libx264", "-c:a", "aac", "-n", str(clip)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "1", "-n", str(image)], check=True, capture_output=True)
        fid = episode_id + ":face"
        faces = [dict(id=fid, observation_id=fid, image=str(image), cluster_id=0)]
        turns = episode_turns(speech, start, end, source)
        observations = [dict(id=fid, modality="face", source=source, start=origin+start+.5,
                             end=origin+start+.5, model="synthetic-face-v1", embedding=[1.,0.])]
        for u in turns:
            if u["end"]-u["start"] >= 2:
                uid = episode_id + ":" + u["id"]
                u["voice_observation_id"] = uid
                observations.append(dict(id=uid, modality="voice", source=source,
                    start=origin+start+u["start"], end=origin+start+u["end"], model="synthetic-voice-v1", embedding=[1.,0.]))
        rows = [dict(timestamp=t, cluster_id=0, frame_id=str(t), bounding_box=[0,0,20,20],
                     extra_data={"face_detection_score":.95}) for t in (.5,1.5,2.5,3.5)]
        return dict(id=episode_id, source=source, time_origin=origin+start, duration=end-start,
                    faces=faces, utterances=turns, observations=observations,
                    face_occurrences=face_occurrences(rows,faces,episode_id=episode_id,source=source,
                        origin=origin+start,duration=end-start), face_timeline_version=1,
                    video=str(clip), episode_video_sha256=file_digest(clip))


class CharacterLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_boundary_identity_and_spoken_visual_evidence_survive_restart(self):
        load_grove()
        with tempfile.TemporaryDirectory() as directory, model_server() as (fixture, base):
            root = Path(directory)
            video = root / "colors.mp4"
            create_video(video)
            wav = root / "source.wav"
            audio(video, wav)
            speech_path = root / "speech.json"
            write(speech_path, dict(audio_sha256=file_digest(wav), segments=[
                dict(id=f"u{i}", start=i*4+.5, end=i*4+3, text="hello") for i in range(3)], raw={}))
            cfg = config(base)
            character_config = dict(speech_input=str(speech_path), perception={}, association={}, accept_links=True,
                grounding=dict(endpoint=base+"/v1/chat/completions", model="synthetic-grounding",
                               model_revision="fixture-v1", sample_fps=1, max_frames=20))
            cfg["characters"] = character_config
            output = root / "run"
            output.mkdir()
            models = InferenceFixture()
            factory = identity_provider_factory(video, output, character_config, "camera", 86400,
                perception_extract=models.perceive, association_backend=models)
            memory, _ = await run_video(video,cfg,output,"camera",stop_after_windows=2,identity_factory=factory)
            self.assertEqual(models.scopes, [])  # No recognition before any boundary.
            memory, report = await run_video(video,cfg,output,"camera",queries=["cup"],identity_factory=factory)
            self.assertEqual(models.scopes, [(0,4),(4,8),(8,12)])
            self.assertEqual(len(memory.graph.episodes), 3)
            self.assertEqual(len(memory.graph.facts), 6)
            self.assertEqual(models.inferences, 3)
            self.assertEqual(fixture.calls["grounding"], 3)
            self.assertEqual(fixture.calls["character_facts"], 3)
            self.assertTrue(memory.graph.character_catalog)
            speakers = {r["character_id"] for fact in memory.graph.facts.values() for r in fact.character_refs
                        if r["role"] == "speaker"}
            self.assertEqual(len(speakers), 1)  # Same embeddings confirmed through real cross-episode grouping.
            character = speakers.pop()
            self.assertTrue(character.startswith("person:"))
            for fact in memory.graph.facts.values():
                self.assertEqual(fact.evidence_alignment["status"], "complete")
                self.assertNotIn("<face_", fact.content)
                self.assertNotIn("<voice_", fact.content)
                if "says" in fact.content:
                    ref = fact.evidence_refs[0]
                    episode = memory.graph.episodes[fact.episode_ids[0]]
                    self.assertEqual((ref["start"], ref["end"]), (episode.start_seconds+.5, episode.start_seconds+3))
                    self.assertEqual(len(fact.clip_evidence), 3)
                else:
                    self.assertTrue(fact.evidence_refs[0]["sample"])
                    self.assertTrue(fact.evidence_refs[0]["frame_evidence"][0]["sha256"])
            selected = await memory.retrieve("hello", characters=[character], character_roles=["speaker"])
            self.assertTrue(any(r["node_type"] == "fact" for r in selected.evidence))
            before = memory.graph.to_dict()
            restored, _ = await run_video(video,cfg,output,"camera",identity_factory=factory)
            self.assertEqual(restored.graph.to_dict(), before)
            self.assertEqual(models.inferences, 3)
