"""Actual Grove loop + ffmpeg + HTTP, synthetic model responses only."""
import json
from pathlib import Path
import tempfile
import unittest
from lab.full_loop import run_video
from tests.http_models import model_server, config, create_video


class FullLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "colors.mp4"
        create_video(self.video)
        self.server = model_server()
        self.fixture, self.base = self.server.__enter__()
        self.config = config(self.base)

    async def asyncTearDown(self):
        self.server.__exit__(None, None, None)
        self.temp.cleanup()

    async def run_loop(self, **kwargs):
        return await run_video(self.video, self.config, self.root / "run", "camera", **kwargs)

    async def test_real_video_dynamic_graph_both_indices_retrieval_and_reload(self):
        memory, report = await self.run_loop(queries=["red cup"])
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["accuracy_measured"])
        self.assertEqual([(e.start_seconds, e.end_seconds) for e in memory.graph.episodes.values()],
                         [(86400,86404),(86404,86408),(86408,86412)])
        self.assertEqual(len(memory.graph.topics), 1)
        self.assertEqual(len(memory.graph.facts), 12)
        self.assertEqual(len(memory.graph.visual_clips), 12)
        self.assertTrue(all(f.evidence_alignment["status"] == "complete" for f in memory.graph.facts.values()))
        for task in ("caption_frames", "boundary_stream", "episode_summary", "fact_extract",
                     "topic_create", "topic_match", "topic_update", "episode_roles", "embedding", "visual_image", "visual_query"):
            self.assertGreater(self.fixture.calls[task], 0, task)
        evidence = report["queries"][0]["result"]["evidence"]
        self.assertEqual({r["node_type"] for r in evidence}, {"fact", "visual_clip"})
        self.assertTrue(any("red" in row["content"] for row in evidence))
        self.assertEqual(memory._dual.bm25.search("red", "fact")[0][0] in memory.graph.facts, True)
        self.assertEqual(memory._dual.bm25.search("蓝色杯子", "fact"), [])
        counts = self.fixture.calls.copy()
        restored, _ = await self.run_loop(queries=["red cup"])
        self.assertEqual(memory.graph.to_dict(), restored.graph.to_dict())
        self.assertEqual(self.fixture.calls["fact_extract"], counts["fact_extract"])
        self.assertEqual(self.fixture.calls["caption_frames"], counts["caption_frames"])
        limited = await restored.retrieve("cup", current_time_span="1-00:00:00-00:00:04")
        for row in limited.evidence:
            if row["node_type"] == "visual_clip":
                self.assertLessEqual(row["end_seconds"], 86404)
            for edge in row.get("clip_evidence", []):
                self.assertLessEqual(edge["end"], 86404)
        missing = await restored.retrieve("cup", source_id="other-camera")
        self.assertFalse(missing.evidence)

    async def test_failed_fact_transaction_rolls_back_then_resumes_without_duplicates(self):
        memory, report = await self.run_loop(stop_after_windows=2)
        self.assertEqual(len(memory.graph.episodes), 0)
        path = self.root / "run/checkpoint.json"
        checkpoint = path.read_bytes()
        self.fixture.fail.add("fact_extract")
        with self.assertRaises(RuntimeError):
            await self.run_loop()
        self.assertEqual(path.read_bytes(), checkpoint)
        self.fixture.fail.clear()
        restored, _ = await self.run_loop()
        self.assertEqual(len(restored.graph.episodes), 3)
        self.assertEqual(len(restored.graph.facts), 12)
        self.assertEqual(len(restored.state["windows"]), 6)

    async def test_bad_summary_does_not_publish_fallback_and_embedding_failure_recovers(self):
        await self.run_loop(stop_after_windows=2)
        checkpoint = (self.root / "run/checkpoint.json").read_bytes()
        self.fixture.invalid.add("episode_summary")
        with self.assertRaises(RuntimeError):
            await self.run_loop()
        self.assertEqual((self.root / "run/checkpoint.json").read_bytes(), checkpoint)
        self.fixture.invalid.clear()
        self.fixture.fail.add("embedding")
        with self.assertRaises(RuntimeError):
            await self.run_loop()
        self.assertFalse((self.root / "run/index_manifest.json").exists())
        self.assertTrue(json.loads((self.root / "run/checkpoint.json").read_text())["finished"])
        self.fixture.fail.clear()
        memory, _ = await self.run_loop()
        self.assertEqual(len(memory.graph.facts), 12)

    async def test_core_without_visual_encoder_and_cache_invalidation(self):
        self.config["grove"]["use_visual_channel"] = False
        memory, _ = await self.run_loop(queries=["cup"])
        self.assertEqual(self.fixture.calls["visual_image"], 0)
        self.assertTrue(memory.graph.facts)
        manifest = json.loads((self.root / "run/index_manifest.json").read_text())
        cache = self.root / "run/indices" / manifest["directory"] / "bm25_index.pkl"
        cache.write_text("broken cache")
        repaired, _ = await self.run_loop(queries=["cup"])
        self.assertTrue(repaired._dual.bm25.search("red", "fact"))
        self.config["embedding"]["model_revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            await self.run_loop()
