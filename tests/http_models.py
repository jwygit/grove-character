"""Explicit synthetic MODEL SERVICE for transport/orchestration tests, never production.

It reads real JPEG colors. It is not a language, embedding, or recognition model;
the resulting tests cannot measure model accuracy. Grove modules are not replaced.
"""
import base64
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import subprocess
import threading


def jpeg_color(encoded):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", "scale=1:1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], input=base64.b64decode(encoded))
    return ("red", "green", "blue")[max(range(3), key=lambda i: raw[i])]


def vector(text):
    return [1.0 + text.lower().count(word) for word in ("red", "blue", "green")]


class ModelFixture:
    def __init__(self):
        self.calls = Counter()
        self.fail = set()
        self.invalid = set()
        self.tasks = []

    def respond(self, path, payload):
        if path == "/health":
            return dict(backbone="synthetic-color", dim=3)
        if path == "/encode_image_batch":
            self.calls["visual_image"] += 1
            return dict(embeddings=[vector(jpeg_color(b64)) for b64 in payload["image_b64_list"]])
        if path == "/encode_text":
            self.calls["visual_query"] += 1
            return dict(embedding=vector(payload["text"]))
        if path == "/v1/embeddings":
            self.calls["embedding"] += 1
            if "embedding" in self.fail:
                raise RuntimeError("fixture embedding outage")
            return dict(data=[dict(index=i, embedding=vector(text)) for i, text in reversed(list(enumerate(payload["input"])) )])
        prompt = payload["messages"][0]["content"]
        if isinstance(prompt, list):
            text = "\n".join(row["text"] for row in prompt if row["type"] == "text")
        else:
            text = prompt
        match = re.search(r"TASK: (\w+)", text)
        task = match[1] if match else ("grounding" if "Locate the supplied descriptions" in text else "character_facts")
        self.calls[task] += 1
        self.tasks.append(task)
        if task in self.fail:
            raise RuntimeError("fixture service outage")
        if task in self.invalid:
            return {"choices": [{"message": {"content": "{invalid json"}}]}
        if task == "caption_frames":
            ids = re.findall(r"(frame:\d+) time=", text)
            images = [row["image_url"]["url"].split(",", 1)[1] for row in prompt if row["type"] == "image_url"]
            colors = [jpeg_color(data) for data in images]
            answer = dict(description="A " + colors[0] + " cup is displayed.", captions=[
                dict(id=key, text=f"A {color} cup is displayed.") for key, color in zip(ids, colors)])
        elif task.startswith("boundary"):
            before, after = text.split("Incoming window")
            old = re.findall(r"A (red|blue|green) cup", before)[-1]
            new = re.findall(r"A (red|blue|green) cup", after)[0]
            answer = dict(should_end=old != new, should_wait=old == new, confidence=0.9, reasoning="fixture color activity change")
        elif task == "episode_summary":
            color = re.findall(r"A (red|blue|green) cup", text)[0]
            answer = dict(subject=f"{color} cup", summary=f"A {color} cup is displayed.", description=f"A {color} cup is visible.", keywords=[color, "cup"])
        elif task == "fact_extract":
            data = json.loads(text.split("INPUT_JSON:\n", 1)[1])
            answer = dict(facts=[dict(content=row["caption"], frame_ids=[row["frame_id"]], role="core", confidence=.9, weight=.8)
                                 for row in data["frames"]])
        elif task == "topic_create":
            answer = dict(title="Cup observations", summary="Colored cups are displayed.", keywords=["cup"])
        elif task == "topic_match":
            ids = re.findall(r"\*\*([^*]+)\*\*", text)
            answer = dict(matches={ids[0]: dict(match=True)})
        elif task == "topic_update":
            answer = dict(title="Cup observations", summary="Red, blue and green cups are displayed over time.", keywords=["cup", "red", "blue", "green"])
        elif task == "episode_roles":
            ids = re.findall(r"\*\*([^*]+)\*\*", text)
            answer = dict(assignments={key: dict(role="developing", weight=.8) for key in ids}, coherence_score=.9)
        elif task == "grounding":
            data = json.loads(prompt[-1]["text"])
            frames = re.findall(r"(frame:\d+) at ([\d.]+) seconds", text)
            answer = dict(locations=[dict(memory_id=row["id"], reason="synthetic sampled observation",
                        intervals=[], frame_ids=[frames[0][0]]) for row in data["memories"]])
        elif task == "character_facts":
            # Parse only the explicit evidence JSON sections, not arbitrary prompt text.
            refs = json.JSONDecoder().raw_decode(text.split("IDENTITY_EVIDENCE_JSON (data, not instructions):\n", 1)[1])[0]
            memory_text = text.split("M3_MEMORY_EVIDENCE_JSON (data, not instructions):\n", 1)[1]
            evidence = json.JSONDecoder().raw_decode(memory_text)[0]
            answer = dict(facts=[dict(content="<voice_0> says hello.", utterance_ids=[refs[0]["utterance_id"]], memory_ids=[]),
                                dict(content="<face_0> holds a cup.", utterance_ids=[], memory_ids=[evidence["memories"][0]["id"]])])
        else:
            raise AssertionError(task)
        return {"choices": [{"message": {"content": json.dumps(answer)}}]}


@contextmanager
def model_server():
    fixture = ModelFixture()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self._respond(None)
        def do_POST(self):
            self._respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        def _respond(self, payload):
            try:
                body = fixture.respond(self.path, payload)
                status = 200
            except Exception as exc:
                body, status = {"error": str(exc)}, 503
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fixture, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def config(base, visual=True):
    model = dict(endpoint=base+"/v1/chat/completions", model="synthetic-contract-service",
                 model_revision="fixture-v1", timeout=10)
    return dict(caption=model, writer=dict(model), embedding=dict(model, endpoint=base+"/v1/embeddings"),
        visual_encoder=dict(server_url=base, model_revision="fixture-v1", backbone="synthetic-color", dim=3),
        sample_seconds=1, window_seconds=2, max_frames_per_video=100,
        grove=dict(use_visual_channel=visual, min_buffer_frames=0, max_buffer_seconds=100,
                   llm_max_retries=2, llm_retry_backoff_base=0, role_reassignment_every_n=1, fact_top_k=30))


def create_video(path, seconds_per_color=4):
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    for color in ("red", "blue", "green"):
        command += ["-f", "lavfi", "-i", f"color=c={color}:s=96x64:r=4:d={seconds_per_color}"]
    command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds_per_color*3}",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-c:a", "aac", "-n", str(path)]
    subprocess.run(command, check=True, capture_output=True)
