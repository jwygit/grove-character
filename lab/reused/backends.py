"""Explicit inference adapters. Imports of GPU libraries are lazy."""
import base64
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import quote, urlparse

from .core import fuse_specialists


def parse_json(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return json.loads(text)


def request_json(url, payload, headers, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Do not log request headers, keys, response bodies or signed URLs.
        raise RuntimeError(f"Inference HTTP error {error.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("Inference endpoint unreachable") from None


def media(path):
    suffix = path.suffix.lower()
    mime = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[suffix]
    return {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode()}


def prompt(clip, mode):
    instruction = (
        "Return JSON only: {\"segments\":[{\"utterance_id\":\"u1\",\"face_id\":\"f1\"}]}. "
        "Return exactly one item for each supplied utterance, preserving its ID. "
        if mode == "association" else
        "Return JSON only: {\"segments\":[{\"start\":0.0,\"end\":1.0,\"face_id\":\"f1\"}]}. "
        "Detect all speech turns, allowing overlapping intervals. Times are seconds relative to video start. ")
    return ("Match audible speech to the supplied face reference images using audiovisual evidence. "
        "Use only supplied face IDs. face_id must be null for offscreen speech or insufficient evidence. "
        "Do not infer identity from gender, clothing or co-occurrence alone. " + instruction +
        "\nDuration: " + str(clip["duration"]) + "\nFace IDs: " + json.dumps([f["id"] for f in clip["faces"]]) +
        ("\nUtterances: " + json.dumps(clip["utterances"]) if mode == "association" else ""))


class Backend:
    def __init__(self, config, root):
        self.config, self.root = config, root
        self.model = self.processor = None
        if config["backend"] not in ("gemini", "local_omni", "specialists", "replay"):
            raise ValueError("unsupported backend")

    def infer(self, clip, mode):
        c = self.config
        if c["backend"] == "replay":
            return json.loads((self.root / c["predictions"]).read_text())[clip["id"]]
        if c["backend"] == "specialists":
            # Existing model services may expose this contract. No dummy model.
            if c.get("scores"):
                raw = json.loads((self.root / c["scores"]).read_text())[clip["id"]]
            else:
                url = c.get("endpoint")
                if not url:
                    raise ValueError("specialists requires endpoint or offline scores file")
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
                    raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials")
                token = os.environ.get(c.get("api_key_env", "AV_SERVICE_TOKEN"))
                if token and parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1"):
                    raise ValueError("use HTTPS when sending service credentials")
                raw = request_json(url, {"schema_version": 1, "clip_id": clip["id"],
                    "mode": mode, "video": media(Path(clip["video"])),
                    "faces": [{"id": f["id"], "image": media(Path(f["image"]))} for f in clip["faces"]],
                    "utterances": clip.get("utterances", []) if mode == "association" else []},
                    {"Authorization": "Bearer " + token} if token else {}, c.get("timeout", 300))
            output = fuse_specialists(clip, raw, mode, c["threshold"], c["margin"])
            output["raw_specialists"] = raw
            return output
        if c["backend"] == "gemini":
            key = os.environ.get(c.get("api_key_env", "GEMINI_API_KEY"))
            if not key:
                raise ValueError("Gemini API key environment variable is missing")
            if not c.get("model") or c["model"].startswith("SET_"):
                raise ValueError("set an available Gemini model ID explicitly")
            parts = [{"text": prompt(clip, mode)}]
            for face in clip["faces"]:
                parts.extend([{"text": face["id"]}, {"inline_data": media(Path(face["image"]))}])
            parts.append({"inline_data": media(Path(clip["video"]))})
            payload = {"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            if len(json.dumps(payload).encode()) > 19_000_000:
                raise ValueError("inline request exceeds experiment 19 MB cap; use shorter clips")
            raw = request_json("https://generativelanguage.googleapis.com/v1beta/models/" +
                quote(c["model"], safe="") + ":generateContent", payload,
                {"x-goog-api-key": key}, c.get("timeout", 300))
            texts = raw["candidates"][0]["content"]["parts"]
            output = parse_json("".join(p.get("text", "") for p in texts if not p.get("thought")))
            output["usage"] = raw.get("usageMetadata", {})
            output["raw_response"] = raw
            return output
        return self.local_omni(clip, mode)

    def local_omni(self, clip, mode):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        from qwen_omni_utils import process_mm_info
        if not torch.cuda.is_available():
            raise RuntimeError("local_omni requires a CUDA host")
        if self.model is None:
            path = (self.root / self.config["model_path"]).resolve()
            if not path.is_dir():
                raise ValueError("local model directory missing; downloads are not automatic")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(path, local_files_only=True)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path,
                local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation=self.config.get("attention", "sdpa"))
            self.model.eval()
        content = [{"type": "text", "text": prompt(clip, mode)}]
        for face in clip["faces"]:
            content.extend([{"type": "text", "text": face["id"]}, {"type": "image", "image": face["image"]}])
        content.append({"type": "video", "video": clip["video"], "fps": self.config.get("fps", 2),
                        "max_pixels": self.config.get("max_pixels", 200704)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        with torch.inference_mode():
            output = self.model.generate(**inputs, use_audio_in_video=True,
                max_new_tokens=self.config.get("max_new_tokens", 2048), do_sample=False)
        response = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        result = parse_json(response)
        result["raw_text"] = response
        result["peak_allocated_bytes_by_gpu"] = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
        return result
