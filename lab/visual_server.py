"""Local-weight SigLIP server for Grove's image/text embedding HTTP contract.

Heavy dependencies are loaded only when starting SiglipEncoder. No model download
or synthetic inference fallback is available in this production entry point.
"""
import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import threading

import numpy as np


def normalized_rows(values, count, dim=None):
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] != count or not rows.shape[1]:
        raise RuntimeError("encoder returned invalid embedding shape")
    if dim is not None and rows.shape[1] != dim:
        raise RuntimeError("encoder dimension changed")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("encoder returned nonfinite/zero embeddings")
    return (rows / norms).tolist()


class SiglipEncoder:
    backbone = "siglip"

    def __init__(self, model, revision, device="cpu"):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, SiglipModel

        path = Path(model).resolve(strict=True)
        if not path.is_dir() or not revision or "SET_" in revision:
            raise ValueError("supply a local model directory and real weight revision")
        self.model_revision = revision
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = SiglipModel.from_pretrained(str(path), local_files_only=True).to(device).eval()
        self.max_length = self.model.config.text_config.max_position_embeddings
        # Real startup probes establish and cross-check the image/text vector space.
        self.dim = len(self._images([Image.new("RGB", (224, 224))])[0])
        if len(self.encode_text("startup probe")) != self.dim:
            raise RuntimeError("image/text embedding dimensions differ")

    def _rows(self, features, count):
        return normalized_rows(features.detach().float().cpu().numpy(), count, getattr(self, "dim", None))

    def _images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self._rows(self.model.get_image_features(**inputs), len(images))

    def encode_images(self, payloads):
        from PIL import Image, UnidentifiedImageError

        images = []
        try:
            for payload in payloads:
                with Image.open(io.BytesIO(payload)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("image exceeds 16 million pixels")
                    images.append(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("invalid/oversized image") from exc
        return self._images(images)

    def encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.max_length).to(self.device)
        with self.torch.inference_mode():
            return self._rows(self.model.get_text_features(**inputs), 1)[0]


class VisualServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, encoder, max_body_bytes=32 * 1024 * 1024, max_batch=32):
        self.encoder = encoder
        self.max_body_bytes = max_body_bytes
        self.max_batch = max_batch
        self.inference_lock = threading.Lock()
        super().__init__(address, VisualHandler)


class VisualHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def log_message(self, fmt, *args):
        logging.info(fmt, *args)

    def reply(self, status, data):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "unknown endpoint"})
        encoder = self.server.encoder
        self.reply(200, {"backbone": encoder.backbone, "dim": encoder.dim,
                         "model_revision": encoder.model_revision})

    def do_POST(self):
        if self.path not in ("/encode_text", "/encode_image_batch"):
            return self.reply(404, {"error": "unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= self.server.max_body_bytes:
                return self.reply(413, {"error": "invalid/excessive body length"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            encoder = self.server.encoder
            if self.path == "/encode_text":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 32768:
                    raise ValueError("text must be nonempty and at most 32768 characters")
                with self.server.inference_lock:
                    vector = encoder.encode_text(text)
                result = {"embedding": normalized_rows([vector], 1, encoder.dim)[0]}
            else:
                items = body.get("image_b64_list")
                if not isinstance(items, list) or not 0 < len(items) <= self.server.max_batch:
                    raise ValueError("invalid image batch size")
                payloads = []
                for item in items:
                    if not isinstance(item, str):
                        raise ValueError("image must be a base64 string")
                    raw = base64.b64decode(item, validate=True)
                    if not raw:
                        raise ValueError("empty image")
                    payloads.append(raw)
                with self.server.inference_lock:
                    vectors = encoder.encode_images(payloads)
                result = {"embeddings": normalized_rows(vectors, len(items), encoder.dim)}
        except (ValueError, binascii.Error, UnicodeError) as exc:
            return self.reply(400, {"error": str(exc)})
        except Exception:
            logging.exception("visual encoder request failed")
            return self.reply(500, {"error": "visual inference failed; see server log"})
        self.reply(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local SigLIP model/processor directory")
    parser.add_argument("--revision", required=True, help="downloaded commit/weight digest for audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9090, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    encoder = SiglipEncoder(args.model, args.revision, args.device)
    with VisualServer((args.host, args.port), encoder) as server:
        logging.info("SigLIP ready on %s:%s, dim=%s revision=%s", *server.server_address,
                     encoder.dim, encoder.model_revision)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
