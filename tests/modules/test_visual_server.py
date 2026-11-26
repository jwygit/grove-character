"""Real HTTP protocol, fixture model boundary; no GPU accuracy claim."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lab.full_loop import VisualAPI
from lab.visual_server import VisualServer, normalized_rows


class EncoderFixture:
    backbone = "siglip"
    dim = 2
    model_revision = "fixture-revision"

    def encode_images(self, images):
        return [[3, 4] for _ in images]

    def encode_text(self, text):
        if text == "break inference":
            raise RuntimeError("fixture inference failure")
        if text == "nonfinite":
            return [float("nan"), 0]
        return [3, 4]


class VisualServerTests(unittest.TestCase):
    def setUp(self):
        self.server = VisualServer(("127.0.0.1", 0), EncoderFixture())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.config = dict(server_url=self.base, backbone="siglip", dim=2,
                           model_revision="fixture-revision")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_existing_grove_client_health_image_batch_and_text(self):
        api = VisualAPI(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "one.jpg", Path(tmp) / "two.jpg"]
            for path in paths:
                path.write_bytes(b"fixture-model-input")
            vectors = api.encode_images(paths)
        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], .6)
        self.assertAlmostEqual(api.encode_text("a red cup")[1], .8)

    def test_revision_and_dimension_mismatch_rejected(self):
        for field, value in (("dim", 3), ("model_revision", "other")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                VisualAPI(dict(self.config, **{field: value}))

    def test_bad_requests_are_errors_not_embeddings(self):
        cases = [("/encode_image_batch", {"image_b64_list": []}),
                 ("/encode_image_batch", {"image_b64_list": ["!bad!"]}),
                 ("/encode_image_batch", {"image_b64_list": [None]}),
                 ("/encode_image_batch", {"image_b64_list": [base64.b64encode(b"x").decode()] * 33}),
                 ("/encode_text", {"text": " "}), ("/encode_text", [])]
        for path, body in cases:
            with self.subTest(body=body), self.assertRaises(HTTPError) as error:
                self.post(path, body)
            self.assertEqual(error.exception.code, 400)
        self.server.max_body_bytes = 2
        with self.assertRaises(HTTPError) as error:
            self.post("/encode_text", {"text": "oversized"})
        self.assertEqual(error.exception.code, 413)

    def test_model_failure_and_invalid_vectors_are_server_errors(self):
        for text in ("break inference", "nonfinite"):
            with self.subTest(text=text), self.assertLogs(level="ERROR"), self.assertRaises(HTTPError) as error:
                self.post("/encode_text", {"text": text})
            self.assertEqual(error.exception.code, 500)

    def test_zero_wrong_shape_and_wrong_dimension_rejected(self):
        for values, count, dim in (([[0, 0]], 1, 2), ([[1, 0]], 2, 2), ([[1, 0]], 1, 3)):
            with self.assertRaises(RuntimeError):
                normalized_rows(values, count, dim)


if __name__ == "__main__":
    unittest.main()
