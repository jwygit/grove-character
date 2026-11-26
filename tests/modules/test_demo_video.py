import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.fetch_demo_video import fetch


class DemoDownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'download-integrity-fixture')
        self.output = self.root / 'output'
        self.asset = dict(filename='demo.mp4', bytes=self.source.stat().st_size,
                          sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                          download_url=self.source.as_uri())

    def test_download_and_offline_verification(self):
        path = fetch(self.asset, self.output)
        self.assertEqual(path.read_bytes(), self.source.read_bytes())
        self.source.unlink()
        self.assertEqual(fetch(self.asset, self.output, verify_only=True), path)
        self.assertEqual(fetch(self.asset, self.output), path)
        self.assertEqual(list(self.output.iterdir()), [path])

    def test_bad_download_never_published_and_partial_removed(self):
        for change in ({'sha256': '0'*64}, {'bytes': 1}, {'bytes': 1000}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    fetch(self.asset | change, self.output)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_corrupt_file_preserved_and_offline_missing_fails(self):
        with self.assertRaises(FileNotFoundError):
            fetch(self.asset, self.output, verify_only=True)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        path = self.output / self.asset['filename']
        path.write_bytes(b'user-copy')
        with self.assertRaises(ValueError):
            fetch(self.asset, self.output)
        self.assertEqual(path.read_bytes(), b'user-copy')
