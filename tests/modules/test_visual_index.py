from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests.visual_fixtures import write_index, FixtureEncoder
from hypervideo.visual_retriever import VisualRetriever


class VisualIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'visual.npz'
        self.vr = write_index(self.path)

    def test_filters_candidates_before_top_k_and_accepts_all_id_schemas(self):
        rows = self.vr.search_within('query', {'c1'}, top_k=1)
        self.assertEqual(rows[0][2]['visual_clip_id'], 'c1')
        self.assertAlmostEqual(rows[0][1], .6)
        self.assertEqual(self.vr.search('query', days=[2])[0][2]['clip_id'], 'c3')
        self.assertEqual(self.vr.search('query', days=[]), [])

    def test_empty_candidates_zero_budget_and_threshold(self):
        self.assertEqual(self.vr.search_within('query', set()), [])
        self.assertEqual(self.vr.search('query', top_k=0), [])
        self.assertEqual(self.vr.search_within('query', {'c1'}, min_score=.7), [])
        for kwargs in [dict(top_k=-1), dict(min_score=2)]:
            with self.assertRaises(ValueError):
                self.vr.search('query', **kwargs)

    def test_missing_candidate_is_not_silent_low_recall(self):
        with self.assertRaisesRegex(ValueError, 'missing candidate'):
            self.vr.search_within('query', {'c99'})

    def test_backbone_and_dimensions_must_match_even_with_valid_size_vector(self):
        for field, value in [('backbone', 'different-model'), ('dim', 3)]:
            self.vr.encoder = FixtureEncoder()
            setattr(self.vr.encoder, field, value)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                self.vr.search('query')

    def test_bad_query_vectors_and_service_failures_propagate(self):
        for vector in ([0.,0.], [float('nan'),1.], [1.,0.,0.]):
            self.vr.encoder.encode_text = lambda query: np.array(vector)
            with self.assertRaisesRegex(ValueError, 'query embedding'):
                self.vr.search('query')
        def broken(query): raise RuntimeError('service unavailable')
        self.vr.encoder.encode_text = broken
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            self.vr.search('query')

    def test_duplicate_ids_and_invalid_index_vectors_rejected(self):
        for embs in ([[0,0],[1,0],[1,0]], [[float('inf'),0],[1,0],[1,0]], [[1,0,0]]*3):
            with self.assertRaises(ValueError):
                write_index(self.path, embeddings=embs)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            write_index(self.path, metadata=[dict(id='same', day=1)]*3)

    def test_remote_encoder_is_used_without_loading_a_local_model(self):
        with patch('hypervideo.visual_indexing.RemoteVisualEncoder', return_value=FixtureEncoder()) as remote:
            vr = VisualRetriever(str(self.path), remote_url='http://encoder.test')
            self.assertIsNone(vr.encoder)
            self.assertEqual(len(vr.search('query')), 3)
            remote.assert_called_once_with('http://encoder.test')
