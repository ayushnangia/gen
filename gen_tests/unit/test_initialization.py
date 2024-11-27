import unittest
import tempfile
import os
from parallel_gen import DialogueGenerator

class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
            'similarity_threshold': 0.9,
            'dataset_name': 'pfb30/multi_woz_v22',
            'total_generations': 5
        }

    def test_basic_initialization(self):
        """Test basic initialization of DialogueGenerator."""
        generator = DialogueGenerator(self.config)
        self.assertIsNotNone(generator)
        self.assertEqual(generator.similarity_threshold, 0.9)

    def test_model_initialization(self):
        """Test model initialization."""
        generator = DialogueGenerator(self.config)
        self.assertIsNotNone(generator.embedding_model)
        self.assertIsNotNone(generator.client)

    def test_emotion_lists_initialization(self):
        """Test emotion lists initialization."""
        generator = DialogueGenerator(self.config)
        self.assertTrue(len(generator.USER_EMOTION_LIST) > 0)
        self.assertTrue(len(generator.ASSISTANT_EMOTION_LIST) > 0)