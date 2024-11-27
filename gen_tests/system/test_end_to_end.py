import unittest
import os
from parallel_gen import DialogueGenerator
import tempfile

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
            'total_generations': 1
        }
        self.generator = DialogueGenerator(self.config)

    def test_complete_pipeline(self):
        """Test complete dialogue generation pipeline."""
        result = self.generator.generate_unique_dialogues(
            num_generations=1,
            min_turns=10,
            max_turns=15,
            temperature_options=[0.7],
            top_p_options=[0.9],
            frequency_penalty_options=[0.0],
            presence_penalty_options=[0.0]
        )
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(self.config['output_file']))