import unittest
import numpy as np
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestParallelProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
        }
        self.generator = DialogueGenerator(self.config)

    def test_parallel_generation(self):
        """Test parallel dialogue generation capabilities."""
        result = self.generator.generate_unique_dialogues(
            num_generations=2,
            min_turns=10,
            max_turns=15,
            temperature_options=[0.7],
            top_p_options=[0.9],
            frequency_penalty_options=[0.0],
            presence_penalty_options=[0.0]
        )
        self.assertIsNotNone(result)