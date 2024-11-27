import unittest
import numpy as np
from unittest.mock import patch
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestDialogueGeneration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
        }
        self.generator = DialogueGenerator(self.config)

    @patch('parallel_gen.DialogueGenerator.generate_unique_dialogue')
    def test_dialogue_generation(self, mock_generate):
        """Test dialogue generation with mocked unique dialogue generation."""
        mock_generate.return_value = ({
            'dialogue_id': 'test',
            'turns': [],
            'services': ['hotel']
        }, np.random.rand(1, 768))

        dialogues = self.generator.generate_unique_dialogues(
            num_generations=2,
            min_turns=10,
            max_turns=15,
            temperature_options=[0.7],
            top_p_options=[0.9],
            frequency_penalty_options=[0.0],
            presence_penalty_options=[0.0]
        )
        self.assertIsNotNone(dialogues)