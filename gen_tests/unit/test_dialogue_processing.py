import unittest
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestDialogueProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
            'similarity_threshold': 0.9
        }
        self.generator = DialogueGenerator(self.config)

    def test_valid_dialogue_processing(self):
        """Test processing of valid dialogue structure."""
        test_dialogue = """
        <User>Book a hotel room</User>
        <Intent>hotel_booking</Intent>
        <Assistant>I'll help you book a hotel room.</Assistant>
        """
        is_valid, turns = self.generator.process_generated_dialogue(test_dialogue)
        self.assertTrue(is_valid)
        self.assertEqual(len(turns), 1)

    def test_invalid_dialogue_processing(self):
        """Test processing of invalid dialogue structure."""
        invalid_dialogue = """
        <User>Incomplete dialogue
        <Intent>incomplete_intent
        """
        is_valid, turns = self.generator.process_generated_dialogue(invalid_dialogue)
        self.assertFalse(is_valid)
        self.assertEqual(len(turns), 0)