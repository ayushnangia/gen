import unittest
import asyncio
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
        }
        self.generator = DialogueGenerator(self.config)

    def test_service_categories(self):
        """Test service category management."""
        categories = self.generator.get_categories_for_service('hotel')
        self.assertTrue(len(categories) > 0)
        self.assertIn('room_reservation', categories)

    def test_persona_selection(self):
        """Test persona selection functionality."""
        persona = self.generator.select_random_persona()
        self.assertIsNotNone(persona)
        self.assertIsInstance(persona, str)

def run_async_test(coro):
    """Helper function to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)