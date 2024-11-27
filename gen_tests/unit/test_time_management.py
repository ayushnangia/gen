import unittest
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestTimeManagement(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
        }
        self.generator = DialogueGenerator(self.config)

    def test_time_slot_generation(self):
        """Test time slot generation functionality."""
        time_slot = (9, 17, "Daytime")
        generated_time = self.generator.generate_random_time(time_slot)
        self.assertRegex(generated_time, r'^\d{2}:\d{2}$')
        
        hour, minute = map(int, generated_time.split(':'))
        self.assertTrue(9 <= hour < 17)
        self.assertTrue(0 <= minute < 60)
        self.assertEqual(minute % 5, 0)