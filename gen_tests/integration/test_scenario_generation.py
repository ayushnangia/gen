import unittest
import asyncio
from unittest.mock import Mock, patch
from parallel_gen import DialogueGenerator
import tempfile
import os

class TestScenarioGeneration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_file': os.path.join(self.test_dir, 'test_dialogues.jsonl'),
            'hash_file': os.path.join(self.test_dir, 'test_hashes.jsonl'),
            'embedding_file': os.path.join(self.test_dir, 'test_embeddings.npy'),
        }
        self.generator = DialogueGenerator(self.config)

    @patch('httpx.AsyncClient.post')
    async def test_scenario_generation(self, mock_post):
        """Test scenario generation functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Test scenario content'
                }
            }]
        }
        mock_post.return_value = mock_response

        scenario, time_slot = await self.generator.generate_dynamic_scenario(
            "hotel", ["hotel"], "London"
        )
        self.assertIsNotNone(scenario)
        self.assertIsNotNone(time_slot)