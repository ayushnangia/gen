import pytest
import pandas as pd
import json
import io
import streamlit as st
from pathlib import Path

@pytest.fixture
def sample_jsonl_file():
    """Create a sample JSONL file for testing"""
    data = [
        {
            "dialogue_id": "test1",
            "services": ["service1", "service2"],
            "num_lines": 2,
            "user_emotions": ["happy", "neutral"],
            "assistant_emotions": ["happy", "neutral"],
            "scenario_category": "general",
            "generated_scenario": "Test scenario",
            "time_slot": {"start": 9, "end": 10, "description": "morning"},
            "regions": ["region1"],
            "resolution_status": "resolved",
            "turns": [
                {
                    "turn_number": 1,
                    "utterance": "Hello",
                    "intent": "greeting",
                    "assistant_response": "Hi there!"
                },
                {
                    "turn_number": 2,
                    "utterance": "How are you?",
                    "intent": "inquiry",
                    "assistant_response": "I'm doing well, thank you!"
                }
            ]
        }
    ]
    
    file_obj = io.StringIO()
    for item in data:
        file_obj.write(json.dumps(item) + '\n')
    file_obj.seek(0)
    return file_obj

@pytest.fixture
def sample_dataframe():
    """Create a sample processed DataFrame for testing"""
    return pd.DataFrame({
        'dialogue_id': ['test1'],
        'User Utterance': ['Hello'],
        'Intent': ['greeting'],
        'Assistant Response': ['Hi there!'],
        'Turn Number': [1],
        'services': [['service1', 'service2']],
        'num_lines': [2],
        'user_emotions': [['happy', 'neutral']],
        'assistant_emotions': [['happy', 'neutral']],
        'scenario_category': ['general'],
        'generated_scenario': ['Test scenario'],
        'time_slot_start': [9],
        'time_slot_end': [10],
        'time_slot_description': ['morning'],
        'regions': [['region1']],
        'resolution_status': ['resolved']
    })

@pytest.fixture
def large_sample_dataframe():
    """Create a larger sample DataFrame for performance testing"""
    base_df = pd.DataFrame({
        'dialogue_id': ['test1'],
        'User Utterance': ['Hello'],
        'Intent': ['greeting'],
        'Assistant Response': ['Hi there!'],
        'Turn Number': [1],
        'services': [['service1', 'service2']],
        'num_lines': [2],
        'user_emotions': [['happy', 'neutral']],
        'assistant_emotions': [['happy', 'neutral']],
        'scenario_category': ['general'],
        'generated_scenario': ['Test scenario'],
        'time_slot_start': [9],
        'time_slot_end': [10],
        'time_slot_description': ['morning'],
        'regions': [['region1']],
        'resolution_status': ['resolved']
    })
    # Repeat the DataFrame 1000 times for performance testing
    return pd.concat([base_df] * 1000, ignore_index=True)

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions for testing UI components"""
    class MockStreamlit:
        def __init__(self):
            self.sidebar = self
            self.container_values = {}
            self.metrics = {}
        
        def markdown(self, text, unsafe_allow_html=False):
            return text
            
        def metric(self, label, value, delta=None):
            self.metrics[label] = value
            
        def multiselect(self, label, options, default=None):
            return default or options
            
        def number_input(self, label, min_value=None, max_value=None, value=None):
            return value or min_value
            
    return MockStreamlit()