import pytest
from dashboard.dashboard import load_data, process_chunk, clean_data, validate_data

def test_load_data(sample_jsonl_file):
    """Test the data loading functionality"""
    result = load_data(sample_jsonl_file)
    assert not result.empty
    assert 'dialogue_id' in result.columns
    assert 'User Utterance' in result.columns
    assert len(result) > 0

def test_process_chunk():
    """Test chunk processing"""
    chunk = [{
        'dialogue_id': 'test1',
        'turns': [{
            'turn_number': 1,
            'utterance': 'Hello',
            'intent': 'greeting',
            'assistant_response': 'Hi there!'
        }],
        'services': ['service1'],
        'num_lines': 1,
        'user_emotions': ['happy'],
        'assistant_emotions': ['neutral'],
        'scenario_category': 'general',
        'generated_scenario': 'Test',
        'time_slot': {'start': 9, 'end': 10, 'description': 'morning'},
        'regions': ['region1'],
        'resolution_status': 'resolved'
    }]
    
    result = process_chunk(chunk)
    assert not result.empty
    assert 'User Utterance' in result.columns
    assert result['dialogue_id'].iloc[0] == 'test1'

def test_empty_file():
    """Test handling of empty file"""
    empty_file = io.StringIO()
    with pytest.raises(SystemExit):
        load_data(empty_file)

def test_corrupted_data():
    """Test handling of corrupted data"""
    corrupted_file = io.StringIO('{"invalid_json":}')
    with pytest.raises(SystemExit):
        load_data(corrupted_file)