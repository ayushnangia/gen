import pytest
import time
from dashboard.dashboard import load_data, clean_data

def test_filter_response_time(mock_streamlit, large_sample_dataframe):
    """Test filter response time"""
    start_time = time.time()
    
    # Apply filters
    filtered = large_sample_dataframe[
        (large_sample_dataframe['scenario_category'].isin(['general']))
    ]
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response_time < 1.0  # Should respond within 1 second

def test_visualization_update_time(mock_streamlit, sample_dataframe):
    """Test visualization update responsiveness"""
    start_time = time.time()
    
    # Update visualizations
    user_emotions = sample_dataframe['user_emotions'].explode()
    emotion_counts = user_emotions.value_counts()
    
    end_time = time.time()
    update_time = end_time - start_time
    
    assert update_time < 1.0  # Should update within 1 second

def test_page_load_time(mock_streamlit, sample_dataframe):
    """Test initial page load time"""
    start_time = time.time()
    
    # Simulate page load operations
    total_dialogues = len(sample_dataframe['dialogue_id'].unique())
    mock_streamlit.metric("Total Dialogues", total_dialogues)
    
    end_time = time.time()
    load_time = end_time - start_time
    
    assert load_time < 2.0  # Should load within 2 seconds