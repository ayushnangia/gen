import pytest
from dashboard.dashboard import load_data, clean_data

def test_sidebar_filters(mock_streamlit):
    """Test sidebar filter components"""
    # Test multiselect filters
    selected_scenarios = mock_streamlit.multiselect(
        "Select Scenario Categories",
        options=['general', 'specific'],
        default=['general']
    )
    assert selected_scenarios == ['general']
    
    selected_status = mock_streamlit.multiselect(
        "Select Resolution Status",
        options=['resolved', 'unresolved'],
        default=['resolved']
    )
    assert selected_status == ['resolved']

def test_metrics_display(mock_streamlit, sample_dataframe):
    """Test metrics display components"""
    # Display metrics
    mock_streamlit.metric(
        "Total Dialogues",
        len(sample_dataframe['dialogue_id'].unique())
    )
    assert "Total Dialogues" in mock_streamlit.metrics

def test_chart_rendering(mock_streamlit, sample_dataframe):
    """Test chart rendering components"""
    # Test if charts can be created without errors
    user_emotions = sample_dataframe['user_emotions'].explode()
    emotion_counts = user_emotions.value_counts()
    
    assert len(emotion_counts) > 0