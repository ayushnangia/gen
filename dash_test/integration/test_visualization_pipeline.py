import pytest
import plotly.graph_objects as go
import plotly.express as px
from dashboard.dashboard import load_data, clean_data

def test_emotion_distribution_visualization(sample_dataframe):
    """Test emotion distribution visualization pipeline"""
    # Create emotion distribution
    user_emotions = sample_dataframe['user_emotions'].explode()
    emotion_counts = user_emotions.value_counts()
    
    # Create visualization
    fig = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title='Emotion Distribution'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == 'pie'

def test_service_usage_visualization(sample_dataframe):
    """Test service usage visualization pipeline"""
    # Create service distribution
    services = sample_dataframe['services'].explode()
    service_counts = services.value_counts()
    
    # Create visualization
    fig = px.bar(
        x=service_counts.index,
        y=service_counts.values,
        title='Service Usage'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == 'bar'

def test_time_slot_visualization(sample_dataframe):
    """Test time slot visualization pipeline"""
    time_slot_counts = sample_dataframe['time_slot_description'].value_counts()
    
    fig = px.bar(
        x=time_slot_counts.index,
        y=time_slot_counts.values,
        title='Time Slot Distribution'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_visualization_data_mapping(sample_dataframe):
    """Test correct data mapping in visualizations"""
    # Test with emotion distribution
    user_emotions = sample_dataframe['user_emotions'].explode()
    fig = px.pie(
        values=user_emotions.value_counts().values,
        names=user_emotions.value_counts().index
    )
    
    # Verify data mapping
    assert len(fig.data[0].values) == len(user_emotions.value_counts())
    assert all(isinstance(val, (int, float)) for val in fig.data[0].values)