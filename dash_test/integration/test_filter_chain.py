import pytest
import pandas as pd
from dashboard.dashboard import load_data, clean_data

def test_filter_chain(sample_dataframe):
    """Test complete filter chain functionality"""
    # Apply multiple filters
    filtered = sample_dataframe[
        (sample_dataframe['scenario_category'].isin(['general'])) &
        (sample_dataframe['resolution_status'].isin(['resolved'])) &
        (sample_dataframe['time_slot_description'].isin(['morning']))
    ]
    
    assert len(filtered) > 0
    assert filtered['scenario_category'].iloc[0] == 'general'
    assert filtered['resolution_status'].iloc[0] == 'resolved'

def test_filter_combinations(sample_dataframe):
    """Test different filter combinations"""
    # Test cases for different filter combinations
    filter_combinations = [
        {'scenario_category': ['general']},
        {'resolution_status': ['resolved']},
        {'time_slot_description': ['morning']},
        {'scenario_category': ['general'], 'resolution_status': ['resolved']},
    ]
    
    for filters in filter_combinations:
        query = ' & '.join([
            f"({col}.isin({val}))" 
            for col, val in filters.items()
        ])
        filtered = sample_dataframe.query(query)
        assert len(filtered) > 0

def test_empty_filter_results(sample_dataframe):
    """Test handling of filters that return no results"""
    filtered = sample_dataframe[
        (sample_dataframe['scenario_category'] == 'nonexistent_category')
    ]
    assert len(filtered) == 0

def test_filter_data_integrity(sample_dataframe):
    """Test that filtering doesn't modify original data"""
    original_length = len(sample_dataframe)
    filtered = sample_dataframe[
        sample_dataframe['scenario_category'].isin(['general'])
    ]
    assert len(sample_dataframe) == original_length