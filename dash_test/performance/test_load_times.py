import pytest
import time
import pandas as pd
from dashboard.dashboard import load_data, process_chunk

def test_data_loading_performance(sample_jsonl_file):
    """Test data loading performance"""
    start_time = time.time()
    result = load_data(sample_jsonl_file)
    end_time = time.time()
    
    load_time = end_time - start_time
    assert load_time < 5.0  # Should load within 5 seconds

def test_chunk_processing_performance(large_sample_dataframe):
    """Test chunk processing performance"""
    chunk_size = 1000
    start_time = time.time()
    
    # Process dataframe in chunks
    for i in range(0, len(large_sample_dataframe), chunk_size):
        chunk = large_sample_dataframe[i:i + chunk_size]
        processed_chunk = process_chunk([chunk.to_dict('records')])
        assert not processed_chunk.empty
    
    end_time = time.time()
    processing_time = end_time - start_time
    assert processing_time < 10.0  # Should process within 10 seconds

def test_filtering_performance(large_sample_dataframe):
    """Test filtering performance"""
    start_time = time.time()
    
    # Apply multiple filters
    filtered = large_sample_dataframe[
        (large_sample_dataframe['scenario_category'].isin(['general'])) &
        (large_sample_dataframe['resolution_status'].isin(['resolved'])) &
        (large_sample_dataframe['time_slot_description'].isin(['morning']))
    ]
    
    end_time = time.time()
    filter_time = end_time - start_time
    assert filter_time < 1.0  # Filtering should be fast