import pytest
import psutil
import os
import pandas as pd
from dashboard.dashboard import load_data, process_chunk

def get_process_memory():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def test_memory_usage_loading(sample_jsonl_file):
    """Test memory usage during data loading"""
    initial_memory = get_process_memory()
    
    result = load_data(sample_jsonl_file)
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100  # Should not increase by more than 100MB

def test_memory_usage_processing(large_sample_dataframe):
    """Test memory usage during data processing"""
    initial_memory = get_process_memory()
    
    # Process the large dataframe
    chunk_size = 1000
    for i in range(0, len(large_sample_dataframe), chunk_size):
        chunk = large_sample_dataframe[i:i + chunk_size]
        processed_chunk = process_chunk([chunk.to_dict('records')])
    
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 200  # Should not increase by more than 200MB

def test_memory_cleanup():
    """Test memory cleanup after processing"""
    initial_memory = get_process_memory()
    
    # Create and delete a large dataframe
    large_df = pd.DataFrame({'a': range(1000000)})
    del large_df
    
    final_memory = get_process_memory()
    memory_difference = abs(final_memory - initial_memory)
    
    assert memory_difference < 50  # Memory should be cleaned up