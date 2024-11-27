import pytest
from dashboard.dashboard import clean_data, validate_data
import pandas as pd

def test_clean_data(sample_dataframe):
    """Test data cleaning functionality"""
    result = clean_data(sample_dataframe)
    assert 'regions_str' in result.columns
    assert 'services_str' in result.columns
    assert isinstance(result['time_slot_start'].iloc[0], int)
    assert isinstance(result['time_slot_end'].iloc[0], int)

def test_validate_data(sample_dataframe):
    """Test data validation"""
    assert validate_data(sample_dataframe) == True

def test_validate_data_missing_columns():
    """Test data validation with missing columns"""
    invalid_df = pd.DataFrame({'column1': [1]})
    with pytest.raises(SystemExit):
        validate_data(invalid_df)

def test_clean_data_missing_values():
    """Test cleaning data with missing values"""
    df_with_missing = pd.DataFrame({
        'dialogue_id': ['test1'],
        'services': [None],
        'user_emotions': [None],
        'assistant_emotions': [None],
        'regions': [None]
    })
    result = clean_data(df_with_missing)
    assert result['services_str'].iloc[0] == ''
    assert result['user_emotions_str'].iloc[0] == ''