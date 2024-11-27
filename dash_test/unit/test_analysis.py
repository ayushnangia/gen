import pytest
from dashboard.dashboard import compute_sentiment_vader, extract_entities_spacy

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    test_cases = [
        ("I am very happy!", 0.5),
        ("I am very sad.", -0.5),
        ("This is neutral.", 0),
        ("", 0),  # Test empty string
        ("!!!!", 0)  # Test non-text input
    ]
    
    for text, expected in test_cases:
        sentiment = compute_sentiment_vader(text)
        assert isinstance(sentiment, float)
        if expected != 0:
            assert (sentiment > 0) == (expected > 0)

def test_entity_extraction():
    """Test named entity recognition"""
    test_cases = [
        ("John works at Microsoft in New York", ['PERSON', 'ORG', 'GPE']),
        ("The cat sat on the mat", []),  # No named entities
        ("", [])  # Empty string
    ]
    
    for text, expected_types in test_cases:
        entities = extract_entities_spacy(text)
        entity_types = [ent[1] for ent in entities]
        for expected_type in expected_types:
            assert expected_type in entity_types

def test_sentiment_batch_processing(sample_dataframe):
    """Test sentiment analysis on batch of data"""
    sample_dataframe['sentiment'] = sample_dataframe['User Utterance'].apply(compute_sentiment_vader)
    assert 'sentiment' in sample_dataframe.columns
    assert all(isinstance(x, float) for x in sample_dataframe['sentiment'])