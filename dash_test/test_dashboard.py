import pytest
import pandas as pd
from unittest.mock import Mock, patch
from dashboard.dashboard import (
    load_data,
    process_chunk,
    clean_data,
    validate_data,
    compute_sentiment_vader,
    extract_entities_spacy
)
import json
import io

# Mock Streamlit functions to prevent actual Streamlit operations during testing
@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    mock_st = Mock()
    monkeypatch.setattr("dashboard.dashboard.st", mock_st)
    return mock_st

def test_load_data_valid_jsonl():
    # Create a mock JSONL file
    mock_data = [
        json.dumps({
            "dialogue_id": "dialogue_1",
            "services": ["service_a"],
            "num_lines": 2,
            "user_emotions": ["happy"],
            "assistant_emotions": ["neutral"],
            "scenario_category": "category_1",
            "generated_scenario": "Scenario text here.",
            "time_slot": {"start": 9, "end": 17, "description": "09:00 - 17:00"},
            "regions": ["region_1"],
            "resolution_status": "resolved",
            "turns": [
                {
                    "utterance": "Hello!",
                    "intent": "greeting",
                    "assistant_response": "Hi there!",
                    "turn_number": 1
                },
                {
                    "utterance": "I need help.",
                    "intent": "request_help",
                    "assistant_response": "Sure, how can I assist you?",
                    "turn_number": 2
                }
            ]
        })
    ]
    mock_file = io.BytesIO("\n".join(mock_data).encode('utf-8'))

    # Call load_data
    dialogues = load_data(mock_file)

    # Assertions
    assert isinstance(dialogues, pd.DataFrame)
    assert len(dialogues) == 2  # Two turns
    assert dialogues['dialogue_id'].iloc[0] == "dialogue_1"
    assert dialogues['User Utterance'].iloc[0] == "Hello!"
    assert dialogues['Assistant Response'].iloc[0] == "Hi there!"

def test_load_data_invalid_jsonl():
    # Create a mock invalid JSONL file
    mock_data = [
        '{"dialogue_id": "dialogue_1", "services": ["service_a"], "num_lines": 2, "user_emotions": ["happy"], "assistant_emotions": ["neutral"], "scenario_category": "category_1", "generated_scenario": "Scenario text here.", "time_slot": {"start": 9, "end": 17, "description": "09:00 - 17:00"}, "regions": ["region_1"], "resolution_status": "resolved", "turns": [INVALID_JSON]'
    ]
    mock_file = io.BytesIO("\n".join(mock_data).encode('utf-8'))

    with pytest.raises(json.JSONDecodeError):
        load_data(mock_file)

def test_process_chunk():
    # Sample chunk
    chunk = [
        {
            "dialogue_id": "dialogue_2",
            "services": ["service_b", "service_c"],
            "num_lines": 3,
            "user_emotions": ["sad"],
            "assistant_emotions": ["empathetic"],
            "scenario_category": "category_2",
            "generated_scenario": "Another scenario.",
            "time_slot": {"start": 18, "end": 22, "description": "18:00 - 22:00"},
            "regions": ["region_2"],
            "resolution_status": "unresolved",
            "turns": [
                {
                    "utterance": "I'm feeling down.",
                    "intent": "express_feeling",
                    "assistant_response": "I'm sorry to hear that. How can I help?",
                    "turn_number": 1
                },
                {
                    "utterance": "I lost my job.",
                    "intent": "share_problem",
                    "assistant_response": "That must be tough. Let's see what we can do.",
                    "turn_number": 2
                },
                {
                    "utterance": "Thank you for listening.",
                    "intent": "appreciation",
                    "assistant_response": "You're welcome. Anytime you need support.",
                    "turn_number": 3
                }
            ]
        }
    ]

    dialogues_chunk = process_chunk(chunk)

    # Assertions
    assert isinstance(dialogues_chunk, pd.DataFrame)
    assert len(dialogues_chunk) == 3  # Three turns
    assert dialogues_chunk['dialogue_id'].iloc[0] == "dialogue_2"
    assert dialogues_chunk['User Utterance'].iloc[1] == "I lost my job."
    assert dialogues_chunk['Resolution Status'].iloc[2] == "unresolved"

def test_clean_data():
    # Sample dialogues DataFrame
    data = {
        'dialogue_id': ["dialogue_3"],
        'services': [["service_d"]],
        'num_lines': [1],
        'user_emotions': [["excited"]],
        'assistant_emotions': [["happy"]],
        'scenario_category': ["category_3"],
        'generated_scenario': ["Exciting scenario."],
        'time_slot': [ {"start": 12, "end": 14, "description": "12:00 - 14:00"} ],
        'regions': [["region_3"]],
        'resolution_status': ["resolved"],
        'utterance': ["Let's start!"],
        'intent': ["initiate"],
        'assistant_response': ["Great! How can I assist you today?"],
        'turn_number': [1]
    }
    dialogues_df = pd.DataFrame(data)

    cleaned_df = clean_data(dialogues_df)

    # Assertions
    assert 'User Utterance' in cleaned_df.columns
    assert 'Intent' in cleaned_df.columns
    assert 'Assistant Response' in cleaned_df.columns
    assert 'Turn Number' in cleaned_df.columns
    assert 'time_slot_start' in cleaned_df.columns
    assert 'time_slot_end' in cleaned_df.columns
    assert 'time_slot_description' in cleaned_df.columns
    assert cleaned_df['time_slot_start'].iloc[0] == 12
    assert cleaned_df['regions_str'].iloc[0] == "region_3"

def test_validate_data_success(mock_streamlit):
    # Sample valid dialogues DataFrame
    data = {
        'dialogue_id': ["dialogue_4"],
        'services': [["service_e"]],
        'num_lines': [2],
        'user_emotions': [["curious"]],
        'assistant_emotions': [["helpful"]],
        'scenario_category': ["category_4"],
        'generated_scenario': ["Helpful scenario."],
        'time_slot_start': [8],
        'time_slot_end': [10],
        'time_slot_description': ["08:00 - 10:00"],
        'regions': [["region_4"]],
        'resolution_status': ["resolved"],
        'User Utterance': ["How does this work?"],
        'Intent': ["inquiry"],
        'Assistant Response': ["Let me explain."],
        'Turn Number': [1]
    }
    dialogues_df = pd.DataFrame(data)

    assert validate_data(dialogues_df) == True

def test_validate_data_missing_columns(mock_streamlit):
    # Sample invalid dialogues DataFrame missing 'intent'
    data = {
        'dialogue_id': ["dialogue_5"],
        'services': [["service_f"]],
        'num_lines': [1],
        'user_emotions': [["neutral"]],
        'assistant_emotions': [["neutral"]],
        'scenario_category': ["category_5"],
        'generated_scenario': ["Neutral scenario."],
        'time_slot_start': [14],
        'time_slot_end': [16],
        'time_slot_description': ["14:00 - 16:00"],
        'regions': [["region_5"]],
        'resolution_status': ["unresolved"],
        'User Utterance': ["Nothing much."],
        # 'Intent' column is missing
        'Assistant Response': ["Alright."],
        'Turn Number': [1]
    }
    dialogues_df = pd.DataFrame(data)

    with pytest.raises(SystemExit):
        validate_data(dialogues_df)
    mock_streamlit.error.assert_called_with("Data is missing required columns: Intent")
    mock_streamlit.stop.assert_called_once()

def test_compute_sentiment_vader():
    from dashboard.dashboard import analyzer  # Import the sentiment analyzer

    positive_text = "I love this product!"
    negative_text = "This is the worst experience ever."
    neutral_text = "I have a meeting at 10 AM."

    positive_score = compute_sentiment_vader(positive_text)
    negative_score = compute_sentiment_vader(negative_text)
    neutral_score = compute_sentiment_vader(neutral_text)

    assert positive_score > 0.5
    assert negative_score < -0.5
    assert -0.1 <= neutral_score <= 0.1

def test_extract_entities_spacy():
    text = "Apple is looking at buying U.K. startup for $1 billion."
    entities = extract_entities_spacy(text)

    expected_entities = [
        ("Apple", "ORG"),
        ("U.K.", "GPE"),
        ("$1 billion", "MONEY")
    ]

    assert entities == expected_entities

def test_end_to_end_load_and_clean(mock_streamlit):
    # Create a mock JSONL file with multiple dialogues
    mock_data = [
        json.dumps({
            "dialogue_id": "dialogue_6",
            "services": ["service_g"],
            "num_lines": 2,
            "user_emotions": ["happy"],
            "assistant_emotions": ["happy"],
            "scenario_category": "category_6",
            "generated_scenario": "Happy scenario.",
            "time_slot": {"start": 10, "end": 12, "description": "10:00 - 12:00"},
            "regions": ["region_6"],
            "resolution_status": "resolved",
            "turns": [
                {
                    "utterance": "Good morning!",
                    "intent": "greeting",
                    "assistant_response": "Good morning! How can I help you today?",
                    "turn_number": 1
                },
                {
                    "utterance": "I need information on your services.",
                    "intent": "request_info",
                    "assistant_response": "Sure, here are the services we offer...",
                    "turn_number": 2
                }
            ]
        }),
        json.dumps({
            "dialogue_id": "dialogue_7",
            "services": ["service_h", "service_i"],
            "num_lines": 3,
            "user_emotions": ["frustrated"],
            "assistant_emotions": ["empathetic"],
            "scenario_category": "category_7",
            "generated_scenario": "Frustrated scenario.",
            "time_slot": {"start": 15, "end": 18, "description": "15:00 - 18:00"},
            "regions": ["region_7"],
            "resolution_status": "unresolved",
            "turns": [
                {
                    "utterance": "I'm having trouble with my account.",
                    "intent": "report_issue",
                    "assistant_response": "I'm sorry to hear that. Let me assist you.",
                    "turn_number": 1
                },
                {
                    "utterance": "It won't let me log in.",
                    "intent": "provide_details",
                    "assistant_response": "Let's try resetting your password.",
                    "turn_number": 2
                },
                {
                    "utterance": "Still not working.",
                    "intent": "escalate",
                    "assistant_response": "I'll escalate this issue to our support team.",
                    "turn_number": 3
                }
            ]
        })
    ]
    mock_file = io.BytesIO("\n".join(mock_data).encode('utf-8'))

    # Load data
    dialogues = load_data(mock_file)

    # Clean data
    cleaned_dialogues = clean_data(dialogues)

    # Validate data
    assert validate_data(cleaned_dialogues) == True

    # Assertions
    assert len(cleaned_dialogues) == 5  # Total turns across dialogues
    assert cleaned_dialogues['dialogue_id'].nunique() == 2
    assert cleaned_dialogues['Intent'].iloc[0] == "greeting"
    assert cleaned_dialogues['Assistant Response'].iloc[-1] == "I'll escalate this issue to our support team."

def test_sentiment_analysis_accuracy():
    from dashboard.dashboard import analyzer

    # Sample texts with known sentiments
    texts = [
        ("I am extremely happy with the service!", "Positive"),
        ("This is terrible and disappointing.", "Negative"),
        ("I have no strong feelings about this.", "Neutral")
    ]

    for text, expected in texts:
        score = compute_sentiment_vader(text)
        if expected == "Positive":
            assert score > 0.1
        elif expected == "Negative":
            assert score < -0.1
        else:
            assert -0.1 <= score <= 0.1

def test_extract_entities_spacy_no_entities():
    text = "This sentence has no named entities."
    entities = extract_entities_spacy(text)
    assert entities == []

def test_load_spacy_model():
    from dashboard.dashboard import load_spacy_model
    model = load_spacy_model()
    assert model is not None
    assert "en_core_web_sm" in model.meta["name"]