import streamlit as st
import json
import pandas as pd

# Load the JSON file
@st.cache_data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the data
data = load_data('../generated_dialogues.json')

st.header("📄 Dialogues Details")

# Pagination for Dialogues
total_dialogues = len(data)
dialogue_number = st.number_input("Dialogue Number", min_value=1, max_value=total_dialogues, value=1)

# Get the dialogue for the current page
dialogue = data[dialogue_number - 1]

# Display dialogue information
st.subheader(f"Dialogue ID: {dialogue['dialogue_id']}")
st.write(f"Services: {', '.join(dialogue['services'])}")
st.write(f"Scenario Category: {dialogue['scenario_category']}")
st.write(f"Time Slot: {dialogue['time_slot']}")
st.write(f"Regions: {', '.join(dialogue['regions'])}")
st.write(f"Resolution Status: {dialogue['resolution_status']}")
st.write(f"Total Turns: {len(dialogue['turns'])}")

# Display the full conversation
st.subheader("Conversation:")
for turn in dialogue['turns']:
    with st.chat_message("user"):
        st.markdown(f"**User ({turn['turn_number']}):** {turn['utterance']}")
        st.markdown(f"*Intent: {turn['intent']}*")
    
    with st.chat_message("assistant"):
        st.markdown(f"**Assistant:** {turn['assistant_response']}")

# Display emotions
st.subheader("Emotions:")
st.write(f"User Emotions: {', '.join(dialogue['user_emotions'])}")
st.write(f"Assistant Emotions: {', '.join(dialogue['assistant_emotions'])}")

# Display generated scenario
st.subheader("Generated Scenario:")
st.write(dialogue['generated_scenario'])

st.write(f"Showing dialogue {dialogue_number} of {total_dialogues}")