# dashboard.py

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter

# Set Streamlit page configuration
st.set_page_config(
    page_title="Dialogue Interaction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the Dashboard
st.title("📊 Dialogue Interaction Dashboard")

@st.cache_data
def load_data(uploaded_file):
    data = json.loads(uploaded_file.getvalue().decode('utf-8'))
    dialogues = pd.json_normalize(data)
    
    # Explode the turns column
    turns = dialogues.apply(lambda x: pd.Series(x['turns']), axis=1).stack().reset_index(level=1, drop=True)
    turns = pd.json_normalize(turns)
    
    # Merge turns with dialogue metadata
    dialogues = dialogues.drop('turns', axis=1).join(turns)
    
    dialogues.rename(columns={
        'utterance': 'User Utterance',
        'intent': 'Intent',
        'assistant_response': 'Assistant Response'
    }, inplace=True)
    
    return dialogues

# File uploader
data_file = st.file_uploader("Upload JSON file", type=["json"])

if data_file:
    dialogues = load_data(data_file)
else:
    st.warning("Please upload a JSON file to proceed.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")

# Scenario Category Filter
scenario_categories = dialogues['scenario_category'].unique().tolist()
selected_scenarios = st.sidebar.multiselect("Select Scenario Categories", options=scenario_categories, default=scenario_categories)

# Resolution Status Filter
resolution_statuses = dialogues['resolution_status'].unique().tolist()
selected_resolutions = st.sidebar.multiselect("Select Resolution Statuses", options=resolution_statuses, default=resolution_statuses)

# Time Slot Filter
time_slots = dialogues['time_slot'].unique().tolist()
selected_time_slots = st.sidebar.multiselect("Select Time Slots", options=time_slots, default=time_slots)

# Region Filter
regions = dialogues['regions'].explode().unique().tolist()
selected_regions = st.sidebar.multiselect("Select Regions", options=regions, default=regions)

# Apply filters
filtered_dialogues = dialogues[
    (dialogues['scenario_category'].isin(selected_scenarios)) &
    (dialogues['resolution_status'].isin(selected_resolutions)) &
    (dialogues['time_slot'].isin(selected_time_slots)) &
    (dialogues['regions'].apply(lambda x: any(region in x for region in selected_regions)))
]

st.sidebar.markdown("---")
st.sidebar.markdown("**Total Dialogues:** {}".format(filtered_dialogues['dialogue_id'].nunique()))
st.sidebar.markdown("**Total Turns:** {}".format(filtered_dialogues.shape[0]))

# Dataset Metadata
st.header("📊 Dataset Metadata")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Dialogues", dialogues['dialogue_id'].nunique())
    st.metric("Total Turns", dialogues.shape[0])
with col2:
    st.metric("Unique Services", len(dialogues['services'].explode().unique()))
    st.metric("Unique Scenario Categories", len(dialogues['scenario_category'].unique()))
with col3:
    st.metric("Unique User Emotions", len(dialogues['user_emotions'].explode().unique()))
    st.metric("Unique Assistant Emotions", len(dialogues['assistant_emotions'].explode().unique()))
with col4:
    st.metric("Unique Regions", len(dialogues['regions'].explode().unique()))
    st.metric("Unique Time Slots", len(dialogues['time_slot'].unique()))

# Main Dashboard Layout
# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Overview", "Emotions", "Categories", "Resolution", "Time Slots", "Regions", "Search", "Dialogue Viewer"])

# Tab 1: Overview
with tab1:
    st.header("🔍 Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_dialogues = filtered_dialogues['dialogue_id'].nunique()
        st.metric("Total Dialogues", total_dialogues)
    with col2:
        total_turns = filtered_dialogues.shape[0]
        st.metric("Total Turns", total_turns)
    with col3:
        unique_users = filtered_dialogues['user_emotions'].explode().nunique()
        unique_assistants = filtered_dialogues['assistant_emotions'].explode().nunique()
        st.metric("Unique User Emotions", unique_users)
        st.metric("Unique Assistant Emotions", unique_assistants)
    st.markdown("### Dialogues Distribution")
    # Services Distribution
    services_exploded = filtered_dialogues.explode('services')
    services_counts = services_exploded['services'].value_counts()
    fig_services = px.bar(
        x=services_counts.index,
        y=services_counts.values,
        labels={'x': 'Service', 'y': 'Count'},
        title='Services Distribution',
        text=services_counts.values
    )
    fig_services.update_traces(textposition='auto')
    st.plotly_chart(fig_services, use_container_width=True)

# Tab 2: Emotions
# Tab 2: Emotions
with tab2:
    st.header("😊 Emotions Distribution")
    tab2_sub1, tab2_sub2 = st.columns(2)
    
    with tab2_sub1:
        st.subheader("User Emotions")
        user_emotions = filtered_dialogues['user_emotions'].explode()
        user_emotion_counts = user_emotions.value_counts()
        fig_user_emotions = px.bar(
            x=user_emotion_counts.index,
            y=user_emotion_counts.values,
            labels={'x': 'User Emotion', 'y': 'Count'},
            title='User Emotions Distribution',
            text=user_emotion_counts.values
        )
        fig_user_emotions.update_traces(textposition='auto')
        st.plotly_chart(fig_user_emotions, use_container_width=True)
    
    with tab2_sub2:
        st.subheader("Assistant Emotions")
        assistant_emotions = filtered_dialogues['assistant_emotions'].explode()
        assistant_emotion_counts = assistant_emotions.value_counts()
        fig_assistant_emotions = px.bar(
            x=assistant_emotion_counts.index,
            y=assistant_emotion_counts.values,
            labels={'x': 'Assistant Emotion', 'y': 'Count'},
            title='Assistant Emotions Distribution',
            text=assistant_emotion_counts.values
        )
        fig_assistant_emotions.update_traces(textposition='auto')
        st.plotly_chart(fig_assistant_emotions, use_container_width=True)
# Tab 3: Categories
with tab3:
    st.header("📂 Scenario Categories")
    scenario_counts = filtered_dialogues['scenario_category'].value_counts()
    fig_scenarios = px.bar(
        x=scenario_counts.index,
        y=scenario_counts.values,
        labels={'x': 'Scenario Category', 'y': 'Count'},
        title='Scenario Categories Distribution',
        text=scenario_counts.values
    )
    fig_scenarios.update_traces(textposition='auto')
    st.plotly_chart(fig_scenarios, use_container_width=True)

# Tab 4: Resolution
with tab4:
    st.header("✅ Resolution Status")
    resolution_counts = filtered_dialogues['resolution_status'].value_counts()
    fig_resolution = px.pie(
        names=resolution_counts.index,
        values=resolution_counts.values,
        title='Resolution Status Distribution'
    )
    st.plotly_chart(fig_resolution, use_container_width=True)
    
    st.subheader("Average Number of Turns per Resolution Status")
    avg_turns_per_resolution = filtered_dialogues.groupby('resolution_status')['num_lines'].mean().round(2)
    fig_avg_turns = px.bar(
        x=avg_turns_per_resolution.index,
        y=avg_turns_per_resolution.values,
        labels={'x': 'Resolution Status', 'y': 'Average Number of Turns'},
        title='Average Number of Turns per Resolution Status',
        text=avg_turns_per_resolution.values
    )
    fig_avg_turns.update_traces(textposition='auto')
    st.plotly_chart(fig_avg_turns, use_container_width=True)

# Tab 5: Time Slots
with tab5:
    st.header("⏰ Time Slot Distribution")
    time_slot_counts = filtered_dialogues['time_slot'].value_counts()
    fig_time_slots = px.bar(
        x=time_slot_counts.index,
        y=time_slot_counts.values,
        labels={'x': 'Time Slot', 'y': 'Count'},
        title='Time Slot Distribution',
        text=time_slot_counts.values
    )
    fig_time_slots.update_traces(textposition='auto')
    st.plotly_chart(fig_time_slots, use_container_width=True)
    
    st.subheader("Dialogues Over Time Slots")
    fig_time_slots_line = px.line(
        x=time_slot_counts.index,
        y=time_slot_counts.values,
        labels={'x': 'Time Slot', 'y': 'Count'},
        title='Dialogues Over Time Slots'
    )
    st.plotly_chart(fig_time_slots_line, use_container_width=True)

# Tab 6: Regions
with tab6:
    st.header("🌍 Regional Distribution")
    region_counts = filtered_dialogues['regions'].explode().value_counts()
    fig_regions = px.bar(
        x=region_counts.index,
        y=region_counts.values,
        labels={'x': 'Region', 'y': 'Count'},
        title='Regional Distribution',
        text=region_counts.values
    )
    fig_regions.update_traces(textposition='auto')
    st.plotly_chart(fig_regions, use_container_width=True)
    
    st.subheader("Dialogues per Region and Scenario Category")
    dialogues_per_region_scenario = filtered_dialogues.explode('regions').groupby(['regions', 'scenario_category']).size().reset_index(name='count')
    fig_region_scenario = px.bar(
        dialogues_per_region_scenario,
        x='regions',
        y='count',
        color='scenario_category',
        labels={'regions': 'Region', 'count': 'Count', 'scenario_category': 'Scenario Category'},
        title='Dialogues per Region and Scenario Category',
        text='count'
    )
    fig_region_scenario.update_traces(textposition='auto')
    st.plotly_chart(fig_region_scenario, use_container_width=True)

# Tab 7: Advanced Search and Analysis
with tab7:
    st.header("🔍 Advanced Search and Analysis")

    # Multi-field search
    st.subheader("Multi-field Search")
    col1, col2 = st.columns(2)
    with col1:
        search_intent = st.text_input("Search by Intent")
        search_utterance = st.text_input("Search by User Utterance")
    with col2:
        search_emotion = st.text_input("Search by Emotion")
        search_response = st.text_input("Search by Assistant Response")

    if search_intent or search_utterance or search_emotion or search_response:
        search_filtered = filtered_dialogues[
            (filtered_dialogues['Intent'].str.contains(search_intent, case=False, na=False)) &
            (filtered_dialogues['User Utterance'].str.contains(search_utterance, case=False, na=False)) &
            (
                filtered_dialogues['user_emotions'].apply(lambda x: any(search_emotion.lower() in emotion.lower() for emotion in x)) |
                filtered_dialogues['assistant_emotions'].apply(lambda x: any(search_emotion.lower() in emotion.lower() for emotion in x))
            ) &
            (filtered_dialogues['Assistant Response'].str.contains(search_response, case=False, na=False))
        ]
        st.write(f"Found {len(search_filtered)} matching dialogues")
        st.dataframe(
            search_filtered[['dialogue_id', 'services', 'User Utterance', 'Intent', 'Assistant Response',
                             'user_emotions', 'assistant_emotions', 'scenario_category',
                             'time_slot', 'regions', 'resolution_status']]
        )

    # Intent Analysis
    st.subheader("Intent Analysis")
    all_intents = filtered_dialogues['Intent'].value_counts()
    top_intents = all_intents.head(10)
    fig_intents = px.bar(
        x=top_intents.index,
        y=top_intents.values,
        labels={'x': 'Intent', 'y': 'Count'},
        title='Top 10 Intents',
        text=top_intents.values
    )
    fig_intents.update_traces(textposition='auto')
    st.plotly_chart(fig_intents, use_container_width=True)

    # Word Cloud
    st.subheader("Word Cloud of User Utterances")
    try:
        from wordcloud import WordCloud
        text = " ".join(filtered_dialogues['User Utterance'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ImportError:
        st.warning("The WordCloud feature is not available. Please install the 'wordcloud' package to use this feature.")
        st.info("You can install it by running: pip install wordcloud")



with tab8:
    st.header("📄 Dialogue Viewer")

    # Load the JSON file
    @st.cache_data
    def load_dialogue_data(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    # Load the data
    dialogue_data = load_dialogue_data('../generated_dialogues.json')

    # Pagination for Dialogues
    total_dialogues = len(dialogue_data)
    dialogue_number = st.number_input("Dialogue Number", min_value=1, max_value=total_dialogues, value=1)

    # Get the dialogue for the current page
    dialogue = dialogue_data[dialogue_number - 1]

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

