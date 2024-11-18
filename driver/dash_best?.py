# dashboard.py

import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import re

# Set Streamlit page configuration
st.set_page_config(
    page_title="Dialogue Interaction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .metric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the Dashboard
st.title("📊 Dialogue Interaction Dashboard")

@st.cache_data
def load_data(uploaded_file):
    data = json.loads(uploaded_file.getvalue().decode('utf-8'))

    # Data Validation
    validate_data(data)

    # Flatten the data using pd.json_normalize
    dialogues = pd.json_normalize(
        data,
        record_path='turns',
        meta=[
            'dialogue_id', 'services', 'num_lines', 'user_emotions', 'assistant_emotions',
            'scenario_category', 'generated_scenario', 'time_slot', 'regions', 'resolution_status'
        ]
    )

    # Rename columns
    dialogues.rename(columns={
        'utterance': 'User Utterance',
        'intent': 'Intent',
        'assistant_response': 'Assistant Response',
        'turn_number': 'Turn Number'
    }, inplace=True)

    # Split 'time_slot' into separate columns
    time_slot_df = pd.DataFrame(dialogues['time_slot'].tolist(), index=dialogues.index)
    dialogues[['time_slot_start', 'time_slot_end', 'time_slot_description']] = time_slot_df

    # Handle possible missing values
    dialogues['services'] = dialogues['services'].apply(lambda x: x if isinstance(x, list) else [])
    dialogues['user_emotions'] = dialogues['user_emotions'].apply(lambda x: x if isinstance(x, list) else [])
    dialogues['assistant_emotions'] = dialogues['assistant_emotions'].apply(lambda x: x if isinstance(x, list) else [])
    dialogues['regions'] = dialogues['regions'].apply(lambda x: x if isinstance(x, list) else [])

    # Ensure correct data types
    dialogues['Turn Number'] = dialogues['Turn Number'].astype(int)
    dialogues['num_lines'] = dialogues['num_lines'].astype(int)
    dialogues['time_slot_start'] = dialogues['time_slot_start'].astype(int)
    dialogues['time_slot_end'] = dialogues['time_slot_end'].astype(int)

    return dialogues

def validate_data(data):
    """
    Validates the structure and content of the data.

    Args:
        data (list): List of dialogues loaded from the JSON file.

    Raises:
        ValueError: If any required fields are missing or data types are incorrect.
    """
    required_fields = {'dialogue_id', 'services', 'turns', 'num_lines', 'user_emotions', 'assistant_emotions',
                       'scenario_category', 'generated_scenario', 'time_slot', 'regions', 'resolution_status'}
    required_turn_fields = {'turn_number', 'utterance', 'intent', 'assistant_response'}

    for dialogue in data:
        dialogue_id = dialogue.get('dialogue_id', 'Unknown')
        # Check for missing fields in dialogue
        missing_fields = required_fields - dialogue.keys()
        if missing_fields:
            st.error(f"Dialogue '{dialogue_id}' is missing required fields: {', '.join(missing_fields)}")
            st.stop()

        # Validate data types
        if not isinstance(dialogue['services'], list):
            st.error(f"Dialogue '{dialogue_id}' - 'services' must be a list.")
            st.stop()
        if not isinstance(dialogue['user_emotions'], list):
            st.error(f"Dialogue '{dialogue_id}' - 'user_emotions' must be a list.")
            st.stop()
        if not isinstance(dialogue['assistant_emotions'], list):
            st.error(f"Dialogue '{dialogue_id}' - 'assistant_emotions' must be a list.")
            st.stop()
        if not isinstance(dialogue['regions'], list):
            st.error(f"Dialogue '{dialogue_id}' - 'regions' must be a list.")
            st.stop()
        if not isinstance(dialogue['num_lines'], int):
            st.error(f"Dialogue '{dialogue_id}' - 'num_lines' must be an integer.")
            st.stop()
        if not isinstance(dialogue['time_slot'], list) or len(dialogue['time_slot']) != 3:
            st.error(f"Dialogue '{dialogue_id}' - 'time_slot' must be a list of three elements.")
            st.stop()

        # Validate 'turns' structure
        if not isinstance(dialogue['turns'], list):
            st.error(f"Dialogue '{dialogue_id}' - 'turns' must be a list.")
            st.stop()
        for turn in dialogue['turns']:
            missing_turn_fields = required_turn_fields - turn.keys()
            if missing_turn_fields:
                st.error(f"Turn {turn.get('turn_number', 'Unknown')} in dialogue '{dialogue_id}' is missing required fields: {', '.join(missing_turn_fields)}")
                st.stop()

            # Validate turn data types
            if not isinstance(turn['turn_number'], int):
                st.error(f"Turn number in dialogue '{dialogue_id}' must be an integer.")
                st.stop()
            if not isinstance(turn['utterance'], str) or not turn['utterance'].strip():
                st.error(f"Turn {turn['turn_number']} in dialogue '{dialogue_id}' has an invalid 'utterance'.")
                st.stop()
            if not isinstance(turn['intent'], str) or not turn['intent'].strip():
                st.error(f"Turn {turn['turn_number']} in dialogue '{dialogue_id}' has an invalid 'intent'.")
                st.stop()
            if not isinstance(turn['assistant_response'], str) or not turn['assistant_response'].strip():
                st.error(f"Turn {turn['turn_number']} in dialogue '{dialogue_id}' has an invalid 'assistant_response'.")
                st.stop()

    # Optionally, return True if validation passes
    return True
def compute_sentiment(text):
    return TextBlob(text).sentiment.polarity

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
time_slots = dialogues['time_slot_description'].unique().tolist()
selected_time_slots = st.sidebar.multiselect("Select Time Slots", options=time_slots, default=time_slots)

# Region Filter
all_regions = dialogues['regions'].explode().dropna().unique().tolist()
selected_regions = st.sidebar.multiselect("Select Regions", options=all_regions, default=all_regions)

# Service Filter
all_services = dialogues['services'].explode().dropna().unique().tolist()
selected_services = st.sidebar.multiselect("Select Services", options=all_services, default=all_services)

# Emotion Filter
all_user_emotions = dialogues['user_emotions'].explode().dropna().unique().tolist()
selected_user_emotions = st.sidebar.multiselect("Select User Emotions", options=all_user_emotions, default=all_user_emotions)

all_assistant_emotions = dialogues['assistant_emotions'].explode().dropna().unique().tolist()
selected_assistant_emotions = st.sidebar.multiselect("Select Assistant Emotions", options=all_assistant_emotions, default=all_assistant_emotions)

# Apply filters
def extract_entities(text):
    # Simple regex patterns for demonstration
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|' \
                   r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|' \
                   r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
    time_pattern = r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b'
    location_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'  # Simple pattern for capitalized words

    dates = re.findall(date_pattern, text)
    times = re.findall(time_pattern, text)
    locations = re.findall(location_pattern, text)

    return {'dates': dates, 'times': times, 'locations': locations}

# Apply filters
filtered_dialogues = dialogues[
    (dialogues['scenario_category'].isin(selected_scenarios)) &
    (dialogues['resolution_status'].isin(selected_resolutions)) &
    (dialogues['time_slot_description'].isin(selected_time_slots)) &
    (dialogues['regions'].apply(lambda x: any(region in x for region in selected_regions))) &
    (dialogues['services'].apply(lambda x: any(service in x for service in selected_services))) &
    (dialogues['user_emotions'].apply(lambda x: any(emotion in x for emotion in selected_user_emotions))) &
    (dialogues['assistant_emotions'].apply(lambda x: any(emotion in x for emotion in selected_assistant_emotions)))
]

# Add sentiment analysis
filtered_dialogues['User Sentiment'] = filtered_dialogues['User Utterance'].apply(lambda x: compute_sentiment(str(x)))
filtered_dialogues['Assistant Sentiment'] = filtered_dialogues['Assistant Response'].apply(lambda x: compute_sentiment(str(x)))

# Add entity extraction
filtered_dialogues['User Entities'] = filtered_dialogues['User Utterance'].apply(lambda x: extract_entities(str(x)))
filtered_dialogues['Assistant Entities'] = filtered_dialogues['Assistant Response'].apply(lambda x: extract_entities(str(x)))


st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Dialogues:** {filtered_dialogues['dialogue_id'].nunique()}")
st.sidebar.markdown(f"**Total Turns:** {filtered_dialogues.shape[0]}")

# Dataset Metadata
st.header("📊 Dataset Metadata")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Dialogues", dialogues['dialogue_id'].nunique())
    st.metric("Total Turns", dialogues.shape[0])
with col2:
    st.metric("Unique Services", len(pd.Series([item for sublist in dialogues['services'] for item in sublist]).unique()))
    st.metric("Unique Scenario Categories", len(dialogues['scenario_category'].unique()))
with col3:
    st.metric("Unique User Emotions", len(pd.Series([item for sublist in dialogues['user_emotions'] for item in sublist]).unique()))
    st.metric("Unique Assistant Emotions", len(pd.Series([item for sublist in dialogues['assistant_emotions'] for item in sublist]).unique()))
with col4:
    st.metric("Unique Regions", len(pd.Series([item for sublist in dialogues['regions'] for item in sublist]).unique()))
    st.metric("Resolution Statuses", len(dialogues['resolution_status'].unique()))

# Sentiment Analysis Function

# Extract entities using regex (e.g., dates, times, locations)
def extract_entities(text):
    # Simple regex patterns for demonstration
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|' \
                   r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|' \
                   r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
    time_pattern = r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b'
    location_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'  # Simple pattern for capitalized words

    dates = re.findall(date_pattern, text)
    times = re.findall(time_pattern, text)
    locations = re.findall(location_pattern, text)

    return {'dates': dates, 'times': times, 'locations': locations}

dialogues['User Entities'] = dialogues['User Utterance'].apply(lambda x: extract_entities(str(x)))
dialogues['Assistant Entities'] = dialogues['Assistant Response'].apply(lambda x: extract_entities(str(x)))

# Main Dashboard Layout
# Tabs for different sections
tab_names = ["Overview", "Emotions", "Categories", "Resolution", "Time Slots", "Regions", "Services", "Sentiment Analysis", "Advanced Analysis", "Dialogue Viewer"]
tabs = st.tabs(tab_names)

# Tab 1: Overview
with tabs[0]:
    st.header("🔍 Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_dialogues = filtered_dialogues['dialogue_id'].nunique()
        st.metric("Total Dialogues", total_dialogues)
    with col2:
        total_turns = filtered_dialogues.shape[0]
        st.metric("Total Turns", total_turns)
    with col3:
        unique_users = len(pd.Series([item for sublist in filtered_dialogues['user_emotions'] for item in sublist]).unique())
        unique_assistants = len(pd.Series([item for sublist in filtered_dialogues['assistant_emotions'] for item in sublist]).unique())
        st.metric("Unique User Emotions", unique_users)
        st.metric("Unique Assistant Emotions", unique_assistants)
    st.markdown("### Dialogues Distribution")
    # Services Distribution
    services_exploded = filtered_dialogues.explode('services')
    services_counts = services_exploded['services'].value_counts()
    fig_services = px.pie(
        names=services_counts.index,
        values=services_counts.values,
        title='Services Distribution',
        hole=0.4
    )
    fig_services.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_services, use_container_width=True)

# Tab 2: Emotions
with tabs[1]:
    st.header("😊 Emotions Distribution")
    tab2_sub1, tab2_sub2 = st.columns(2)

    with tab2_sub1:
        st.subheader("User Emotions")
        user_emotions = filtered_dialogues['user_emotions'].explode()
        user_emotion_counts = user_emotions.value_counts()
        fig_user_emotions = px.pie(
            names=user_emotion_counts.index,
            values=user_emotion_counts.values,
            title='User Emotions Distribution',
            hole=0.4
        )
        fig_user_emotions.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_user_emotions, use_container_width=True)

    with tab2_sub2:
        st.subheader("Assistant Emotions")
        assistant_emotions = filtered_dialogues['assistant_emotions'].explode()
        assistant_emotion_counts = assistant_emotions.value_counts()
        fig_assistant_emotions = px.pie(
            names=assistant_emotion_counts.index,
            values=assistant_emotion_counts.values,
            title='Assistant Emotions Distribution',
            hole=0.4
        )
        fig_assistant_emotions.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_assistant_emotions, use_container_width=True)

    st.subheader("Emotion Co-occurrence Heatmap")
    # Create a co-occurrence matrix
    co_occurrence = pd.crosstab(user_emotions, assistant_emotions)
    fig_heatmap = px.imshow(
        co_occurrence,
        labels=dict(x="Assistant Emotions", y="User Emotions", color="Count"),
        title="Emotion Co-occurrence between User and Assistant",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Tab 3: Categories
with tabs[2]:
    st.header("📂 Scenario Categories")
    scenario_counts = filtered_dialogues['scenario_category'].value_counts()
    fig_scenarios = px.bar(
        x=scenario_counts.values,
        y=scenario_counts.index,
        orientation='h',
        labels={'x': 'Count', 'y': 'Scenario Category'},
        title='Scenario Categories Distribution',
        text=scenario_counts.values,
        color=scenario_counts.values,
        color_continuous_scale='Blues'
    )
    fig_scenarios.update_layout(showlegend=False)
    fig_scenarios.update_traces(textposition='outside')
    st.plotly_chart(fig_scenarios, use_container_width=True)

    st.subheader("Top Intents by Category")
    top_intents_by_category = filtered_dialogues.groupby('scenario_category')['Intent'].apply(lambda x: x.value_counts().head(3))
    st.write(top_intents_by_category)

# Tab 4: Resolution
with tabs[3]:
    st.header("✅ Resolution Status")
    resolution_counts = filtered_dialogues['resolution_status'].value_counts()
    fig_resolution = px.pie(
        names=resolution_counts.index,
        values=resolution_counts.values,
        title='Resolution Status Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_resolution.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_resolution, use_container_width=True)

    st.subheader("Average Number of Turns per Resolution Status")
    avg_turns_per_resolution = filtered_dialogues.groupby('resolution_status')['num_lines'].mean().round(2)
    fig_avg_turns = px.bar(
        x=avg_turns_per_resolution.index,
        y=avg_turns_per_resolution.values,
        labels={'x': 'Resolution Status', 'y': 'Average Number of Turns'},
        title='Average Number of Turns per Resolution Status',
        text=avg_turns_per_resolution.values,
        color=avg_turns_per_resolution.values,
        color_continuous_scale='Viridis'
    )
    fig_avg_turns.update_traces(textposition='auto')
    st.plotly_chart(fig_avg_turns, use_container_width=True)

# Tab 5: Time Slots
with tabs[4]:
    st.header("⏰ Time Slot Distribution")
    time_slot_counts = filtered_dialogues['time_slot_description'].value_counts()
    fig_time_slots = px.bar(
        x=time_slot_counts.index,
        y=time_slot_counts.values,
        labels={'x': 'Time Slot', 'y': 'Count'},
        title='Time Slot Distribution',
        text=time_slot_counts.values,
        color=time_slot_counts.values,
        color_continuous_scale='Agsunset'
    )
    fig_time_slots.update_traces(textposition='auto')
    st.plotly_chart(fig_time_slots, use_container_width=True)

    st.subheader("Dialogues Over Time Slots")
    fig_time_slots_line = px.line(
        x=time_slot_counts.index,
        y=time_slot_counts.values,
        labels={'x': 'Time Slot', 'y': 'Count'},
        title='Dialogues Over Time Slots',
        markers=True,
        color_discrete_sequence=['#FF6361']
    )
    st.plotly_chart(fig_time_slots_line, use_container_width=True)

# Tab 6: Regions
with tabs[5]:
    st.header("🌍 Regional Distribution")
    region_counts = filtered_dialogues['regions'].explode().value_counts()
    fig_regions = px.bar(
        x=region_counts.values,
        y=region_counts.index,
        orientation='h',
        labels={'x': 'Count', 'y': 'Region'},
        title='Regional Distribution',
        text=region_counts.values,
        color=region_counts.values,
        color_continuous_scale='tealrose'
    )
    fig_regions.update_traces(textposition='outside')
    st.plotly_chart(fig_regions, use_container_width=True)

    st.subheader("Dialogues per Region and Scenario Category")
    dialogues_per_region_scenario = filtered_dialogues.explode('regions').groupby(['regions', 'scenario_category']).size().reset_index(name='count')
    fig_region_scenario = px.sunburst(
        dialogues_per_region_scenario,
        path=['regions', 'scenario_category'],
        values='count',
        color='count',
        color_continuous_scale='RdBu',
        title='Dialogues per Region and Scenario Category'
    )
    st.plotly_chart(fig_region_scenario, use_container_width=True)

# Tab 7: Services
with tabs[6]:
    st.header("🛠️ Services Analysis")
    service_counts = services_exploded['services'].value_counts()
    fig_service_counts = px.bar(
        x=service_counts.values,
        y=service_counts.index,
        orientation='h',
        labels={'x': 'Count', 'y': 'Service'},
        title='Service Usage Distribution',
        text=service_counts.values,
        color=service_counts.values,
        color_continuous_scale='sunset'
    )
    fig_service_counts.update_traces(textposition='outside')
    st.plotly_chart(fig_service_counts, use_container_width=True)

    st.subheader("Top Intents by Service")
    top_intents_by_service = services_exploded.groupby('services')['Intent'].apply(lambda x: x.value_counts().head(3))
    st.write(top_intents_by_service)

# Tab 8: Sentiment Analysis
with tabs[7]:
    st.header("📈 Sentiment Analysis")
    st.subheader("User Sentiment Distribution")
    user_sentiment = filtered_dialogues['User Sentiment']
    fig_user_sentiment = px.histogram(
        user_sentiment,
        nbins=20,
        title='User Sentiment Score Distribution',
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_user_sentiment, use_container_width=True)

    st.subheader("Assistant Sentiment Distribution")
    assistant_sentiment = filtered_dialogues['Assistant Sentiment']
    fig_assistant_sentiment = px.histogram(
        assistant_sentiment,
        nbins=20,
        title='Assistant Sentiment Score Distribution',
        color_discrete_sequence=['#EF553B']
    )
    st.plotly_chart(fig_assistant_sentiment, use_container_width=True)

    st.subheader("Sentiment Over Turns")
    avg_sentiment_per_turn = filtered_dialogues.groupby('Turn Number').agg({
        'User Sentiment': 'mean',
        'Assistant Sentiment': 'mean'
    }).reset_index()
    fig_sentiment_over_turns = px.line(
        avg_sentiment_per_turn,
        x='Turn Number',
        y=['User Sentiment', 'Assistant Sentiment'],
        markers=True,
        title='Average Sentiment Over Turns'
    )
    st.plotly_chart(fig_sentiment_over_turns, use_container_width=True)

# Tab 9: Advanced Analysis
with tabs[8]:
    st.header("🔬 Advanced Analysis")

    st.subheader("Entity Extraction Overview")
    entity_types = ['dates', 'times', 'locations']
    entity_counts = {'Type': [], 'Count': []}
    for entity_type in entity_types:
        count = filtered_dialogues['User Entities'].apply(lambda x: len(x[entity_type])).sum()
        entity_counts['Type'].append(entity_type.capitalize())
        entity_counts['Count'].append(count)
    fig_entities = px.bar(
        x=entity_counts['Type'],
        y=entity_counts['Count'],
        labels={'x': 'Entity Type', 'y': 'Count'},
        title='Extracted Entities in User Utterances',
        text=entity_counts['Count'],
        color=entity_counts['Count'],
        color_continuous_scale='thermal'
    )
    fig_entities.update_traces(textposition='auto')
    st.plotly_chart(fig_entities, use_container_width=True)

    # st.subheader("Word Cloud of Assistant Responses")
    # text = " ".join(filtered_dialogues['Assistant Response'].dropna())
    # wordcloud = WordCloud(
    #     width=1200, 
    #     height=600,
    #     background_color='white',
    #     colormap='viridis',  # Use a better color scheme
    #     max_words=100,       # Limit number of words
    #     min_font_size=10,    # Set minimum font size
    #     max_font_size=150,   # Set maximum font size
    #     random_state=42,     # For reproducibility
    #     collocations=False   # Avoid splitting words
    # ).generate(text)

    # # Create figure with better size
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.axis('off')
    # plt.tight_layout(pad=0)
    # st.pyplot(fig)

# In the Advanced Analysis tab section
    st.subheader("Word Cloud Analysis")

    # Text preprocessing
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    # Common English stop words to remove
    custom_stop_words = set([
        'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'on', 'with',
        'be', 'this', 'will', 'can', 'at', 'by', 'an', 'are', 'so', 'it', 'as',
        'would', 'could', 'should', 'i', 'you', 'your', 'we', 'my', 'me', 'he',
        'she', 'they', 'am', 'is', 'are', 'was', 'were', 'been', 'being'
    ])

    # Prepare text
    text = " ".join(filtered_dialogues['Assistant Response'].dropna())
    processed_text = preprocess_text(text)

    # Create and configure word cloud
    wordcloud = WordCloud(
        width=2400,              # Higher resolution
        height=1200,
        background_color='white',
        colormap='viridis',       # More vibrant colormap
        max_words=150,           # Show more words
        min_font_size=8,
        max_font_size=160,
        random_state=42,
        collocations=False,
        stopwords=custom_stop_words,
        prefer_horizontal=0.7,   # 70% horizontal words for better readability
        font_path=None,          # Use default font, but you can specify a custom font file
        relative_scaling=0.5,    # Relative importance of frequencies
        contour_width=3,
        contour_color='steelblue'
    ).generate(processed_text)

    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='black')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_facecolor('black')
    plt.tight_layout(pad=0)

    # Add a subtle title
    plt.title('Most Common Terms in Assistant Responses', 
            color='white', 
            pad=20, 
            fontsize=16, 
            fontweight='bold')

    # Display metrics alongside the visualization
    col1, col2 = st.columns([3, 1])

    with col1:
        st.pyplot(fig)

    with col2:
        # Display some interesting metrics
        word_freq = Counter(processed_text.split())
        total_words = len(word_freq)
        unique_words = len(set(processed_text.split()))
        
        st.metric("Total Unique Words", unique_words)
        st.metric("Most Common Word", max(word_freq.items(), key=lambda x: x[1])[0])
        
        # Top 5 most common words
        st.write("**Top 5 Most Common Words:**")
        for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"- {word}: {count}")

    # Add explanatory text
    st.markdown("""
        <div style='padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <h4>About this Visualization</h4>
            <p>This word cloud visualizes the most frequently used terms in assistant responses. 
            The size of each word corresponds to its frequency in the conversations. 
            Common stop words have been removed to focus on meaningful content.</p>
        </div>
        """, unsafe_allow_html=True)




    st.subheader("Correlation Matrix")
    # Compute correlations
    numeric_cols = filtered_dialogues[['Turn Number', 'User Sentiment', 'Assistant Sentiment', 'num_lines']]
    corr_matrix = numeric_cols.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Correlation Matrix'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 10: Dialogue Viewer
with tabs[9]:
    st.header("📄 Dialogue Viewer")

    # Get unique dialogue IDs
    dialogue_ids = dialogues['dialogue_id'].unique().tolist()
    selected_dialogue_id = st.selectbox("Select Dialogue ID", options=dialogue_ids)

    dialogue = dialogues[dialogues['dialogue_id'] == selected_dialogue_id]

    # Display dialogue information
    st.subheader(f"Dialogue ID: {selected_dialogue_id}")
    st.write(f"**Services:** {', '.join(set([item for sublist in dialogue['services'] for item in sublist]))}")
    st.write(f"**Scenario Category:** {dialogue['scenario_category'].iloc[0]}")
    st.write(f"**Generated Scenario:** {dialogue['generated_scenario'].iloc[0]}")
    st.write(f"**Time Slot:** {dialogue['time_slot_description'].iloc[0]} ({int(dialogue['time_slot_start'].iloc[0]):02d}:00-{int(dialogue['time_slot_end'].iloc[0]):02d}:00)")
    st.write(f"**Regions:** {', '.join(set([item for sublist in dialogue['regions'] for item in sublist]))}")
    st.write(f"**Resolution Status:** {dialogue['resolution_status'].iloc[0]}")
    st.write(f"**Total Turns:** {dialogue['num_lines'].iloc[0]}")

    # Display emotions
    st.subheader("Emotions")
    user_emotions_set = set([item for sublist in dialogue['user_emotions'] for item in sublist])
    assistant_emotions_set = set([item for sublist in dialogue['assistant_emotions'] for item in sublist])
    st.write(f"**User Emotions:** {', '.join(user_emotions_set)}")
    st.write(f"**Assistant Emotions:** {', '.join(assistant_emotions_set)}")

    # Display the full conversation
    st.subheader("Conversation")
    for _, row in dialogue.sort_values('Turn Number').iterrows():
        st.markdown(f"**User (Turn {int(row['Turn Number'])}):** {row['User Utterance']}")
        st.markdown(f"*Intent: {row['Intent']}*")
        st.markdown(f"**Assistant:** {row['Assistant Response']}")
        st.markdown("---")

    st.write(f"Showing dialogue {dialogue_ids.index(selected_dialogue_id)+1} of {len(dialogue_ids)}")

# Footer
st.markdown("""
    <hr style='border:1px solid gray'>
    <center>Developed by Your Name | Powered by Streamlit</center>
    """, unsafe_allow_html=True)
