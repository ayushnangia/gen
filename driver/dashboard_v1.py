# dashboard.py

import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter, defaultdict
from wordcloud import WordCloud
from textblob import TextBlob
import re
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict

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

CORE_SERVICES = ['hotel', 'restaurant', 'train', 'attraction', 'taxi', 'bus', 'hospital', 'flight']

LOGICAL_COMBINATIONS = {
    'double': [
        # Hospital-related combinations
        ['hospital', 'taxi'],
        ['hospital', 'hotel'],
        # Flight-related combinations
        ['flight', 'taxi'],
        ['flight', 'hotel'],
        ['flight', 'train'],
        ['flight', 'bus'],
        ['flight', 'restaurant'],
        # Travel & Accommodation
        ['hotel', 'taxi'],
        ['hotel', 'train'],
        ['hotel', 'bus'],
        # Dining & Entertainment
        ['restaurant', 'taxi'],
        ['restaurant', 'attraction'],
        ['attraction', 'taxi'],
        # Transport Connections
        ['train', 'taxi'],
        ['bus', 'taxi'],
        ['train', 'bus'],
        # Common Pairings
        ['hotel', 'restaurant'],
        ['attraction', 'restaurant']
    ],
    'triple': [
        # Hospital-related combinations
        ['hospital', 'taxi', 'hotel'],
        ['hospital', 'hotel', 'restaurant'],
        # Travel & Stay Combinations
        ['hotel', 'restaurant', 'taxi'],
        ['hotel', 'train', 'taxi'],
        ['hotel', 'bus', 'taxi'],
        # Tourism Combinations
        ['attraction', 'restaurant', 'taxi'],
        ['attraction', 'hotel', 'taxi'],
        ['attraction', 'train', 'taxi'],
        # Extended Travel Plans
        ['train', 'hotel', 'restaurant'],
        ['bus', 'hotel', 'restaurant'],
        ['train', 'restaurant', 'taxi'],
        # Flight-related triples
        ['flight', 'hotel', 'taxi'],
        ['flight', 'train', 'taxi'],
        ['flight', 'bus', 'taxi'],
        ['flight', 'restaurant', 'taxi'],
    ],
    'quadruple': [
        # Hospital-related combinations
        ['hospital', 'hotel', 'restaurant', 'taxi'],
        # Full Tourism Package
        ['hotel', 'restaurant', 'attraction', 'taxi'],
        ['train', 'hotel', 'restaurant', 'taxi'],
        ['bus', 'hotel', 'restaurant', 'taxi'],
        # Extended Tourism
        ['train', 'hotel', 'attraction', 'taxi'],
        ['bus', 'hotel', 'attraction', 'taxi'],
        ['flight', 'hotel', 'restaurant', 'taxi'],
        ['flight', 'train', 'hotel', 'taxi'],
        ['flight', 'bus', 'hotel', 'taxi'],
    ]
}

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

# Main Dashboard Layout
# Tabs for different sections
tab_names = ["Overview", "Emotions", "Categories", "Resolution", "Time Slots", "Regions", "Sentiment Analysis", "Advanced Analysis", "Dialogue Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("🔍 Overview")
    
    # First row - Service Usage Statistics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Single Service Usage")
        
        # Get total unique dialogues
        total_dialogues = filtered_dialogues['dialogue_id'].nunique()
        
        # Filter for dialogues that use only one service and drop duplicates to get unique dialogues
        single_service_dialogues = filtered_dialogues[filtered_dialogues['services'].map(len) == 1].drop_duplicates(subset='dialogue_id')
        
        # Get service counts for single-service dialogues using str[0] for robustness
        service_counts = single_service_dialogues['services'].str[0].value_counts()
        
        # Donut chart for single services
        fig_standalone = px.pie(
            values=service_counts.values,
            names=service_counts.index,
            title=f'Single Service Distribution (Total Single-Service Dialogues: {service_counts.sum()})',
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_standalone.update_traces(
            textposition='outside',
            textinfo='label+percent+value',
            pull=[0.05] * len(service_counts)
        )
        fig_standalone.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_standalone, use_container_width=True)
    
    with col2:
        st.subheader("Quick Stats")
        # Count single and multi-service dialogues
        service_counts_per_dialogue = filtered_dialogues.groupby('dialogue_id')['services'].first().apply(len)
        single_service_count = (service_counts_per_dialogue == 1).sum()
        multi_service_count = (service_counts_per_dialogue > 1).sum()
        
        st.metric("Total Dialogues", total_dialogues)
        st.metric("Single Service Dialogues", single_service_count)
        st.metric("Multi-Service Dialogues", multi_service_count)
        
        # Add percentage breakdown
        st.write("**Percentage Breakdown:**")
        st.write(f"- Single Service: {single_service_count/total_dialogues*100:.1f}%")
        st.write(f"- Multi Service: {multi_service_count/total_dialogues*100:.1f}%")
    
    
    st.subheader("Multi-Service Analysis")
    
    # Filter multi-service dialogues ensuring unique service counts per dialogue
    multi_service_dialogues = filtered_dialogues[
        filtered_dialogues['services'].map(len) > 1
    ].drop_duplicates(subset='dialogue_id')

    # Ensure services are unique per dialogue to prevent overcounting
    multi_service_dialogues_unique = multi_service_dialogues.copy()
    multi_service_dialogues_unique['services'] = multi_service_dialogues_unique['services'].apply(lambda x: list(set(x)) if isinstance(x, list) else [])

    # Create tabs for different types of analysis
    multi_tabs = st.tabs(["Service Pairs", "Service Triples", "Service Quadruples"])

    with multi_tabs[0]:
        st.subheader("Common Service Pairs")
        pair_counts = defaultdict(int)
        
        for _, row in multi_service_dialogues_unique.iterrows():
            services_set = set(row['services'])
            for pair in LOGICAL_COMBINATIONS['double']:
                if set(pair).issubset(services_set):
                    pair_key = '+'.join(sorted(pair))
                    pair_counts[pair_key] += 1  # Increment count per dialogue containing the pair
        
        if pair_counts:
            pair_df = pd.DataFrame(list(pair_counts.items()), columns=['Pair', 'Count'])
            pair_df = pair_df.sort_values('Count', ascending=True)
            
            fig_pairs = px.bar(
                pair_df,
                x='Count',
                y='Pair',
                orientation='h',
                title='Common Service Pairs in Multi-Service Dialogues',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_pairs.update_layout(height=max(400, len(pair_counts) * 30))
            st.plotly_chart(fig_pairs, use_container_width=True)
        else:
            st.write("No service pairs found in the data.")
    
    with multi_tabs[1]:
        st.subheader("Common Service Triples")
        triple_counts = defaultdict(int)
        
        for _, row in multi_service_dialogues_unique.iterrows():
            services_set = set(row['services'])
            for triple in LOGICAL_COMBINATIONS['triple']:
                if set(triple).issubset(services_set):
                    triple_key = '+'.join(sorted(triple))
                    triple_counts[triple_key] += 1  # Increment count per dialogue containing the triple
        
        if triple_counts:
            triple_df = pd.DataFrame(list(triple_counts.items()), columns=['Triple', 'Count'])
            triple_df = triple_df.sort_values('Count', ascending=True)
            
            fig_triples = px.bar(
                triple_df,
                x='Count',
                y='Triple',
                orientation='h',
                title='Common Service Triples in Multi-Service Dialogues',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_triples.update_layout(height=max(400, len(triple_counts) * 30))
            st.plotly_chart(fig_triples, use_container_width=True)
        else:
            st.write("No service triples found in the data.")
    
    with multi_tabs[2]:
        st.subheader("Common Service Quadruples")
        quad_counts = defaultdict(int)
        
        for _, row in multi_service_dialogues_unique.iterrows():
            services_set = set(row['services'])
            for quad in LOGICAL_COMBINATIONS['quadruple']:
                if set(quad).issubset(services_set):
                    quad_key = '+'.join(sorted(quad))
                    quad_counts[quad_key] += 1  # Increment count per dialogue containing the quadruple
        
        if quad_counts:
            quad_df = pd.DataFrame(list(quad_counts.items()), columns=['Quadruple', 'Count'])
            quad_df = quad_df.sort_values('Count', ascending=True)
            
            fig_quads = px.bar(
                quad_df,
                x='Count',
                y='Quadruple',
                orientation='h',
                title='Common Service Quadruples in Multi-Service Dialogues',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_quads.update_layout(height=max(400, len(quad_counts) * 30))
            st.plotly_chart(fig_quads, use_container_width=True)
        else:
            st.write("No service quadruples found in the data.")
    
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
    
    # Get top 10 regions by dialogue count
    top_regions = (filtered_dialogues.explode('regions')
                .groupby('regions')
                .size()
                .nlargest(10)
                .index)

    # Get top 5 scenario categories
    top_categories = (filtered_dialogues['scenario_category']
                    .value_counts()
                    .nlargest(5)
                    .index)

    # Filter data for top regions and categories
    dialogues_summary = (filtered_dialogues[filtered_dialogues['scenario_category'].isin(top_categories)]
                        .explode('regions')
                        .query('regions in @top_regions')
                        .groupby(['regions', 'scenario_category'])
                        .size()
                        .reset_index(name='count'))

    # Create a simpler bar chart
    fig_region_scenario = px.bar(
        dialogues_summary,
        x='regions',
        y='count',
        color='scenario_category',
        title='Top 10 Regions by Most Common Scenario Categories',
        labels={
            'regions': 'Region',
            'count': 'Number of Dialogues',
            'scenario_category': 'Category'
        },
        barmode='stack'  # Stack bars for clearer view
    )

    # Clean up the layout
    fig_region_scenario.update_layout(
        xaxis_tickangle=-45,
        showlegend=True,
        legend_title_text='Category',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig_region_scenario, use_container_width=True)



with tabs[6]:
    st.header("📈 Sentiment Analysis")
    
    st.markdown("""
        ### What is Sentiment Analysis?
        Sentiment analysis helps us understand the emotional tone of conversations by analyzing the text. 
        The scores range from:
        - **-1.0** (Very Negative) 
        - **0.0** (Neutral)
        - **+1.0** (Very Positive)
    """)

    # Calculate basic statistics
    avg_user_sentiment = filtered_dialogues['User Sentiment'].mean()
    avg_assistant_sentiment = filtered_dialogues['Assistant Sentiment'].mean()
    
    # Display key metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Average User Sentiment", 
            f"{avg_user_sentiment:.2f}",
            delta=None,
            help="Average sentiment score of all user messages"
        )
    with col2:
        st.metric(
            "Average Assistant Sentiment",
            f"{avg_assistant_sentiment:.2f}",
            delta=None,
            help="Average sentiment score of all assistant responses"
        )

    # Simple bar chart comparing sentiment categories
    st.subheader("😊 Distribution of Sentiments")
    
    def get_sentiment_category(score):
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    filtered_dialogues['User Sentiment Category'] = filtered_dialogues['User Sentiment'].apply(get_sentiment_category)
    filtered_dialogues['Assistant Sentiment Category'] = filtered_dialogues['Assistant Sentiment'].apply(get_sentiment_category)
    
    user_sentiment_counts = filtered_dialogues['User Sentiment Category'].value_counts()
    assistant_sentiment_counts = filtered_dialogues['Assistant Sentiment Category'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(name='User', x=['Positive', 'Neutral', 'Negative'], 
               y=[user_sentiment_counts.get('Positive', 0),
                  user_sentiment_counts.get('Neutral', 0),
                  user_sentiment_counts.get('Negative', 0)]),
        go.Bar(name='Assistant', x=['Positive', 'Neutral', 'Negative'],
               y=[assistant_sentiment_counts.get('Positive', 0),
                  assistant_sentiment_counts.get('Neutral', 0),
                  assistant_sentiment_counts.get('Negative', 0)])
    ])
    
    fig.update_layout(
        title="Comparison of User and Assistant Sentiments",
        xaxis_title="Sentiment Category",
        yaxis_title="Number of Messages",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Show some example messages
    st.subheader("📝 Example Messages")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 😊 Most Positive")
        most_positive = filtered_dialogues.loc[filtered_dialogues['User Sentiment'].idxmax()]
        st.info(f"Score: {most_positive['User Sentiment']:.2f}\n\n{most_positive['User Utterance']}")
        
    with col2:
        st.markdown("### 😐 Most Neutral")
        neutral_idx = (filtered_dialogues['User Sentiment'] - 0).abs().idxmin()
        most_neutral = filtered_dialogues.loc[neutral_idx]
        st.info(f"Score: {most_neutral['User Sentiment']:.2f}\n\n{most_neutral['User Utterance']}")
        
    with col3:
        st.markdown("### 😔 Most Negative")
        most_negative = filtered_dialogues.loc[filtered_dialogues['User Sentiment'].idxmin()]
        st.info(f"Score: {most_negative['User Sentiment']:.2f}\n\n{most_negative['User Utterance']}")

    # Key Findings
    st.subheader("🔍 Key Findings")
    
    findings = []
    
    # Compare average sentiments
    if avg_user_sentiment > avg_assistant_sentiment:
        findings.append("Users generally express more positive sentiments than the assistant.")
    else:
        findings.append("The assistant generally maintains a more positive tone than users.")
    
    # Check most common sentiment
    most_common_user = user_sentiment_counts.index[0]
    findings.append(f"The most common sentiment from users is {most_common_user.lower()}.")
    
    # Calculate percentage of neutral responses
    neutral_percentage = (user_sentiment_counts.get('Neutral', 0) / len(filtered_dialogues)) * 100
    findings.append(f"Approximately {neutral_percentage:.1f}% of user messages are neutral in tone.")
    
    for finding in findings:
        st.markdown(f"- {finding}")

    st.markdown("""
        ### 📚 Understanding the Results
        - **Positive sentiment** often indicates satisfaction, agreement, or happiness
        - **Neutral sentiment** typically appears in factual statements or simple queries
        - **Negative sentiment** might show frustration, disagreement, or problems
        
        This analysis helps us understand the overall tone of conversations and how well the assistant maintains a positive interaction.
    """)



with tabs[7]:
    st.header("🔬 Advanced Analysis")

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
        if word_freq:
            most_common_word, most_common_count = word_freq.most_common(1)[0]
            st.metric("Most Common Word", most_common_word, delta=most_common_count)
        else:
            st.metric("Most Common Word", "N/A")
        
        # Top 5 most common words
        st.write("**Top 5 Most Common Words:**")
        top_5 = word_freq.most_common(5)
        for word, count in top_5:
            st.write(f"- {word}: {count}")

    st.markdown("---")
    
    st.subheader("📊 Conversation Patterns")
    
    # Average conversation length by scenario
    avg_turns = filtered_dialogues.groupby('scenario_category')['Turn Number'].mean().round(1)
    fig_turns = px.bar(
        x=avg_turns.index,
        y=avg_turns.values,
        title='Average Conversation Length by Scenario',
        labels={'x': 'Scenario Category', 'y': 'Average Number of Turns'},
        color=avg_turns.values,
        color_continuous_scale='Viridis'
    )
    fig_turns.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_turns, use_container_width=True)
    
    st.markdown("""
        **Key Insight:** This chart shows which types of scenarios typically need 
        more back-and-forth conversation to resolve. Longer conversations might 
        indicate more complex requests.
    """)


    st.markdown("---")
    
    st.subheader("🎯 Intent Analysis")
    
    # Get intent counts
    intent_counts = filtered_dialogues['Intent'].value_counts().head(10)
    total_intents = filtered_dialogues['Intent'].nunique()
    
    # Display metric
    st.metric(
        "Total Unique Intents",
        total_intents,
        help="Total number of different intents in the dataset"
    )
    
    # Create visualization
    fig_intent = px.bar(
        x=intent_counts.index,
        y=intent_counts.values,
        title='Top 10 Most Common User Intents',
        labels={'x': 'Intent', 'y': 'Number of Occurrences'},
        color=intent_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig_intent.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_intent, use_container_width=True)


with tabs[8]:
    st.header("💬 Dialogue Explorer")

    # Enhanced Dark Mode styling with comprehensive UI
    st.markdown("""
        <style>
            /* Dark theme colors */
            :root {
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --accent-blue: #4d9fff;
                --accent-green: #4CAF50;
                --text-primary: #ffffff;
                --text-secondary: #b0b0b0;
                --border-color: #404040;
            }

            /* Containers */
            .container {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                color: var(--text-primary);
            }

            /* Scenario box */
            .scenario-box {
                background-color: #2c3e50;
                border-left: 4px solid var(--accent-blue);
                padding: 20px;
                margin: 10px 0;
                border-radius: 4px;
                color: var(--text-primary);
            }

            .scenario-text {
                font-size: 1.1em;
                line-height: 1.6;
                color: #ecf0f1;
            }

            /* Messages */
            .user-message {
                background-color: #2c3e50;
                border-left: 4px solid var(--accent-blue);
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                color: var(--text-primary);
            }
            
            .assistant-message {
                background-color: #2d3436;
                border-left: 4px solid var(--accent-green);
                padding: 15px;
                margin: 10px 0 10px 20px;
                border-radius: 4px;
                color: var(--text-primary);
            }
            
            /* Headers and labels */
            .header {
                color: var(--accent-blue);
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 15px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 8px;
            }
            
            .label {
                color: var(--text-secondary);
                font-size: 0.9em;
                margin-right: 5px;
            }
            
            /* Status indicators */
            .status {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.9em;
                font-weight: 500;
            }
            
            .status-resolved {
                background-color: #2e7d32;
                color: white;
            }
            
            .status-unresolved {
                background-color: #c62828;
                color: white;
            }
            
            /* Navigation */
            .nav-text {
                color: var(--text-secondary);
                text-align: center;
                font-size: 1.1em;
                margin: 10px 0;
            }

            /* Tags */
            .metadata-tag {
                background-color: #3d3d3d;
                color: var(--text-primary);
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px;
                display: inline-block;
            }

            .metadata-tag-sub {
                background-color: #2c3e50;
                color: var(--text-secondary);
                padding: 2px 6px;
                border-radius: 4px;
                margin: 2px;
                font-size: 0.9em;
            }

            .emotion-tag {
                background-color: #2c3e50;
                color: var(--text-primary);
                padding: 4px 8px;
                border-radius: 12px;
                margin: 2px;
                display: inline-block;
                border: 1px solid var(--accent-blue);
            }

            /* Emotion boxes */
            .emotion-box {
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }

            .emotion-header {
                color: var(--accent-blue);
                margin-bottom: 10px;
                font-weight: bold;
            }

            /* Info groups */
            .info-group {
                margin: 10px 0;
                padding: 10px;
                background-color: rgba(0,0,0,0.2);
                border-radius: 4px;
            }

            /* Intent tag */
            .intent-tag {
                background-color: #3d3d3d;
                color: var(--text-secondary);
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.85em;
                margin-top: 5px;
                display: inline-block;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'dialogue_index' not in st.session_state:
        st.session_state.dialogue_index = 0
        st.session_state.dialogue_number = 1  # Initialize dialogue_number

    # Get unique dialogue IDs
    unique_dialogue_ids = filtered_dialogues['dialogue_id'].unique()
    total_dialogues = len(unique_dialogue_ids)

    # Callback function to update dialogue_index based on dialogue_number
    def update_dialogue_index():
        st.session_state.dialogue_index = st.session_state.dialogue_number - 1

    # Navigation Controls
    nav_col2, nav_col3 = st.columns([2, 2])
    
    
    with nav_col2:
        st.markdown(f"""
            <div class='nav-text'>
                Viewing dialogue {st.session_state.dialogue_number} of {total_dialogues}
            </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        dialogue_number = st.number_input(
            "Go to dialogue number",
            min_value=1,
            max_value=total_dialogues,
            value=st.session_state.dialogue_number,
            step=1,
            on_change=update_dialogue_index,
            key='dialogue_number',
            help="Enter a number between 1 and " + str(total_dialogues)
        )
    
    # Display Current Dialogue
    if total_dialogues > 0:
        current_dialogue_id = unique_dialogue_ids[st.session_state.dialogue_index]
        dialogue = filtered_dialogues[filtered_dialogues['dialogue_id'] == current_dialogue_id]
    else:
        st.warning("No dialogues available with the selected filters.")
        st.stop()

    # Scenario Section
    st.markdown("""
        <div class='header'>📝 Scenario</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='scenario-box'>
            <p class='scenario-text'>{dialogue['generated_scenario'].iloc[0]}</p>
        </div>
    """, unsafe_allow_html=True)

    # Main Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='header'>📌 Basic Information</div>
            <div class='info-group'>
                <p><span class='label'>Category:</span> 
                    <span class='metadata-tag'>{dialogue['scenario_category'].iloc[0]}</span>
                </p>
                <p><span class='label'>Services:</span><br> 
                    {"".join([f"<span class='metadata-tag'>{service}</span> " for service in dialogue['services'].iloc[0]])}
                </p>
                <p><span class='label'>Status:</span> 
                    <span class='status status-{dialogue['resolution_status'].iloc[0]}'>
                        {dialogue['resolution_status'].iloc[0].upper()}
                    </span>
                </p>
                <p><span class='label'>Total Turns:</span> 
                    <span class='metadata-tag'>{dialogue['num_lines'].iloc[0]}</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='header'>🔍 Context Information</div>
            <div class='info-group'>
                <p><span class='label'>Time Slot:</span> 
                    <span class='metadata-tag'>{dialogue['time_slot_description'].iloc[0]}</span>
                    <span class='metadata-tag-sub'>({dialogue['time_slot_start'].iloc[0]}:00 - {dialogue['time_slot_end'].iloc[0]}:00)</span>
                </p>
                <p><span class='label'>Regions:</span><br>
                    {"".join([f"<span class='metadata-tag'>{region}</span> " for region in dialogue['regions'].iloc[0]])}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Emotions Section
    st.markdown("<div class='header'>🎭 Emotions</div>", unsafe_allow_html=True)
    emo_col1, emo_col2 = st.columns(2)
    
    with emo_col1:
        st.markdown(f"""
            <div class='emotion-box'>
                <p class='emotion-header'>👤 User Emotions</p>
                {"".join([f"<span class='emotion-tag'>{emotion}</span> " for emotion in dialogue['user_emotions'].iloc[0]])}
            </div>
        """, unsafe_allow_html=True)
    
    with emo_col2:
        st.markdown(f"""
            <div class='emotion-box'>
                <p class='emotion-header'>🤖 Assistant Emotions</p>
                {"".join([f"<span class='emotion-tag'>{emotion}</span> " for emotion in dialogue['assistant_emotions'].iloc[0]])}
            </div>
        """, unsafe_allow_html=True)

    # Conversation Section
    st.markdown("<div class='header'>💭 Conversation</div>", unsafe_allow_html=True)
    
    for _, turn in dialogue.sort_values('Turn Number').iterrows():
        # User message
        st.markdown(f"""
            <div class='user-message'>
                <strong>👤 User (Turn {int(turn['Turn Number'])})</strong><br>
                {turn['User Utterance']}<br>
                <span class='intent-tag'>🎯 Intent: {turn['Intent']}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        st.markdown(f"""
            <div class='assistant-message'>
                <strong>🤖 Assistant</strong><br>
                {turn['Assistant Response']}
            </div>
        """, unsafe_allow_html=True)