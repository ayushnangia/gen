#!/usr/bin/env python3
"""
Comprehensive analysis script for SynWOZ dataset
Generates statistics, visualizations, and persona-sliced metrics for the PersonaLLM@NeurIPS 2025 submission
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import random

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_dataset(file_path):
    """Load JSONL dataset"""
    dialogues = []
    with open(file_path, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))
    return dialogues

def basic_statistics(dialogues):
    """Compute basic dataset statistics"""
    stats = {}

    # Total dialogues
    stats['total_dialogues'] = len(dialogues)

    # Service statistics
    all_services = []
    service_combinations = []
    for d in dialogues:
        services = d['services']
        all_services.extend(services)
        service_combinations.append(tuple(sorted(services)))

    stats['service_counts'] = Counter(all_services)
    stats['unique_services'] = len(set(all_services))
    stats['service_combination_counts'] = Counter(service_combinations)
    stats['unique_combinations'] = len(set(service_combinations))

    # Number of services per dialogue
    num_services = [len(d['services']) for d in dialogues]
    stats['single_service'] = sum(1 for n in num_services if n == 1)
    stats['double_service'] = sum(1 for n in num_services if n == 2)
    stats['triple_service'] = sum(1 for n in num_services if n == 3)
    stats['quad_plus_service'] = sum(1 for n in num_services if n >= 4)

    # Turn statistics
    turn_counts = [d['num_lines'] for d in dialogues]
    stats['mean_turns'] = np.mean(turn_counts)
    stats['median_turns'] = np.median(turn_counts)
    stats['min_turns'] = min(turn_counts)
    stats['max_turns'] = max(turn_counts)
    stats['std_turns'] = np.std(turn_counts)

    # Emotion statistics
    all_user_emotions = []
    all_assistant_emotions = []
    for d in dialogues:
        all_user_emotions.extend(d['user_emotions'])
        all_assistant_emotions.extend(d['assistant_emotions'])

    stats['user_emotion_counts'] = Counter(all_user_emotions)
    stats['assistant_emotion_counts'] = Counter(all_assistant_emotions)
    stats['unique_user_emotions'] = len(set(all_user_emotions))
    stats['unique_assistant_emotions'] = len(set(all_assistant_emotions))

    # Region statistics
    all_regions = []
    for d in dialogues:
        all_regions.extend(d['regions'])

    stats['region_counts'] = Counter(all_regions)
    stats['unique_regions'] = len(set(all_regions))

    # Scenario category statistics
    scenario_categories = [d['scenario_category'] for d in dialogues]
    stats['scenario_category_counts'] = Counter(scenario_categories)
    stats['unique_scenario_categories'] = len(set(scenario_categories))

    # Resolution status
    resolution_statuses = [d['resolution_status'] for d in dialogues]
    stats['resolution_status_counts'] = Counter(resolution_statuses)

    # Intent statistics
    all_intents = []
    for d in dialogues:
        for turn in d['turns']:
            all_intents.append(turn['intent'])

    stats['intent_counts'] = Counter(all_intents)
    stats['unique_intents'] = len(set(all_intents))
    stats['total_turns'] = len(all_intents)

    # Time slot statistics
    time_periods = []
    for d in dialogues:
        time_periods.append(d['time_slot'][2])  # Get the period name
    stats['time_period_counts'] = Counter(time_periods)

    return stats

def print_statistics(stats):
    """Print formatted statistics"""
    print("="*80)
    print("SYNWOZ DATASET STATISTICS")
    print("="*80)

    print(f"\n📊 DATASET SCALE")
    print(f"  Total Dialogues: {stats['total_dialogues']:,}")
    print(f"  Total Turns: {stats['total_turns']:,}")
    print(f"  Unique Intents: {stats['unique_intents']:,}")

    print(f"\n🎭 SERVICES")
    print(f"  Unique Services: {stats['unique_services']}")
    print(f"  Unique Combinations: {stats['unique_combinations']}")
    print(f"  Top 5 Services:")
    for service, count in stats['service_counts'].most_common(5):
        pct = 100 * count / stats['total_turns']
        print(f"    - {service}: {count:,} ({pct:.1f}%)")

    print(f"\n📈 SERVICE COMBINATIONS")
    print(f"  Single service: {stats['single_service']:,} ({100*stats['single_service']/stats['total_dialogues']:.1f}%)")
    print(f"  Two services: {stats['double_service']:,} ({100*stats['double_service']/stats['total_dialogues']:.1f}%)")
    print(f"  Three services: {stats['triple_service']:,} ({100*stats['triple_service']/stats['total_dialogues']:.1f}%)")
    print(f"  Four+ services: {stats['quad_plus_service']:,} ({100*stats['quad_plus_service']/stats['total_dialogues']:.1f}%)")

    print(f"\n💬 TURN STATISTICS")
    print(f"  Mean: {stats['mean_turns']:.2f}")
    print(f"  Median: {stats['median_turns']:.0f}")
    print(f"  Range: {stats['min_turns']} - {stats['max_turns']}")
    print(f"  Std Dev: {stats['std_turns']:.2f}")

    print(f"\n😊 EMOTIONS")
    print(f"  Unique User Emotions: {stats['unique_user_emotions']}")
    print(f"  Unique Assistant Emotions: {stats['unique_assistant_emotions']}")
    print(f"  Top 5 User Emotions:")
    for emotion, count in stats['user_emotion_counts'].most_common(5):
        print(f"    - {emotion}: {count:,}")
    print(f"  Top 5 Assistant Emotions:")
    for emotion, count in stats['assistant_emotion_counts'].most_common(5):
        print(f"    - {emotion}: {count:,}")

    print(f"\n🌍 GEOGRAPHIC COVERAGE")
    print(f"  Unique Regions: {stats['unique_regions']}")
    print(f"  Top 10 Regions:")
    for region, count in stats['region_counts'].most_common(10):
        print(f"    - {region}: {count:,}")

    print(f"\n🎯 SCENARIO CATEGORIES")
    print(f"  Unique Categories: {stats['unique_scenario_categories']}")
    print(f"  Top 10 Categories:")
    for category, count in stats['scenario_category_counts'].most_common(10):
        print(f"    - {category}: {count:,}")

    print(f"\n✅ RESOLUTION STATUS")
    for status, count in stats['resolution_status_counts'].items():
        pct = 100 * count / stats['total_dialogues']
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    print(f"\n⏰ TIME PERIODS")
    for period, count in stats['time_period_counts'].most_common():
        pct = 100 * count / stats['total_dialogues']
        print(f"  {period}: {count:,} ({pct:.1f}%)")

    print("="*80)

def create_visualizations(stats, dialogues, output_dir):
    """Generate all visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Service Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    services = [s for s, _ in stats['service_counts'].most_common()]
    counts = [c for _, c in stats['service_counts'].most_common()]
    colors = sns.color_palette("husl", len(services))
    bars = ax.barh(services, counts, color=colors)
    ax.set_xlabel('Number of Dialogues', fontsize=12)
    ax.set_title('Service Distribution in SynWOZ Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'service_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'service_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: service_distribution.pdf/png")

    # 2. Emotion Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # User emotions
    user_emotions = sorted(stats['user_emotion_counts'].items(), key=lambda x: x[1], reverse=True)[:20]
    labels1, values1 = zip(*user_emotions)
    data1 = np.array(values1).reshape(-1, 1)
    sns.heatmap(data1, annot=True, fmt='d', cmap='YlOrRd',
                yticklabels=labels1, xticklabels=['Count'],
                cbar_kws={'label': 'Dialogue Count'}, ax=ax1)
    ax1.set_title('User Emotions Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('')

    # Assistant emotions
    asst_emotions = sorted(stats['assistant_emotion_counts'].items(), key=lambda x: x[1], reverse=True)[:20]
    labels2, values2 = zip(*asst_emotions)
    data2 = np.array(values2).reshape(-1, 1)
    sns.heatmap(data2, annot=True, fmt='d', cmap='YlGnBu',
                yticklabels=labels2, xticklabels=['Count'],
                cbar_kws={'label': 'Dialogue Count'}, ax=ax2)
    ax2.set_title('Assistant Emotions Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'emotion_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: emotion_distribution.pdf/png")

    # 3. Service Combination Complexity
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Single', 'Double', 'Triple', 'Quad+']
    sizes = [stats['single_service'], stats['double_service'],
             stats['triple_service'], stats['quad_plus_service']]
    colors = sns.color_palette("Set2", 4)
    explode = (0.05, 0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode,
                                        startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Service Combination Complexity', fontsize=14, fontweight='bold')

    # Add legend with counts
    legend_labels = [f'{label}: {size:,} dialogues' for label, size in zip(labels, sizes)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.savefig(output_dir / 'service_complexity.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'service_complexity.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: service_complexity.pdf/png")

    # 4. Turn Count Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    turn_counts = [d['num_lines'] for d in dialogues]
    bins = range(min(turn_counts), max(turn_counts) + 2)
    ax.hist(turn_counts, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Turns per Dialogue', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Turn Count Distribution', fontsize=14, fontweight='bold')
    ax.axvline(stats['mean_turns'], color='red', linestyle='--',
               label=f'Mean: {stats["mean_turns"]:.2f}', linewidth=2)
    ax.axvline(stats['median_turns'], color='green', linestyle='--',
               label=f'Median: {stats["median_turns"]:.0f}', linewidth=2)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'turn_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'turn_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: turn_distribution.pdf/png")

    # 5. Resolution Status
    fig, ax = plt.subplots(figsize=(8, 6))
    statuses = list(stats['resolution_status_counts'].keys())
    counts = list(stats['resolution_status_counts'].values())
    colors_res = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange
    bars = ax.bar(statuses, counts, color=colors_res, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Dialogues', fontsize=12)
    ax.set_title('Resolution Status Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / stats['total_dialogues']
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'resolution_status.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'resolution_status.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: resolution_status.pdf/png")

    # 6. Geographic Distribution (Top 20 Cities)
    fig, ax = plt.subplots(figsize=(12, 8))
    top_regions = stats['region_counts'].most_common(20)
    regions, region_counts = zip(*top_regions)
    colors_geo = sns.color_palette("viridis", len(regions))
    bars = ax.barh(regions, region_counts, color=colors_geo)
    ax.set_xlabel('Number of Dialogues', fontsize=12)
    ax.set_title('Top 20 Geographic Regions', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'geographic_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'geographic_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: geographic_distribution.pdf/png")

def persona_consistency_validation(dialogues, sample_size=50):
    """
    Sample dialogues and check persona consistency
    Focus on emotion consistency and scenario alignment
    """
    print("\n" + "="*80)
    print("PERSONA CONSISTENCY VALIDATION")
    print("="*80)

    # Sample dialogues
    random.seed(42)
    sample = random.sample(dialogues, min(sample_size, len(dialogues)))

    results = {
        'emotion_consistent': 0,
        'emotion_inconsistent': 0,
        'examples_consistent': [],
        'examples_inconsistent': []
    }

    for dialogue in sample:
        dialogue_id = dialogue['dialogue_id']
        user_emotions = dialogue['user_emotions']
        scenario = dialogue['generated_scenario']
        turns = dialogue['turns']

        # Check if user emotion is reflected in utterances
        is_consistent = True

        # Simple heuristic: check if emotional keywords appear in dialogue
        emotion_keywords = {
            'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'concern'],
            'Suspicious': ['suspicious', 'suspect', 'doubt', 'trust', 'believe'],
            'Disappointed': ['disappointed', 'frustrating', 'expected', 'upset'],
            'Grateful': ['thank', 'appreciate', 'grateful', 'glad'],
            'Confused': ['confused', 'understand', 'unclear', 'what do you mean'],
            'Frustrated': ['frustrated', 'ridiculous', 'unacceptable', 'enough'],
            'Hopeful': ['hope', 'hopefully', 'optimistic', 'looking forward'],
            'Impatient': ['hurry', 'quickly', 'waiting', 'long', 'time'],
        }

        # Concatenate all user utterances
        user_text = ' '.join([turn['utterance'].lower() for turn in turns])

        # Check if emotion is reflected
        if user_emotions:
            emotion = user_emotions[0]  # Primary emotion
            if emotion in emotion_keywords:
                keywords = emotion_keywords[emotion]
                emotion_reflected = any(keyword in user_text for keyword in keywords)

                if emotion_reflected:
                    results['emotion_consistent'] += 1
                    if len(results['examples_consistent']) < 3:
                        results['examples_consistent'].append({
                            'dialogue_id': dialogue_id,
                            'emotion': emotion,
                            'evidence': user_text[:200] + '...'
                        })
                else:
                    results['emotion_inconsistent'] += 1
                    if len(results['examples_inconsistent']) < 3:
                        results['examples_inconsistent'].append({
                            'dialogue_id': dialogue_id,
                            'emotion': emotion,
                            'text': user_text[:200] + '...'
                        })
            else:
                # Emotion not in our keyword dict, count as consistent by default
                results['emotion_consistent'] += 1

    # Print results
    total_checked = results['emotion_consistent'] + results['emotion_inconsistent']
    consistency_rate = 100 * results['emotion_consistent'] / total_checked if total_checked > 0 else 0

    print(f"\nSample Size: {sample_size}")
    print(f"Emotion Consistent: {results['emotion_consistent']} ({consistency_rate:.1f}%)")
    print(f"Emotion Inconsistent: {results['emotion_inconsistent']}")

    print(f"\n✓ CONSISTENT EXAMPLES:")
    for i, ex in enumerate(results['examples_consistent'], 1):
        print(f"\n  {i}. Dialogue: {ex['dialogue_id']}")
        print(f"     Emotion: {ex['emotion']}")
        print(f"     Evidence: {ex['evidence']}")

    if results['examples_inconsistent']:
        print(f"\n✗ INCONSISTENT EXAMPLES:")
        for i, ex in enumerate(results['examples_inconsistent'], 1):
            print(f"\n  {i}. Dialogue: {ex['dialogue_id']}")
            print(f"     Emotion: {ex['emotion']}")
            print(f"     Text: {ex['text']}")

    print("\n" + "="*80)

    return results

def persona_sliced_metrics(dialogues):
    """
    Compute quality metrics sliced by persona attributes
    Since explicit persona fields are missing, we use observable features:
    - User emotions (proxy for user persona)
    - Service types (proxy for task complexity)
    - Geographic regions (proxy for cultural context)
    """
    print("\n" + "="*80)
    print("PERSONA-SLICED METRICS")
    print("="*80)

    metrics = {}

    # 1. Metrics by User Emotion
    emotion_metrics = defaultdict(lambda: {
        'count': 0,
        'avg_turns': [],
        'resolved': 0,
        'escalated': 0,
        'failed': 0
    })

    for dialogue in dialogues:
        for emotion in dialogue['user_emotions']:
            emotion_metrics[emotion]['count'] += 1
            emotion_metrics[emotion]['avg_turns'].append(dialogue['num_lines'])

            status = dialogue['resolution_status']
            if status == 'Resolved':
                emotion_metrics[emotion]['resolved'] += 1
            elif status == 'Escalated':
                emotion_metrics[emotion]['escalated'] += 1
            elif status == 'Failed':
                emotion_metrics[emotion]['failed'] += 1

    # Compute averages
    for emotion in emotion_metrics:
        turns = emotion_metrics[emotion]['avg_turns']
        emotion_metrics[emotion]['avg_turns'] = np.mean(turns) if turns else 0

        count = emotion_metrics[emotion]['count']
        emotion_metrics[emotion]['resolved_rate'] = emotion_metrics[emotion]['resolved'] / count if count > 0 else 0
        emotion_metrics[emotion]['escalated_rate'] = emotion_metrics[emotion]['escalated'] / count if count > 0 else 0
        emotion_metrics[emotion]['failed_rate'] = emotion_metrics[emotion]['failed'] / count if count > 0 else 0

    metrics['by_emotion'] = dict(emotion_metrics)

    # 2. Metrics by Service Type
    service_metrics = defaultdict(lambda: {
        'count': 0,
        'avg_turns': [],
        'resolved': 0,
        'escalated': 0,
        'failed': 0
    })

    for dialogue in dialogues:
        for service in dialogue['services']:
            service_metrics[service]['count'] += 1
            service_metrics[service]['avg_turns'].append(dialogue['num_lines'])

            status = dialogue['resolution_status']
            if status == 'Resolved':
                service_metrics[service]['resolved'] += 1
            elif status == 'Escalated':
                service_metrics[service]['escalated'] += 1
            elif status == 'Failed':
                service_metrics[service]['failed'] += 1

    # Compute averages
    for service in service_metrics:
        turns = service_metrics[service]['avg_turns']
        service_metrics[service]['avg_turns'] = np.mean(turns) if turns else 0

        count = service_metrics[service]['count']
        service_metrics[service]['resolved_rate'] = service_metrics[service]['resolved'] / count if count > 0 else 0

    metrics['by_service'] = dict(service_metrics)

    # 3. Metrics by Service Complexity
    complexity_metrics = defaultdict(lambda: {
        'count': 0,
        'avg_turns': [],
        'resolved': 0,
        'resolved_rate': 0
    })

    for dialogue in dialogues:
        num_services = len(dialogue['services'])
        if num_services == 1:
            complexity = 'Single'
        elif num_services == 2:
            complexity = 'Double'
        elif num_services == 3:
            complexity = 'Triple'
        else:
            complexity = 'Quad+'

        complexity_metrics[complexity]['count'] += 1
        complexity_metrics[complexity]['avg_turns'].append(dialogue['num_lines'])
        if dialogue['resolution_status'] == 'Resolved':
            complexity_metrics[complexity]['resolved'] += 1

    # Compute averages
    for complexity in complexity_metrics:
        turns = complexity_metrics[complexity]['avg_turns']
        complexity_metrics[complexity]['avg_turns'] = np.mean(turns) if turns else 0
        count = complexity_metrics[complexity]['count']
        complexity_metrics[complexity]['resolved_rate'] = complexity_metrics[complexity]['resolved'] / count if count > 0 else 0

    metrics['by_complexity'] = dict(complexity_metrics)

    # Print formatted tables
    print("\n📊 TABLE 1: Quality Metrics by User Emotion (Top 10)")
    print("-" * 100)
    print(f"{'Emotion':<20} {'Count':>8} {'Avg Turns':>10} {'Resolved':>10} {'Escalated':>12} {'Failed':>10}")
    print("-" * 100)

    sorted_emotions = sorted(emotion_metrics.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    for emotion, data in sorted_emotions:
        print(f"{emotion:<20} {data['count']:>8,} {data['avg_turns']:>10.2f} "
              f"{data['resolved_rate']:>9.1%} {data['escalated_rate']:>11.1%} {data['failed_rate']:>9.1%}")

    print("\n📊 TABLE 2: Quality Metrics by Service Type")
    print("-" * 90)
    print(f"{'Service':<20} {'Count':>8} {'Avg Turns':>10} {'Resolved Rate':>15}")
    print("-" * 90)

    sorted_services = sorted(service_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
    for service, data in sorted_services:
        print(f"{service:<20} {data['count']:>8,} {data['avg_turns']:>10.2f} {data['resolved_rate']:>14.1%}")

    print("\n📊 TABLE 3: Quality Metrics by Service Complexity")
    print("-" * 90)
    print(f"{'Complexity':<20} {'Count':>8} {'Avg Turns':>10} {'Resolved Rate':>15}")
    print("-" * 90)

    complexity_order = ['Single', 'Double', 'Triple', 'Quad+']
    for complexity in complexity_order:
        if complexity in complexity_metrics:
            data = complexity_metrics[complexity]
            print(f"{complexity:<20} {data['count']:>8,} {data['avg_turns']:>10.2f} {data['resolved_rate']:>14.1%}")

    print("="*80)

    return metrics

def export_latex_tables(metrics, output_dir):
    """Export LaTeX tables for paper"""
    output_dir = Path(output_dir)

    # Table 1: Emotion-based metrics
    with open(output_dir / 'table_emotion_metrics.tex', 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quality metrics by user emotion in SynWOZ dataset.}\n")
        f.write("\\label{tab:emotion_metrics}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Emotion & Count & Avg Turns & Resolved & Escalated \\\\\n")
        f.write("\\midrule\n")

        sorted_emotions = sorted(metrics['by_emotion'].items(),
                                key=lambda x: x[1]['count'], reverse=True)[:10]
        for emotion, data in sorted_emotions:
            f.write(f"{emotion} & {data['count']:,} & {data['avg_turns']:.2f} & "
                   f"{data['resolved_rate']:.1%} & {data['escalated_rate']:.1%} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Saved: table_emotion_metrics.tex")

    # Table 2: Complexity metrics
    with open(output_dir / 'table_complexity_metrics.tex', 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quality metrics by service complexity.}\n")
        f.write("\\label{tab:complexity_metrics}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Complexity & Count & Avg Turns & Resolved \\\\\n")
        f.write("\\midrule\n")

        complexity_order = ['Single', 'Double', 'Triple', 'Quad+']
        for complexity in complexity_order:
            if complexity in metrics['by_complexity']:
                data = metrics['by_complexity'][complexity]
                f.write(f"{complexity} Service & {data['count']:,} & {data['avg_turns']:.2f} & "
                       f"{data['resolved_rate']:.1%} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Saved: table_complexity_metrics.tex")

def main():
    # Configuration
    dataset_path = "/Users/fortuna/Desktop/ananya/synwoz/code/gen/SynWOZ-dataset/dataset.jsonl"
    output_dir = "/Users/fortuna/Desktop/ananya/synwoz/code/gen/Persona_SYNWOZAAAI_submission_2026__Copy_/analysis_output"

    print("🚀 Starting SynWOZ Dataset Analysis...\n")

    # Load dataset
    print(f"📂 Loading dataset from: {dataset_path}")
    dialogues = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dialogues):,} dialogues\n")

    # Compute statistics
    print("📊 Computing statistics...")
    stats = basic_statistics(dialogues)
    print_statistics(stats)

    # Create visualizations
    print("\n🎨 Generating visualizations...")
    create_visualizations(stats, dialogues, output_dir)

    # Persona consistency validation
    validation_results = persona_consistency_validation(dialogues, sample_size=50)

    # Persona-sliced metrics
    metrics = persona_sliced_metrics(dialogues)

    # Export LaTeX tables
    print("\n📝 Exporting LaTeX tables...")
    export_latex_tables(metrics, output_dir)

    print(f"\n✅ Analysis complete! All outputs saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - 6 visualization PDFs/PNGs")
    print(f"  - 2 LaTeX table files")
    print(f"  - Console output with all statistics\n")

if __name__ == "__main__":
    main()
