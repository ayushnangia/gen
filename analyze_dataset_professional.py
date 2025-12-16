#!/usr/bin/env python3
"""
Professional publication-quality analysis script for SynWOZ dataset
Generates uniform, high-quality visualizations for PersonaLLM@NeurIPS 2025 submission
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import random

# ============================================================================
# PROFESSIONAL STYLING CONFIGURATION
# ============================================================================

# Color palette (colorblind-safe, professional)
COLORS = {
    'primary': '#2E5C8A',      # Deep blue
    'secondary': '#E67E22',    # Warm orange
    'tertiary': '#27AE60',     # Forest green
    'quaternary': '#8E44AD',   # Purple
    'accent1': '#C0392B',      # Deep red
    'accent2': '#F39C12',      # Gold
    'neutral_dark': '#34495E', # Dark gray
    'neutral': '#7F8C8D',      # Medium gray
    'neutral_light': '#BDC3C7' # Light gray
}

# Colorblind-safe palettes
PALETTE_CATEGORICAL = ['#2E5C8A', '#E67E22', '#27AE60', '#8E44AD',
                       '#C0392B', '#F39C12', '#16A085', '#D35400']
PALETTE_SEQUENTIAL_BLUE = ['#EFF3FF', '#C6DBEF', '#9ECAE1', '#6BAED6',
                           '#4292C6', '#2171B5', '#084594']
PALETTE_SEQUENTIAL_GREEN = ['#F7FCF5', '#E5F5E0', '#C7E9C0', '#A1D99B',
                            '#74C476', '#41AB5D', '#238B45', '#005A32']

# Typography
FONT_FAMILY = 'DejaVu Sans'
FONT_SIZES = {
    'title': 13,
    'label': 11,
    'tick': 9,
    'legend': 9,
    'annotation': 9
}

# Layout
DPI = 300
FIGURE_SIZES = {
    'single_column': (4.5, 3.5),   # inches (for single column)
    'double_column': (7, 4.5),      # inches (for double column)
    'tall': (4.5, 5.5),            # inches (for tall figures)
    'wide': (7, 3.5)               # inches (for wide figures)
}

# Configure matplotlib globally
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = FONT_SIZES['tick']
plt.rcParams['axes.labelsize'] = FONT_SIZES['label']
plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']
plt.rcParams['legend.fontsize'] = FONT_SIZES['legend']
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.edgecolor'] = COLORS['neutral']
plt.rcParams['axes.linewidth'] = 0.8

sns.set_palette(PALETTE_CATEGORICAL)

# ============================================================================
# DATA LOADING & ANALYSIS (from original script)
# ============================================================================

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

    # Resolution status
    resolution_statuses = [d['resolution_status'] for d in dialogues]
    stats['resolution_status_counts'] = Counter(resolution_statuses)

    return stats

# ============================================================================
# PROFESSIONAL VISUALIZATIONS
# ============================================================================

def create_emotion_distribution(stats, output_dir):
    """Create professional emotion distribution figure"""
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=FIGURE_SIZES['double_column'])
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # User emotions (left panel)
    ax1 = fig.add_subplot(gs[0, 0])
    user_emotions = sorted(stats['user_emotion_counts'].items(),
                          key=lambda x: x[1], reverse=True)[:20]
    labels1, values1 = zip(*user_emotions)

    # Create horizontal bar chart
    y_pos = np.arange(len(labels1))
    bars1 = ax1.barh(y_pos, values1, color=COLORS['primary'],
                     edgecolor='white', linewidth=0.5, alpha=0.85)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels1, fontsize=FONT_SIZES['tick'])
    ax1.set_xlabel('Count', fontsize=FONT_SIZES['label'], fontweight='semibold')
    ax1.set_title('User Emotions', fontsize=FONT_SIZES['title'],
                 fontweight='bold', pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, values1)):
        ax1.text(val + 30, i, f'{val:,}',
                va='center', ha='left', fontsize=FONT_SIZES['annotation']-1)

    # Assistant emotions (right panel)
    ax2 = fig.add_subplot(gs[0, 1])
    asst_emotions = sorted(stats['assistant_emotion_counts'].items(),
                          key=lambda x: x[1], reverse=True)[:20]
    labels2, values2 = zip(*asst_emotions)

    y_pos2 = np.arange(len(labels2))
    bars2 = ax2.barh(y_pos2, values2, color=COLORS['tertiary'],
                     edgecolor='white', linewidth=0.5, alpha=0.85)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels2, fontsize=FONT_SIZES['tick'])
    ax2.set_xlabel('Count', fontsize=FONT_SIZES['label'], fontweight='semibold')
    ax2.set_title('Assistant Styles', fontsize=FONT_SIZES['title'],
                 fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, values2)):
        ax2.text(val + 30, i, f'{val:,}',
                va='center', ha='left', fontsize=FONT_SIZES['annotation']-1)

    plt.savefig(output_dir / 'emotion_distribution_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'emotion_distribution_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: emotion_distribution_professional.pdf/png")

def create_service_complexity(stats, output_dir):
    """Create professional service complexity pie chart"""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    labels = ['Single', 'Double', 'Triple', 'Quad+']
    sizes = [stats['single_service'], stats['double_service'],
             stats['triple_service'], stats['quad_plus_service']]
    colors = [COLORS['primary'], COLORS['secondary'],
             COLORS['tertiary'], COLORS['quaternary']]
    explode = (0.02, 0.02, 0.02, 0.02)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%',
                                        colors=colors, explode=explode,
                                        startangle=90,
                                        textprops={'fontsize': FONT_SIZES['annotation'],
                                                  'fontweight': 'semibold'},
                                        pctdistance=0.85)

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(FONT_SIZES['annotation'])
        autotext.set_fontweight('bold')

    ax.set_title('Service Combination Complexity',
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)

    # Create custom legend
    legend_labels = [f'{label}: {size:,} ({100*size/sum(sizes):.1f}%)'
                    for label, size in zip(labels, sizes)]
    legend_patches = [mpatches.Patch(color=color, label=label)
                     for color, label in zip(colors, legend_labels)]

    ax.legend(handles=legend_patches, loc='center left',
             bbox_to_anchor=(1, 0.5), frameon=True,
             fancybox=False, shadow=False,
             fontsize=FONT_SIZES['legend'])

    plt.savefig(output_dir / 'service_complexity_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'service_complexity_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: service_complexity_professional.pdf/png")

def create_geographic_distribution(stats, output_dir):
    """Create professional geographic distribution"""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['tall'])

    top_regions = stats['region_counts'].most_common(20)
    regions, region_counts = zip(*top_regions)

    # Create color gradient
    colors_geo = plt.cm.viridis(np.linspace(0.3, 0.9, len(regions)))

    y_pos = np.arange(len(regions))
    bars = ax.barh(y_pos, region_counts, color=colors_geo,
                   edgecolor='white', linewidth=0.5, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=FONT_SIZES['tick'])
    ax.set_xlabel('Number of Dialogues', fontsize=FONT_SIZES['label'],
                 fontweight='semibold')
    ax.set_title('Geographic Distribution (Top 20 Cities)',
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, region_counts)):
        ax.text(count + 5, i, f'{count:,}',
               va='center', ha='left', fontsize=FONT_SIZES['annotation']-1)

    plt.savefig(output_dir / 'geographic_distribution_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'geographic_distribution_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: geographic_distribution_professional.pdf/png")

def create_turn_distribution(stats, dialogues, output_dir):
    """Create professional turn distribution histogram"""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    turn_counts = [d['num_lines'] for d in dialogues]

    # Create histogram
    n, bins, patches = ax.hist(turn_counts, bins=range(min(turn_counts), max(turn_counts) + 2),
                               color=COLORS['primary'], edgecolor='white',
                               linewidth=0.5, alpha=0.85)

    # Add mean and median lines
    mean_line = ax.axvline(stats['mean_turns'], color=COLORS['accent1'],
                          linestyle='--', linewidth=2.5, alpha=0.8,
                          label=f'Mean: {stats["mean_turns"]:.2f}')
    median_line = ax.axvline(stats['median_turns'], color=COLORS['tertiary'],
                            linestyle='--', linewidth=2.5, alpha=0.8,
                            label=f'Median: {stats["median_turns"]:.0f}')

    ax.set_xlabel('Number of Turns per Dialogue', fontsize=FONT_SIZES['label'],
                 fontweight='semibold')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'], fontweight='semibold')
    ax.set_title('Turn Count Distribution', fontsize=FONT_SIZES['title'],
                fontweight='bold', pad=10)
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=True, fancybox=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.savefig(output_dir / 'turn_distribution_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'turn_distribution_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: turn_distribution_professional.pdf/png")

def create_resolution_status(stats, output_dir):
    """Create professional resolution status chart"""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    statuses = list(stats['resolution_status_counts'].keys())
    counts = list(stats['resolution_status_counts'].values())

    # Color mapping
    status_colors = {
        'Resolved': COLORS['tertiary'],
        'Escalated': COLORS['secondary'],
        'Failed': COLORS['accent1']
    }
    colors_res = [status_colors.get(s, COLORS['neutral']) for s in statuses]

    bars = ax.bar(statuses, counts, color=colors_res,
                  edgecolor='white', linewidth=1, alpha=0.85)

    ax.set_ylabel('Number of Dialogues', fontsize=FONT_SIZES['label'],
                 fontweight='semibold')
    ax.set_title('Resolution Status Distribution', fontsize=FONT_SIZES['title'],
                fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / stats['total_dialogues']
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=FONT_SIZES['annotation'],
                fontweight='semibold')

    plt.savefig(output_dir / 'resolution_status_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'resolution_status_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: resolution_status_professional.pdf/png")

def create_service_distribution(stats, output_dir):
    """Create professional service distribution chart"""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    services = [s for s, _ in stats['service_counts'].most_common()]
    counts = [c for _, c in stats['service_counts'].most_common()]

    # Use categorical palette
    colors = PALETTE_CATEGORICAL[:len(services)]

    y_pos = np.arange(len(services))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white',
                   linewidth=0.5, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(services, fontsize=FONT_SIZES['tick'])
    ax.set_xlabel('Number of Dialogues', fontsize=FONT_SIZES['label'],
                 fontweight='semibold')
    ax.set_title('Service Type Distribution', fontsize=FONT_SIZES['title'],
                fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 150, i, f'{count:,}',
               va='center', ha='left', fontsize=FONT_SIZES['annotation'])

    plt.savefig(output_dir / 'service_distribution_professional.pdf',
               bbox_inches='tight', dpi=DPI)
    plt.savefig(output_dir / 'service_distribution_professional.png',
               bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"✓ Saved: service_distribution_professional.pdf/png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    dataset_path = "/Users/fortuna/Desktop/ananya/synwoz/code/gen/SynWOZ-dataset/dataset.jsonl"
    output_dir = "/Users/fortuna/Desktop/ananya/synwoz/code/gen/Persona_SYNWOZAAAI_submission_2026__Copy_/figures_professional"

    print("="*80)
    print("PROFESSIONAL FIGURE GENERATION FOR SYNWOZ DATASET")
    print("="*80)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load dataset
    print(f"\n📂 Loading dataset from: {dataset_path}")
    dialogues = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dialogues):,} dialogues\n")

    # Compute statistics
    print("📊 Computing statistics...")
    stats = basic_statistics(dialogues)
    print("✓ Statistics computed\n")

    # Create professional visualizations
    print("🎨 Generating professional visualizations...\n")

    create_emotion_distribution(stats, output_dir)
    create_service_complexity(stats, output_dir)
    create_geographic_distribution(stats, output_dir)
    create_turn_distribution(stats, dialogues, output_dir)
    create_resolution_status(stats, output_dir)
    create_service_distribution(stats, output_dir)

    print(f"\n✅ All professional figures saved to: {output_dir}")
    print("\nGenerated 6 publication-quality visualizations:")
    print("  - emotion_distribution_professional.pdf/png")
    print("  - service_complexity_professional.pdf/png")
    print("  - geographic_distribution_professional.pdf/png")
    print("  - turn_distribution_professional.pdf/png")
    print("  - resolution_status_professional.pdf/png")
    print("  - service_distribution_professional.pdf/png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
