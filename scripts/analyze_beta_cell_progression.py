"""
Beta-Cell Progression Analysis: HOMA-IR vs HOMA-B

This script analyzes the relationship between insulin resistance (HOMA-IR)
and beta-cell function (HOMA-B) to map the progression pathway to Type 2 Diabetes.

Progression Stages:
  Stage 1: Healthy (normal HOMA-IR, normal HOMA-B)
  Stage 2: Insulin Resistance (high HOMA-IR, normal HOMA-B)
  Stage 3: Compensatory Phase (high HOMA-IR, high HOMA-B) - pancreas overworking
  Stage 4: Beta-Cell Exhaustion (high HOMA-IR, low HOMA-B) - diabetes develops!

Purpose: Show the complete metabolic cascade from lifestyle to diabetes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def load_data():
    """Load the feature-engineered dataset with HOMA-IR and HOMA-B."""
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "processed" / "feature_engineered.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file, low_memory=False)
    print(f"[OK] Loaded {len(df):,} participants")

    return df


def categorize_stages(homa_ir, homa_b):
    """
    Categorize diabetes progression stages based on HOMA-IR and HOMA-B.

    Uses more flexible thresholds based on actual data distribution:
    - HOMA-IR threshold: 2.5 (standard clinical cutoff)
    - HOMA-B thresholds: Tertiles from data (low/normal/high)

    Stage 1: Healthy - Low IR, Normal Beta-Cell
    Stage 2: Insulin Resistance - High IR, Normal Beta-Cell
    Stage 3: Compensatory - High IR, High Beta-Cell (overworking pancreas)
    Stage 4: Beta-Cell Exhaustion - High IR, Low Beta-Cell (failure)
    """
    stages = []

    # Calculate HOMA-B tertiles for more realistic categorization
    homa_b_series = pd.Series(homa_b).dropna()
    homa_b_low = homa_b_series.quantile(0.33)  # Bottom 33%
    homa_b_high = homa_b_series.quantile(0.67)  # Top 33%

    print(f"\n[INFO] HOMA-B thresholds:")
    print(f"  Low function: < {homa_b_low:.1f}%")
    print(f"  Normal function: {homa_b_low:.1f}% - {homa_b_high:.1f}%")
    print(f"  High function (compensatory): > {homa_b_high:.1f}%")

    for ir, b in zip(homa_ir, homa_b):
        if pd.isna(ir) or pd.isna(b):
            stages.append(np.nan)
        # Stage 1: Healthy (normal IR, normal beta-cell)
        elif ir < 2.5 and homa_b_low <= b <= homa_b_high:
            stages.append("Stage 1: Healthy")
        # Stage 2: Insulin Resistance (high IR, normal beta-cell)
        elif ir >= 2.5 and homa_b_low <= b <= homa_b_high:
            stages.append("Stage 2: Insulin Resistance")
        # Stage 3: Compensatory (high IR, high beta-cell - pancreas overworking)
        elif ir >= 2.5 and b > homa_b_high:
            stages.append("Stage 3: Compensatory")
        # Stage 4: Beta-Cell Exhaustion (high IR, low beta-cell - pancreas failing)
        elif ir >= 2.5 and b < homa_b_low:
            stages.append("Stage 4: Beta-Cell Exhaustion")
        # Edge cases
        elif ir < 2.5 and b > homa_b_high:
            stages.append("Early Compensation")
        elif ir < 2.5 and b < homa_b_low:
            stages.append("Low Function")
        else:
            stages.append("Other")

    return pd.Series(stages)


def main():
    """Analyze the relationship between HOMA-IR and HOMA-B."""

    print("\n" + "=" * 70)
    print("BETA-CELL PROGRESSION ANALYSIS")
    print("=" * 70)
    print("\nMapping the pathway from lifestyle to Type 2 Diabetes")

    # Load data
    df = load_data()

    # Filter to cases with both HOMA-IR and HOMA-B
    required_cols = ['HOMA_IR', 'HOMA_B', 'comprehensive_inactivity_score',
                     'DR1TSUGR', 'DR1TFIBE', 'LBXHSCRP', 'BMXBMI', 'LBXGLU',
                     'RIDAGEYR', 'RIAGENDR']

    df_complete = df[required_cols].dropna()
    print(f"\nComplete cases (HOMA-IR and HOMA-B): {len(df_complete):,}")

    # Add stage categorization
    # First calculate thresholds from ALL cases (not just complete)
    homa_b_values = df_complete['HOMA_B'].dropna()
    homa_b_low = homa_b_values.quantile(0.33)
    homa_b_high = homa_b_values.quantile(0.67)

    print(f"\n[INFO] HOMA-B thresholds from data:")
    print(f"  Low function: < {homa_b_low:.1f}%")
    print(f"  Normal function: {homa_b_low:.1f}% - {homa_b_high:.1f}%")
    print(f"  High function (compensatory): > {homa_b_high:.1f}%")

    # Categorize each case
    stages = []
    for idx, row in df_complete.iterrows():
        ir = row['HOMA_IR']
        b = row['HOMA_B']

        if pd.isna(ir) or pd.isna(b):
            stages.append(np.nan)
        elif ir < 2.5 and homa_b_low <= b <= homa_b_high:
            stages.append("Stage 1: Healthy")
        elif ir >= 2.5 and homa_b_low <= b <= homa_b_high:
            stages.append("Stage 2: Insulin Resistance")
        elif ir >= 2.5 and b > homa_b_high:
            stages.append("Stage 3: Compensatory")
        elif ir >= 2.5 and b < homa_b_low:
            stages.append("Stage 4: Beta-Cell Exhaustion")
        else:
            stages.append("Other")

    df_complete['progression_stage'] = stages

    # =========================================================================
    # 1. OVERALL DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIABETES PROGRESSION STAGES")
    print("=" * 70)

    stage_counts = df_complete['progression_stage'].value_counts()
    print("\n" + stage_counts.to_string())

    total = len(df_complete)
    print(f"\nPercentages:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count/total*100:.1f}%")

    # =========================================================================
    # 2. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CORRELATION: HOMA-IR vs HOMA-B")
    print("=" * 70)

    r_pearson, p_pearson = pearsonr(df_complete['HOMA_IR'], df_complete['HOMA_B'])
    r_spearman, p_spearman = spearmanr(df_complete['HOMA_IR'], df_complete['HOMA_B'])

    print(f"\nPearson correlation:  r = {r_pearson:+.3f}, p = {p_pearson:.4e}")
    print(f"Spearman correlation: r = {r_spearman:+.3f}, p = {p_spearman:.4e}")

    if r_pearson > 0:
        print("\n[FINDING] POSITIVE correlation: As insulin resistance increases,")
        print("          beta-cells COMPENSATE by producing more insulin")
        print("          (This is the compensatory phase!)")

    # =========================================================================
    # 3. STAGE-SPECIFIC ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE-SPECIFIC CHARACTERISTICS")
    print("=" * 70)

    for stage in ["Stage 1: Healthy", "Stage 2: Insulin Resistance",
                  "Stage 3: Compensatory", "Stage 4: Beta-Cell Exhaustion"]:

        stage_data = df_complete[df_complete['progression_stage'] == stage]

        if len(stage_data) == 0:
            print(f"\n{stage}: No cases")
            continue

        print(f"\n{stage}: N = {len(stage_data)}")
        print(f"  HOMA-IR:       {stage_data['HOMA_IR'].mean():.2f} +/- {stage_data['HOMA_IR'].std():.2f}")
        print(f"  HOMA-B:        {stage_data['HOMA_B'].mean():.1f}% +/- {stage_data['HOMA_B'].std():.1f}%")
        print(f"  Glucose:       {stage_data['LBXGLU'].mean():.1f} mg/dL")
        print(f"  BMI:           {stage_data['BMXBMI'].mean():.1f}")
        print(f"  Inactivity:    {stage_data['comprehensive_inactivity_score'].mean():.1f}")
        print(f"  Sugar:         {stage_data['DR1TSUGR'].mean():.1f} g/day")
        print(f"  CRP:           {stage_data['LBXHSCRP'].mean():.2f} mg/L")

    # =========================================================================
    # 4. LIFESTYLE FACTORS BY STAGE
    # =========================================================================
    print("\n" + "=" * 70)
    print("LIFESTYLE FACTORS ACROSS PROGRESSION STAGES")
    print("=" * 70)

    lifestyle_by_stage = df_complete.groupby('progression_stage').agg({
        'comprehensive_inactivity_score': 'mean',
        'DR1TSUGR': 'mean',
        'DR1TFIBE': 'mean',
        'LBXHSCRP': 'mean',
        'BMXBMI': 'mean'
    }).round(2)

    print("\n" + lifestyle_by_stage.to_string())

    # =========================================================================
    # 5. CRITICAL THRESHOLDS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CRITICAL THRESHOLDS FOR BETA-CELL EXHAUSTION")
    print("=" * 70)

    exhaustion_cases = df_complete[df_complete['progression_stage'] == 'Stage 4: Beta-Cell Exhaustion']

    if len(exhaustion_cases) > 0:
        print(f"\nBeta-Cell Exhaustion Cases: {len(exhaustion_cases)}")
        print(f"  Mean HOMA-IR:  {exhaustion_cases['HOMA_IR'].mean():.2f}")
        print(f"  Mean HOMA-B:   {exhaustion_cases['HOMA_B'].mean():.1f}%")
        print(f"  Mean Glucose:  {exhaustion_cases['LBXGLU'].mean():.1f} mg/dL")
        print(f"  Mean BMI:      {exhaustion_cases['BMXBMI'].mean():.1f}")

        print(f"\nLifestyle Pattern (Exhaustion):")
        print(f"  Inactivity Score: {exhaustion_cases['comprehensive_inactivity_score'].mean():.1f}")
        print(f"  Sugar Intake:     {exhaustion_cases['DR1TSUGR'].mean():.1f} g/day")
        print(f"  Fiber Intake:     {exhaustion_cases['DR1TFIBE'].mean():.1f} g/day")
        print(f"  CRP Level:        {exhaustion_cases['LBXHSCRP'].mean():.2f} mg/L")
    else:
        print("\n[NOTE] No Stage 4 cases in dataset")
        print("  (Beta-cell exhaustion is rare in pediatric population)")

    # =========================================================================
    # 6. VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "models" / "final" / "beta_cell_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Scatter plot HOMA-IR vs HOMA-B with stages
    fig, ax = plt.subplots(figsize=(10, 8))

    stage_colors = {
        "Stage 1: Healthy": "green",
        "Stage 2: Insulin Resistance": "yellow",
        "Stage 3: Compensatory": "orange",
        "Stage 4: Beta-Cell Exhaustion": "red",
        "Early Compensation": "cyan",
        "Low Function (Rare)": "purple",
        "Other": "gray"
    }

    for stage in df_complete['progression_stage'].unique():
        if pd.notna(stage):
            stage_data = df_complete[df_complete['progression_stage'] == stage]
            ax.scatter(
                stage_data['HOMA_IR'],
                stage_data['HOMA_B'],
                c=stage_colors.get(stage, 'gray'),
                label=stage,
                alpha=0.6,
                s=50
            )

    # Add threshold lines
    ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='HOMA-IR threshold (2.5)')
    ax.axhline(y=50, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='HOMA-B low (50%)')
    ax.axhline(y=150, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='HOMA-B high (150%)')

    ax.set_xlabel('HOMA-IR (Insulin Resistance)', fontsize=12)
    ax.set_ylabel('HOMA-B (Beta-Cell Function %)', fontsize=12)
    ax.set_title('Diabetes Progression: Insulin Resistance vs Beta-Cell Function', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_file = output_dir / "homa_ir_vs_homa_b_scatter.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {plot1_file.name}")

    # Plot 2: Box plots by stage
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    variables = [
        ('HOMA_IR', 'HOMA-IR'),
        ('HOMA_B', 'HOMA-B (%)'),
        ('BMXBMI', 'BMI'),
        ('comprehensive_inactivity_score', 'Inactivity Score'),
        ('DR1TSUGR', 'Sugar Intake (g/day)'),
        ('LBXHSCRP', 'CRP (mg/L)')
    ]

    for idx, (var, label) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]

        # Filter stages with data
        stage_order = ["Stage 1: Healthy", "Stage 2: Insulin Resistance",
                       "Stage 3: Compensatory", "Stage 4: Beta-Cell Exhaustion"]
        plot_data = df_complete[df_complete['progression_stage'].isin(stage_order)]

        if len(plot_data) > 0:
            sns.boxplot(
                data=plot_data,
                x='progression_stage',
                y=var,
                ax=ax,
                palette=['green', 'yellow', 'orange', 'red']
            )
            ax.set_xlabel('')
            ax.set_ylabel(label, fontsize=10)
            ax.set_xticklabels(['Healthy', 'IR', 'Comp', 'Exhaust'], rotation=0, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Metabolic and Lifestyle Markers Across Diabetes Progression Stages',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot2_file = output_dir / "progression_stages_boxplots.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {plot2_file.name}")

    # Plot 3: Progression pathway diagram
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate mean values for each stage
    stages_in_order = ["Stage 1: Healthy", "Stage 2: Insulin Resistance",
                       "Stage 3: Compensatory", "Stage 4: Beta-Cell Exhaustion"]

    stage_stats = []
    for stage in stages_in_order:
        stage_data = df_complete[df_complete['progression_stage'] == stage]
        if len(stage_data) > 0:
            stage_stats.append({
                'stage': stage,
                'n': len(stage_data),
                'homa_ir': stage_data['HOMA_IR'].mean(),
                'homa_b': stage_data['HOMA_B'].mean()
            })

    if len(stage_stats) > 0:
        x_positions = range(len(stage_stats))
        homa_ir_values = [s['homa_ir'] for s in stage_stats]
        homa_b_values = [s['homa_b'] for s in stage_stats]

        ax2 = ax.twinx()

        line1 = ax.plot(x_positions, homa_ir_values, 'r-o', linewidth=3, markersize=10, label='HOMA-IR (Insulin Resistance)')
        line2 = ax2.plot(x_positions, homa_b_values, 'b-s', linewidth=3, markersize=10, label='HOMA-B (Beta-Cell Function)')

        ax.set_xlabel('Diabetes Progression Stage', fontsize=12)
        ax.set_ylabel('HOMA-IR (Insulin Resistance)', fontsize=12, color='r')
        ax2.set_ylabel('HOMA-B (Beta-Cell Function %)', fontsize=12, color='b')

        ax.set_xticks(x_positions)
        ax.set_xticklabels([s['stage'].replace('Stage ', 'S') for s in stage_stats], rotation=30, ha='right')

        ax.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='b')

        # Add count labels
        for i, s in enumerate(stage_stats):
            ax.text(i, homa_ir_values[i] + 0.5, f"n={s['n']}", ha='center', fontsize=9)

        ax.set_title('The Diabetes Progression Pathway\n(From Healthy to Beta-Cell Exhaustion)',
                     fontsize=14, fontweight='bold')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=10)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot3_file = output_dir / "progression_pathway_diagram.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {plot3_file.name}")

    # =========================================================================
    # 7. SAVE RESULTS TO CSV
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 70)

    # Save stage-categorized data
    output_file = output_dir / "progression_stages_data.csv"
    df_complete.to_csv(output_file, index=False)

    print(f"\n[OK] Saved progression data: {output_file.name}")
    print(f"  Rows: {len(df_complete):,}")
    print(f"  Columns: {df_complete.shape[1]}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("\n1. Diabetes Progression Pathway Identified:")
    for stage in stages_in_order:
        count = (df_complete['progression_stage'] == stage).sum()
        pct = count / len(df_complete) * 100
        print(f"   {stage}: {count} cases ({pct:.1f}%)")

    print(f"\n2. HOMA-IR and HOMA-B Correlation:")
    print(f"   Correlation: r = {r_pearson:+.3f}")
    if r_pearson > 0.3:
        print("   -> POSITIVE: Beta-cells COMPENSATE for insulin resistance")

    print(f"\n3. Beta-Cell Exhaustion (Stage 4):")
    exhaustion_count = (df_complete['progression_stage'] == 'Stage 4: Beta-Cell Exhaustion').sum()
    if exhaustion_count > 0:
        print(f"   {exhaustion_count} teenagers ({exhaustion_count/len(df_complete)*100:.1f}%) show beta-cell exhaustion")
        print("   -> This is the FINAL STAGE before Type 2 Diabetes!")
    else:
        print("   No cases in pediatric dataset (rare but critical to prevent)")

    compensatory_count = (df_complete['progression_stage'] == 'Stage 3: Compensatory').sum()
    print(f"\n4. Compensatory Phase (Stage 3):")
    print(f"   {compensatory_count} teenagers ({compensatory_count/len(df_complete)*100:.1f}%) in compensatory phase")
    print("   -> Pancreas is OVERWORKING - risk of eventual exhaustion!")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nVisualizations saved to: {output_dir}")
    print("Use these findings to:")
    print("  1. Show the complete progression pathway")
    print("  2. Demonstrate both HOMA-IR AND HOMA-B matter")
    print("  3. Identify at-risk teens in compensatory phase")


if __name__ == "__main__":
    main()
