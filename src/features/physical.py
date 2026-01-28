"""
Physical Activity Feature Engineering

Engineers physical activity features from NHANES Physical Activity
Questionnaire (PAQ) data.

Physical activity is a key protective factor against insulin resistance
and Type 2 Diabetes.
"""

import pandas as pd
import numpy as np


class PhysicalActivityFeatureCalculator:
    """Calculates physical activity features from PAQ data."""

    def __init__(self):
        """Initialize physical activity feature calculator."""
        # WHO/CDC guidelines for adolescents
        self.min_weekly_activity = 300  # minutes/week moderate-vigorous activity
        self.target_daily_activity = 60  # minutes/day
        self.sedentary_threshold = 480  # 8 hours/day

    def calculate_sedentary_score(
        self,
        sedentary_min: pd.Series
    ) -> pd.Series:
        """
        Calculate daily sedentary time score.

        Args:
            sedentary_min: Daily sedentary minutes (PAD680)

        Returns:
            Sedentary score (normalized to 0-10 scale, higher is worse)
        """
        if sedentary_min.isna().all():
            print("[WARN] All sedentary data missing")
            return pd.Series([np.nan] * len(sedentary_min), index=sedentary_min.index)

        # Normalize to 0-10 scale (cap at 16 hours = 960 minutes)
        sedentary_score = np.clip(sedentary_min / 96, 0, 10)

        valid_values = sedentary_score.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Sedentary score calculated: {len(valid_values)} values")
            print(f"  Mean sedentary time: {sedentary_min.mean():.0f} min/day")
            print(f"  Median sedentary time: {sedentary_min.median():.0f} min/day")

            excessive = (sedentary_min > self.sedentary_threshold).sum()
            print(f"  Excessive sedentary (>{self.sedentary_threshold/60:.0f} hrs): "
                  f"{excessive} ({excessive/len(valid_values)*100:.1f}%)")

        return sedentary_score

    def calculate_active_score(
        self,
        vigorous_days: pd.Series = None,
        vigorous_min: pd.Series = None,
        moderate_days: pd.Series = None,
        moderate_min: pd.Series = None
    ) -> pd.Series:
        """
        Calculate weekly active time score from vigorous and moderate activity.

        Args:
            vigorous_days: Days per week of vigorous activity (PAQ706)
            vigorous_min: Minutes of vigorous activity per day (PAD733)
            moderate_days: Days per week of moderate activity
            moderate_min: Minutes of moderate activity per day

        Returns:
            Active score (0-10 scale, higher is better)
        """
        total_active_min = pd.Series(0, index=vigorous_days.index if vigorous_days is not None else moderate_days.index)

        # Add vigorous activity (if available)
        if vigorous_days is not None and vigorous_min is not None:
            vigorous_weekly = vigorous_days * vigorous_min
            total_active_min = total_active_min + vigorous_weekly

        # Add moderate activity (if available)
        if moderate_days is not None and moderate_min is not None:
            moderate_weekly = moderate_days * moderate_min
            total_active_min = total_active_min + moderate_weekly

        # Normalize to 0-10 scale (cap at 600 minutes/week)
        active_score = np.clip(total_active_min / 60, 0, 10)

        valid_values = active_score.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Active score calculated: {len(valid_values)} values")
            print(f"  Mean active time: {total_active_min.mean():.0f} min/week")
            print(f"  Median active time: {total_active_min.median():.0f} min/week")

            meets_guidelines = (total_active_min >= self.min_weekly_activity).sum()
            print(f"  Meets guidelines (>={self.min_weekly_activity} min/week): "
                  f"{meets_guidelines} ({meets_guidelines/len(valid_values)*100:.1f}%)")

        return active_score

    def create_activity_ratio(
        self,
        active_score: pd.Series,
        sedentary_score: pd.Series
    ) -> pd.Series:
        """
        Calculate ratio of active to sedentary time.

        Higher ratio indicates better activity balance.

        Args:
            active_score: Active time score (0-10)
            sedentary_score: Sedentary time score (0-10)

        Returns:
            Activity ratio
        """
        # Inverse sedentary score (so lower sedentary = higher score)
        sedentary_inverse = 10 - sedentary_score

        # Ratio: active / (sedentary + 1) to avoid division by zero
        activity_ratio = active_score / (sedentary_score + 1)

        valid_values = activity_ratio.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Activity ratio calculated: {len(valid_values)} values")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

        return activity_ratio

    def categorize_activity_level(
        self,
        active_score: pd.Series
    ) -> pd.Series:
        """
        Categorize activity level into risk categories.

        Categories based on WHO/CDC guidelines:
        - 0: Inactive (<150 min/week, score <2.5)
        - 1: Moderate (150-300 min/week, score 2.5-5)
        - 2: Active (>300 min/week, score >5)

        Args:
            active_score: Active score (0-10 scale)

        Returns:
            Activity level categories
        """
        categories = pd.cut(
            active_score,
            bins=[-np.inf, 2.5, 5, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        )

        valid_categories = categories.dropna()
        if len(valid_categories) > 0:
            inactive = (valid_categories == 0).sum()
            moderate = (valid_categories == 1).sum()
            active = (valid_categories == 2).sum()

            print(f"[OK] Activity level categorized: {len(valid_categories)} values")
            print(f"  Inactive: {inactive} ({inactive/len(valid_categories)*100:.1f}%)")
            print(f"  Moderate: {moderate} ({moderate/len(valid_categories)*100:.1f}%)")
            print(f"  Active: {active} ({active/len(valid_categories)*100:.1f}%)")

        return categories.astype(float)

    def calculate_physical_activity_score(
        self,
        active_score: pd.Series,
        sedentary_score: pd.Series
    ) -> pd.Series:
        """
        Calculate composite physical activity score.

        Combines active time (positive) and sedentary time (negative).

        Args:
            active_score: Active score (0-10)
            sedentary_score: Sedentary score (0-10)

        Returns:
            Physical activity score (0-100 scale)
        """
        # Composite: active benefit - sedentary penalty
        # Weight active more heavily (0.7) than sedentary (0.3)
        pa_score = (active_score * 0.7 - sedentary_score * 0.3) * 10

        # Normalize to 0-100 scale
        pa_score = np.clip(pa_score, 0, 100)

        valid_values = pa_score.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Physical activity score calculated: {len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

        return pa_score

    def calculate_comprehensive_inactivity_score(
        self,
        vig_rec: pd.Series = None,       # PAQ650 - Vigorous recreational (1=Yes, 2=No)
        mod_rec: pd.Series = None,       # PAQ665 - Moderate recreational (1=Yes, 2=No)
        walk_bike: pd.Series = None,     # PAQ635 - Walk/bicycle (1=Yes, 2=No)
        sedentary_min: pd.Series = None, # PAD680 - Sedentary minutes
        tv_hours: pd.Series = None,      # PAQ710 - TV hours/day
        comp_hours: pd.Series = None     # PAQ715 - Computer hours/day
    ) -> pd.Series:
        """
        Calculate comprehensive inactivity score using high-coverage variables.

        Combines multiple activity indicators:
        - Recreational activities (vigorous + moderate)
        - Active transportation (walking/biking)
        - Sedentary behavior
        - Screen time

        Score interpretation:
        - 0-33: Active (low inactivity)
        - 34-66: Moderate inactivity
        - 67-100: Inactive (high inactivity risk)

        Args:
            vig_rec: Vigorous recreational activity (1=Yes, 2=No)
            mod_rec: Moderate recreational activity (1=Yes, 2=No)
            walk_bike: Walk or bicycle for transportation (1=Yes, 2=No)
            sedentary_min: Daily sedentary minutes
            tv_hours: Daily TV/video hours
            comp_hours: Daily computer hours

        Returns:
            Comprehensive inactivity score (0-100, higher = more inactive)
        """
        # Initialize score
        score = pd.Series(0.0, index=vig_rec.index if vig_rec is not None else pd.RangeIndex(0))

        components_used = []

        # Component 1: Lack of vigorous activity (0-25 points)
        if vig_rec is not None:
            # 1=Yes (active) → 0 points, 2=No (inactive) → 25 points
            no_vigorous = (vig_rec == 2).astype(float) * 25
            score = score + no_vigorous.fillna(0)
            components_used.append('vigorous_rec')

        # Component 2: Lack of moderate activity (0-20 points)
        if mod_rec is not None:
            no_moderate = (mod_rec == 2).astype(float) * 20
            score = score + no_moderate.fillna(0)
            components_used.append('moderate_rec')

        # Component 3: No active transportation (0-15 points)
        if walk_bike is not None:
            no_walk_bike = (walk_bike == 2).astype(float) * 15
            score = score + no_walk_bike.fillna(0)
            components_used.append('walk_bike')

        # Component 4: High sedentary time (0-20 points)
        if sedentary_min is not None:
            # >8 hours (480 min) = 20 points, 0 hours = 0 points
            sedentary_penalty = np.clip(sedentary_min / 480 * 20, 0, 20)
            score = score + sedentary_penalty.fillna(0)
            components_used.append('sedentary')

        # Component 5: High screen time (0-20 points)
        if tv_hours is not None and comp_hours is not None:
            total_screen = tv_hours.fillna(0) + comp_hours.fillna(0)
            # >6 hours = 20 points, 0 hours = 0 points
            screen_penalty = np.clip(total_screen / 6 * 20, 0, 20)
            score = score + screen_penalty
            components_used.append('screen_time')

        # Normalize to 0-100 scale
        max_possible = 100  # Sum of all components
        score = np.clip(score, 0, max_possible)

        # Report statistics
        valid_values = score.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Comprehensive inactivity score calculated: {len(valid_values)} values")
            print(f"  Components used: {', '.join(components_used)}")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

            # Risk distribution
            active = (valid_values < 34).sum()
            moderate = ((valid_values >= 34) & (valid_values < 67)).sum()
            inactive = (valid_values >= 67).sum()

            print(f"  Risk Distribution:")
            print(f"    Active (<34): {active} ({active/len(valid_values)*100:.1f}%)")
            print(f"    Moderate (34-66): {moderate} ({moderate/len(valid_values)*100:.1f}%)")
            print(f"    Inactive (>=67): {inactive} ({inactive/len(valid_values)*100:.1f}%)")

        return score


def calculate_all_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all physical activity features from NHANES dataset.

    Args:
        df: DataFrame with PAQ columns (PAD680, PAQ706, PAD733, etc.)

    Returns:
        DataFrame with added physical activity features
    """
    print("\n" + "=" * 70)
    print("PHYSICAL ACTIVITY FEATURE ENGINEERING")
    print("=" * 70)

    calculator = PhysicalActivityFeatureCalculator()

    print(f"\nDataset: {len(df):,} rows")

    # Check which activity data is available
    has_sedentary = 'PAD680' in df.columns
    has_vigorous_days = 'PAQ706' in df.columns
    has_vigorous_min = 'PAD733' in df.columns

    print(f"\nAvailable data:")
    if has_sedentary:
        print(f"  Sedentary time: {df['PAD680'].notna().sum():,} values")
    if has_vigorous_days:
        print(f"  Vigorous activity days: {df['PAQ706'].notna().sum():,} values")
    if has_vigorous_min:
        print(f"  Vigorous activity minutes: {df['PAD733'].notna().sum():,} values")

    # Calculate sedentary score
    if has_sedentary:
        df['sedentary_score'] = calculator.calculate_sedentary_score(df['PAD680'])
    else:
        print("[WARN] Sedentary data (PAD680) not found")
        df['sedentary_score'] = np.nan

    # Calculate active score
    if has_vigorous_days and has_vigorous_min:
        df['active_score'] = calculator.calculate_active_score(
            vigorous_days=df['PAQ706'],
            vigorous_min=df['PAD733']
        )
    else:
        print("[WARN] Vigorous activity data not found")
        df['active_score'] = np.nan

    # Calculate activity ratio
    if has_sedentary and has_vigorous_days:
        df['activity_ratio'] = calculator.create_activity_ratio(
            df['active_score'], df['sedentary_score']
        )
    else:
        print("[WARN] Cannot calculate activity ratio - missing data")
        df['activity_ratio'] = np.nan

    # Categorize activity level
    if has_vigorous_days:
        df['activity_category'] = calculator.categorize_activity_level(df['active_score'])
    else:
        df['activity_category'] = np.nan

    # Calculate composite physical activity score
    if has_sedentary and has_vigorous_days:
        df['physical_activity_score'] = calculator.calculate_physical_activity_score(
            df['active_score'], df['sedentary_score']
        )
    else:
        print("[WARN] Cannot calculate physical activity score - missing data")
        df['physical_activity_score'] = np.nan

    # Calculate comprehensive inactivity score (HIGH COVERAGE - uses yes/no questions)
    has_vig_rec = 'PAQ650' in df.columns
    has_mod_rec = 'PAQ665' in df.columns
    has_walk_bike = 'PAQ635' in df.columns
    has_tv = 'PAQ710' in df.columns
    has_comp = 'PAQ715' in df.columns

    if any([has_vig_rec, has_mod_rec, has_walk_bike, has_sedentary, has_tv]):
        print("\n" + "-" * 70)
        print("COMPREHENSIVE INACTIVITY SCORE (High Coverage)")
        print("-" * 70)
        df['comprehensive_inactivity_score'] = calculator.calculate_comprehensive_inactivity_score(
            vig_rec=df.get('PAQ650'),
            mod_rec=df.get('PAQ665'),
            walk_bike=df.get('PAQ635'),
            sedentary_min=df.get('PAD680'),
            tv_hours=df.get('PAQ710'),
            comp_hours=df.get('PAQ715')
        )
    else:
        print("[WARN] Cannot calculate comprehensive inactivity score - no data available")
        df['comprehensive_inactivity_score'] = np.nan

    print("\n" + "=" * 70)
    print("PHYSICAL ACTIVITY FEATURES COMPLETE")
    print("=" * 70)
    print(f"Features added: sedentary_score, active_score, activity_ratio, "
          f"activity_category, physical_activity_score, comprehensive_inactivity_score")
    print(f"Valid comprehensive inactivity scores: "
          f"{df['comprehensive_inactivity_score'].notna().sum():,}/{len(df):,}")

    return df


if __name__ == "__main__":
    # Test with multi-cycle pediatric data
    print("THE PEDIATRIC SENTINEL - Physical Activity Feature Engineering")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Load multi-cycle pediatric data
    data_file = Path(__file__).parent.parent.parent / "data" / "processed" / "multi_cycle_pediatric.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file)
    print(f"\n[OK] Loaded: {data_file.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # Calculate physical activity features
    df = calculate_all_physical_features(df)

    # Save result
    output_file = data_file.parent / "pediatric_with_physical_features.csv"
    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file.name}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  New columns: {df.shape[1]}")
