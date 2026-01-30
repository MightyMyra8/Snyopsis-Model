"""
Composite Diabetes Risk Score

Combines HOMA-IR (insulin resistance) and HOMA-B (beta-cell function)
to provide a comprehensive diabetes risk assessment.

The composite score identifies 4 progression stages:
  Stage 1 (Low Risk): Normal IR, Normal Beta-Cell
  Stage 2 (Moderate Risk): High IR, Normal Beta-Cell
  Stage 3 (High Risk): High IR, High Beta-Cell (Compensatory)
  Stage 4 (CRITICAL): High IR, Low Beta-Cell (Exhaustion)
"""

import numpy as np
import pandas as pd


class CompositeDiabetesRisk:
    """Calculate composite diabetes risk from HOMA-IR and HOMA-B."""

    def __init__(self):
        """Initialize risk calculator with clinical thresholds."""
        # HOMA-IR thresholds (insulin resistance)
        self.homa_ir_normal = 2.5
        self.homa_ir_severe = 5.0

        # HOMA-B thresholds (beta-cell function)
        self.homa_b_low = 50.0      # Dysfunction/exhaustion
        self.homa_b_high = 150.0    # Compensatory hyperinsulinemia

    def categorize_homa_ir(self, homa_ir):
        """
        Categorize insulin resistance.

        Args:
            homa_ir: HOMA-IR value or array

        Returns:
            Category: 'normal', 'moderate', or 'severe'
        """
        if isinstance(homa_ir, (list, np.ndarray, pd.Series)):
            categories = []
            for ir in homa_ir:
                if pd.isna(ir):
                    categories.append(np.nan)
                elif ir < self.homa_ir_normal:
                    categories.append('normal')
                elif ir < self.homa_ir_severe:
                    categories.append('moderate')
                else:
                    categories.append('severe')
            return np.array(categories)
        else:
            if pd.isna(homa_ir):
                return np.nan
            elif homa_ir < self.homa_ir_normal:
                return 'normal'
            elif homa_ir < self.homa_ir_severe:
                return 'moderate'
            else:
                return 'severe'

    def categorize_homa_b(self, homa_b):
        """
        Categorize beta-cell function.

        Args:
            homa_b: HOMA-B value or array (%)

        Returns:
            Category: 'low', 'normal', or 'high'
        """
        if isinstance(homa_b, (list, np.ndarray, pd.Series)):
            categories = []
            for b in homa_b:
                if pd.isna(b):
                    categories.append(np.nan)
                elif b < self.homa_b_low:
                    categories.append('low')
                elif b <= self.homa_b_high:
                    categories.append('normal')
                else:
                    categories.append('high')
            return np.array(categories)
        else:
            if pd.isna(homa_b):
                return np.nan
            elif homa_b < self.homa_b_low:
                return 'low'
            elif homa_b <= self.homa_b_high:
                return 'normal'
            else:
                return 'high'

    def calculate_composite_risk(self, homa_ir, homa_b):
        """
        Calculate composite diabetes risk from HOMA-IR and HOMA-B.

        Risk Stages:
        - Stage 1 (Low Risk): Normal IR + Normal/Low Beta-Cell
        - Stage 2 (Moderate Risk): Elevated IR + Normal Beta-Cell
        - Stage 3 (High Risk): Elevated IR + High Beta-Cell (Compensatory)
        - Stage 4 (CRITICAL): Elevated IR + Low Beta-Cell (Exhaustion)

        Args:
            homa_ir: HOMA-IR value(s)
            homa_b: HOMA-B value(s) in %

        Returns:
            Dictionary with risk_stage, risk_score (0-100), and recommendations
        """
        # Handle single values vs arrays
        is_array = isinstance(homa_ir, (list, np.ndarray, pd.Series))

        ir_cat = self.categorize_homa_ir(homa_ir)
        b_cat = self.categorize_homa_b(homa_b)

        if is_array:
            results = []
            for ir, b, ir_c, b_c in zip(homa_ir, homa_b, ir_cat, b_cat):
                results.append(self._calculate_single_risk(ir, b, ir_c, b_c))
            return results
        else:
            return self._calculate_single_risk(homa_ir, homa_b, ir_cat, b_cat)

    def _calculate_single_risk(self, homa_ir, homa_b, ir_cat, b_cat):
        """Calculate risk for a single individual."""
        # Handle missing values
        if pd.isna(homa_ir) or pd.isna(homa_b):
            return {
                'risk_stage': 'Unknown',
                'risk_score': np.nan,
                'risk_level': 'Unknown',
                'progression_warning': 'Insufficient data'
            }

        # Determine progression stage
        if ir_cat == 'normal':
            # Stage 1: Healthy (regardless of beta-cell)
            stage = 'Stage 1: Healthy'
            score = 10 + (homa_ir / self.homa_ir_normal) * 15  # 10-25
            level = 'Low Risk'
            warning = 'Maintain healthy lifestyle to prevent insulin resistance'

        elif b_cat == 'normal':
            # Stage 2: Insulin Resistance (normal beta-cell compensation)
            stage = 'Stage 2: Insulin Resistance'
            score = 30 + ((homa_ir - self.homa_ir_normal) /
                          (self.homa_ir_severe - self.homa_ir_normal)) * 20  # 30-50
            level = 'Moderate Risk'
            warning = 'Insulin resistance detected - lifestyle changes needed'

        elif b_cat == 'high':
            # Stage 3: Compensatory (pancreas overworking)
            stage = 'Stage 3: Compensatory'
            compensatory_factor = (homa_b - self.homa_b_high) / 100  # Scale by excess
            score = 50 + min(homa_ir / 10, 1) * 30 + compensatory_factor * 10  # 50-90
            level = 'High Risk'
            warning = 'COMPENSATORY PHASE: Pancreas is overworking - high risk of exhaustion!'

        else:  # b_cat == 'low'
            # Stage 4: Beta-Cell Exhaustion (CRITICAL!)
            stage = 'Stage 4: Beta-Cell Exhaustion'
            score = 85 + min(homa_ir / 10, 1) * 15  # 85-100
            level = 'CRITICAL'
            warning = 'BETA-CELL EXHAUSTION: Pancreas is failing - immediate medical intervention needed!'

        # Ensure score is 0-100
        score = max(0, min(100, score))

        return {
            'risk_stage': stage,
            'risk_score': round(score, 1),
            'risk_level': level,
            'progression_warning': warning,
            'homa_ir': round(homa_ir, 2),
            'homa_b': round(homa_b, 1),
            'ir_category': ir_cat,
            'b_category': b_cat
        }

    def get_recommendations(self, risk_result):
        """
        Get personalized recommendations based on risk assessment.

        Args:
            risk_result: Output from calculate_composite_risk()

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        stage = risk_result['risk_stage']

        if 'Stage 1' in stage:
            # Healthy - prevention
            recommendations.append({
                'priority': 'Low',
                'category': 'Prevention',
                'action': 'Maintain current healthy lifestyle',
                'details': 'Continue regular physical activity (60+ min/day) and balanced diet'
            })

        elif 'Stage 2' in stage:
            # Insulin Resistance - early intervention
            recommendations.append({
                'priority': 'Moderate',
                'category': 'Lifestyle Intervention',
                'action': 'Increase physical activity to 60+ minutes daily',
                'details': 'Focus on moderate-to-vigorous aerobic exercise and strength training'
            })
            recommendations.append({
                'priority': 'Moderate',
                'category': 'Diet Modification',
                'action': 'Reduce added sugars to <25g/day',
                'details': 'Increase fiber intake to 25-30g/day from whole grains, fruits, vegetables'
            })

        elif 'Stage 3' in stage:
            # Compensatory - aggressive intervention
            recommendations.append({
                'priority': 'High',
                'category': 'Medical Monitoring',
                'action': 'Consult healthcare provider for glucose monitoring',
                'details': 'Regular HbA1c testing recommended every 3-6 months'
            })
            recommendations.append({
                'priority': 'High',
                'category': 'Intensive Lifestyle',
                'action': 'Aggressive diet and exercise intervention',
                'details': 'Work with nutritionist and exercise physiologist for structured program'
            })
            recommendations.append({
                'priority': 'High',
                'category': 'Inflammation Control',
                'action': 'Address systemic inflammation',
                'details': 'Anti-inflammatory diet (omega-3, leafy greens, berries), stress reduction'
            })

        else:  # Stage 4
            # Beta-Cell Exhaustion - CRITICAL
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Immediate Medical Care',
                'action': 'See endocrinologist IMMEDIATELY',
                'details': 'Beta-cell exhaustion requires medical intervention - may need medication'
            })
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Glucose Monitoring',
                'action': 'Daily glucose monitoring',
                'details': 'Track fasting and post-meal glucose levels'
            })
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Medical Nutrition Therapy',
                'action': 'Work with certified diabetes educator',
                'details': 'Structured meal planning to prevent glucose spikes'
            })

        return recommendations


# Convenience functions
def calculate_risk(homa_ir, homa_b):
    """
    Calculate composite diabetes risk.

    Args:
        homa_ir: HOMA-IR value(s)
        homa_b: HOMA-B value(s) in %

    Returns:
        Risk assessment dictionary or list
    """
    calculator = CompositeDiabetesRisk()
    return calculator.calculate_composite_risk(homa_ir, homa_b)


def get_risk_recommendations(homa_ir, homa_b):
    """
    Get risk assessment with recommendations.

    Args:
        homa_ir: HOMA-IR value
        homa_b: HOMA-B value in %

    Returns:
        Tuple of (risk_result, recommendations)
    """
    calculator = CompositeDiabetesRisk()
    risk_result = calculator.calculate_composite_risk(homa_ir, homa_b)
    recommendations = calculator.get_recommendations(risk_result)
    return risk_result, recommendations


if __name__ == "__main__":
    # Test the composite risk calculator
    print("=" * 70)
    print("COMPOSITE DIABETES RISK CALCULATOR - TESTS")
    print("=" * 70)

    calculator = CompositeDiabetesRisk()

    # Test cases
    test_cases = [
        ("Healthy", 1.8, 100.0),
        ("Insulin Resistance", 3.5, 120.0),
        ("Compensatory Phase", 5.8, 240.0),
        ("Beta-Cell Exhaustion", 6.2, 45.0),
    ]

    for name, ir, b in test_cases:
        print(f"\n{name}:")
        print(f"  HOMA-IR: {ir}, HOMA-B: {b}%")

        result = calculator.calculate_composite_risk(ir, b)
        print(f"  Stage: {result['risk_stage']}")
        print(f"  Risk Score: {result['risk_score']}/100")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Warning: {result['progression_warning']}")

        recommendations = calculator.get_recommendations(result)
        if recommendations:
            print(f"  Recommendations:")
            for rec in recommendations:
                print(f"    [{rec['priority']}] {rec['action']}")
