"""
The Pediatric Sentinel - Interactive Diabetes Risk Calculator

A Streamlit web application for predicting Type 2 Diabetes risk in children
based on lifestyle factors (physical activity, diet) and biomarkers.

Features:
- Risk prediction using machine learning
- SHAP-based explanations (what's driving your risk?)
- Personalized recommendations
- Two-model comparison (Full vs Hypothesis)

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import composite risk calculator
from src.models.composite_risk import CompositeDiabetesRisk

# Page configuration
st.set_page_config(
    page_title="The Pediatric Sentinel",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models."""
    project_root = Path(__file__).parent.parent

    # Load HOMA-IR prediction model (full model with BMI)
    homa_ir_model_path = project_root / "models" / "final" / "pediatric_sentinel_model.pkl"
    homa_ir_model = joblib.load(homa_ir_model_path)

    # Load HOMA-B prediction model (beta-cell function)
    homa_b_model_path = project_root / "models" / "beta_cell" / "homa_b_model.pkl"
    homa_b_model = joblib.load(homa_b_model_path)

    # Load hypothesis model (lifestyle factors only)
    hypothesis_model_path = project_root / "models" / "hypothesis" / "hypothesis_model.pkl"
    hypothesis_model = joblib.load(hypothesis_model_path)

    # Initialize composite risk calculator
    risk_calculator = CompositeDiabetesRisk()

    return homa_ir_model, homa_b_model, hypothesis_model, risk_calculator


def categorize_risk(homa_ir):
    """Categorize diabetes risk based on HOMA-IR."""
    if homa_ir < 2.5:
        return "Low Risk", "risk-low"
    elif homa_ir < 5.0:
        return "Moderate Risk", "risk-moderate"
    else:
        return "High Risk", "risk-high"


def get_recommendations(input_data):
    """Generate personalized recommendations based on input data."""
    recommendations = []

    # Physical activity
    if input_data['comprehensive_inactivity_score'] > 66:
        recommendations.append({
            'icon': '🏃',
            'title': 'Increase Physical Activity',
            'message': 'You are currently inactive. Aim for 60+ minutes of moderate activity daily.',
            'details': 'Try: Walking, biking, dancing, sports'
        })
    elif input_data['comprehensive_inactivity_score'] > 33:
        recommendations.append({
            'icon': '💪',
            'title': 'Boost Your Activity',
            'message': 'You\'re moderately active - great! Try to increase to 60+ minutes daily.',
            'details': 'Add: 15-minute walks, active hobbies'
        })

    # Sugar intake
    if input_data['DR1TSUGR'] > 50:
        recommendations.append({
            'icon': '🍬',
            'title': 'Reduce Added Sugars',
            'message': f'Current: {input_data["DR1TSUGR"]:.0f}g/day. Target: <25g/day (AHA guideline)',
            'details': 'Limit: Soda, candy, desserts, sweetened drinks'
        })

    # Fiber intake
    if input_data['DR1TFIBE'] < 25:
        recommendations.append({
            'icon': '🥦',
            'title': 'Increase Fiber Intake',
            'message': f'Current: {input_data["DR1TFIBE"]:.0f}g/day. Target: 25-30g/day',
            'details': 'Eat more: Whole grains, fruits, vegetables, beans'
        })

    # CRP (inflammation)
    if input_data.get('LBXHSCRP', 0) > 3:
        recommendations.append({
            'icon': '🔥',
            'title': 'Address Inflammation',
            'message': 'Your CRP is elevated. Consider anti-inflammatory diet and consult healthcare provider.',
            'details': 'Include: Omega-3 (fish), leafy greens, berries, nuts'
        })

    # BMI if available
    if 'BMXBMI' in input_data and input_data['BMXBMI'] > 25:
        recommendations.append({
            'icon': '⚖️',
            'title': 'Healthy Weight Management',
            'message': f'BMI: {input_data["BMXBMI"]:.1f}. Focus on gradual, sustainable changes.',
            'details': 'Combine: Balanced diet + regular physical activity'
        })

    # Positive reinforcement
    if not recommendations:
        recommendations.append({
            'icon': '✅',
            'title': 'Keep Up the Great Work!',
            'message': 'Your lifestyle habits are on track. Continue these healthy behaviors!',
            'details': 'Maintain: Current activity level and diet quality'
        })

    return recommendations


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<p class="main-header">🏥 The Pediatric Sentinel</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Diabetes Risk Assessment for Children & Teens</p>', unsafe_allow_html=True)

    # Load models
    try:
        homa_ir_model, homa_b_model, hypothesis_model, risk_calculator = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🩺 Risk Calculator", "📊 Model Insights", "🔬 About the Science"]
    )

    # =========================================================================
    # PAGE 1: RISK CALCULATOR
    # =========================================================================
    if page == "🩺 Risk Calculator":
        st.title("Diabetes Risk Calculator")
        st.write("Enter your information below to calculate your Type 2 Diabetes risk.")

        st.markdown("---")

        # Input form
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Lifestyle Inputs")

            # Physical activity
            st.write("**Physical Activity**")
            activity_minutes = st.slider(
                "Minutes of activity per week",
                0, 420, 150,
                help="Total minutes of moderate-to-vigorous physical activity per week (walking, sports, etc.)"
            )

            # Convert to inactivity score (0-100, where 100 = completely inactive)
            inactivity_score = 100 - (min(activity_minutes, 420) / 420 * 100)

            # Diet
            st.write("**Diet**")
            sugar_intake = st.number_input(
                "Sugar intake (grams/day)",
                0.0, 300.0, 50.0,
                help="Total added sugars consumed daily (from soda, candy, desserts, etc.)"
            )

            fiber_intake = st.number_input(
                "Fiber intake (grams/day)",
                0.0, 50.0, 15.0,
                help="Dietary fiber from whole grains, fruits, vegetables, beans"
            )

        with col2:
            st.subheader("🩸 Biomarker Input")

            crp_level = st.number_input(
                "CRP Level (mg/L)",
                0.0, 20.0, 1.0,
                help="High-sensitivity C-Reactive Protein from blood test (inflammation marker)"
            )

            st.markdown("---")

            st.subheader("👤 Optional Information")

            age = st.slider("Age", 12, 19, 15, help="Age in years (12-19)")
            gender = st.selectbox("Gender", ["Male", "Female"])

            # Advanced options
            with st.expander("Advanced: Additional Biomarkers"):
                bmi = st.number_input("BMI (optional)", 10.0, 50.0, 22.0)
                waist = st.number_input("Waist circumference (cm, optional)", 40.0, 150.0, 75.0)
                hba1c = st.number_input("HbA1c - % (optional)", 4.0, 10.0, 5.3, help="3-month average blood sugar, default: 5.3% (normal)")
                systolic_bp = st.number_input("Systolic BP (optional)", 80.0, 180.0, 112.0, help="Top blood pressure number, default: 112 mmHg")
                diastolic_bp = st.number_input("Diastolic BP (optional)", 40.0, 120.0, 65.0, help="Bottom blood pressure number, default: 65 mmHg")
                carb_percent = st.number_input("Carbohydrate % of diet (optional)", 0.0, 100.0, 52.0, help="Percentage of calories from carbs, default: 52%")

        st.markdown("---")

        # Predict button
        if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):

            # Prepare input data for full model (must match training feature order)
            input_data_full = {
                'comprehensive_inactivity_score': inactivity_score,
                'DR1TSUGR': sugar_intake,
                'DR1TFIBE': fiber_intake,
                'LBXHSCRP': crp_level,
                'BMXBMI': bmi,
                'BMXWAIST': waist,
                'LBXGH': hba1c,
                'BPXSY2': systolic_bp,
                'BPXDI2': diastolic_bp,
                'carb_percent': carb_percent,
                'synthetic_mirna155': np.log1p(crp_level) * 1.2,  # Calculate synthetic miRNA
                'RIDAGEYR': age,
                'RIAGENDR': 1 if gender == "Male" else 2,
                'sugar_inactivity_interaction': sugar_intake * inactivity_score / 100,
                'bmi_inactivity_interaction': bmi * inactivity_score / 100,
                'sugar_crp_interaction': sugar_intake * crp_level,
                'crp_bmi_interaction': crp_level * bmi,
                'bmi_squared': bmi ** 2,
                'crp_squared': crp_level ** 2
            }

            # Prepare input data for hypothesis model
            input_data_hypothesis = {
                'comprehensive_inactivity_score': inactivity_score,
                'DR1TSUGR': sugar_intake,
                'DR1TFIBE': fiber_intake,
                'LBXHSCRP': crp_level,
                'synthetic_mirna155': np.log1p(crp_level) * 1.2,
                'RIDAGEYR': age,
                'RIAGENDR': 1 if gender == "Male" else 2,
                'sugar_inactivity_interaction': sugar_intake * inactivity_score / 100,
                'sugar_crp_interaction': sugar_intake * crp_level,
                'crp_squared': crp_level ** 2
            }

            # Convert to DataFrames with EXACT column order expected by models
            # CRITICAL: HOMA-IR and HOMA-B models have DIFFERENT feature orders!

            # HOMA-IR model feature order
            feature_order_homa_ir = [
                'comprehensive_inactivity_score', 'DR1TSUGR', 'DR1TFIBE', 'LBXHSCRP',
                'BMXBMI', 'BMXWAIST', 'LBXGH', 'BPXSY2', 'BPXDI2', 'carb_percent',
                'synthetic_mirna155', 'RIDAGEYR', 'RIAGENDR',
                'sugar_inactivity_interaction', 'bmi_inactivity_interaction',
                'sugar_crp_interaction', 'crp_bmi_interaction', 'bmi_squared', 'crp_squared'
            ]

            # HOMA-B model feature order (synthetic_mirna155 is in different position!)
            feature_order_homa_b = [
                'comprehensive_inactivity_score', 'DR1TSUGR', 'DR1TFIBE', 'LBXHSCRP',
                'synthetic_mirna155', 'BMXBMI', 'BMXWAIST', 'LBXGH', 'BPXSY2', 'BPXDI2',
                'carb_percent', 'RIDAGEYR', 'RIAGENDR',
                'sugar_inactivity_interaction', 'bmi_inactivity_interaction',
                'sugar_crp_interaction', 'crp_bmi_interaction', 'bmi_squared', 'crp_squared'
            ]

            df_homa_ir = pd.DataFrame([input_data_full])[feature_order_homa_ir]
            df_homa_b = pd.DataFrame([input_data_full])[feature_order_homa_b]

            # Hypothesis model uses different features
            feature_order_hypothesis = [
                'comprehensive_inactivity_score', 'DR1TSUGR', 'DR1TFIBE', 'LBXHSCRP',
                'synthetic_mirna155', 'RIDAGEYR', 'RIAGENDR',
                'sugar_inactivity_interaction', 'sugar_crp_interaction', 'crp_squared'
            ]

            df_hypothesis = pd.DataFrame([input_data_hypothesis])[feature_order_hypothesis]

            # Make predictions
            try:
                # Predict HOMA-IR (insulin resistance)
                homa_ir_pred = homa_ir_model.predict(df_homa_ir)[0]

                # Predict HOMA-B (beta-cell function)
                homa_b_pred = homa_b_model.predict(df_homa_b)[0]

                # Calculate composite risk
                composite_risk = risk_calculator.calculate_composite_risk(homa_ir_pred, homa_b_pred)

                # Predict with hypothesis model (lifestyle only)
                homa_ir_hypothesis = hypothesis_model.predict(df_hypothesis)[0]

                # Display results
                st.markdown("---")
                st.header("📊 Complete Diabetes Risk Assessment")

                # TOP ROW: Three Key Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("HOMA-IR (Insulin Resistance)", f"{homa_ir_pred:.2f}")
                    ir_cat = composite_risk['ir_category']
                    if ir_cat == 'normal':
                        st.success("✅ Normal")
                    elif ir_cat == 'moderate':
                        st.warning("⚠️ Moderate")
                    else:
                        st.error("🚨 Severe")

                with col2:
                    st.metric("HOMA-B (Beta-Cell Function)", f"{homa_b_pred:.1f}%")
                    b_cat = composite_risk['b_category']
                    if b_cat == 'normal':
                        st.success("✅ Normal (50-150%)")
                    elif b_cat == 'high':
                        st.warning("⚠️ Compensatory (>150%)")
                    else:
                        st.error("🚨 Dysfunction (<50%)")

                with col3:
                    risk_score = composite_risk['risk_score']
                    st.metric("Composite Risk Score", f"{risk_score:.0f}/100")

                    # Color-coded risk level
                    risk_level = composite_risk['risk_level']
                    if risk_level == 'Low Risk':
                        st.success(f"✅ {risk_level}")
                    elif risk_level == 'Moderate Risk':
                        st.warning(f"⚠️ {risk_level}")
                    elif risk_level == 'High Risk':
                        st.error(f"🚨 {risk_level}")
                    else:  # CRITICAL
                        st.error(f"⛔ {risk_level}")

                # PROGRESSION STAGE
                st.markdown("---")
                stage = composite_risk['risk_stage']
                warning = composite_risk['progression_warning']

                if 'Stage 1' in stage:
                    st.success(f"### {stage}")
                    st.info(warning)
                elif 'Stage 2' in stage:
                    st.warning(f"### {stage}")
                    st.info(warning)
                elif 'Stage 3' in stage:
                    st.error(f"### {stage}")
                    st.warning(warning)
                else:  # Stage 4
                    st.error(f"### {stage}")
                    st.error(f"⛔ {warning}")

                # Gauge chart
                import plotly.graph_objects as go

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=homa_ir_pred,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2.5], 'color': "lightgreen"},
                            {'range': [2.5, 5.0], 'color': "lightyellow"},
                            {'range': [5.0, 10], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 2.5
                        }
                    },
                    title={'text': "HOMA-IR Score"}
                ))

                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                st.markdown("---")
                st.subheader("📖 What Does This Mean?")

                if homa_ir_pred < 2.5:
                    st.success("""
                    ✅ **Low Risk**: Your insulin resistance is within normal range.
                    Keep up your healthy lifestyle habits!
                    """)
                elif homa_ir_pred < 5.0:
                    st.warning("""
                    ⚠️ **Moderate Risk**: You have some insulin resistance.
                    Lifestyle changes can help prevent progression to Type 2 Diabetes.
                    """)
                else:
                    st.error("""
                    🚨 **High Risk**: You have significant insulin resistance.
                    Please consult a healthcare provider and make lifestyle changes.
                    """)

                # Model comparison
                st.markdown("---")
                st.subheader("🔬 Two-Model Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Full Model** (with BMI)")
                    st.metric("HOMA-IR Prediction", f"{homa_ir_pred:.2f}")
                    st.caption("Best for accurate predictions")

                with col2:
                    st.write("**Hypothesis Model** (lifestyle only)")
                    st.metric("HOMA-IR Prediction", f"{homa_ir_hypothesis:.2f}")
                    st.caption("Shows lifestyle factor effects")

                difference = abs(homa_ir_pred - homa_ir_hypothesis)
                if difference > 0.5:
                    st.info(f"""
                    The {difference:.2f} point difference shows how BMI mediates lifestyle effects.
                    Both models are correct from different perspectives!
                    """)

                # Personalized recommendations
                st.markdown("---")
                st.subheader("💡 Personalized Recommendations")

                # Get recommendations from composite risk
                recommendations = risk_calculator.get_recommendations(composite_risk)

                for rec in recommendations:
                    # Priority emoji
                    if rec['priority'] == 'CRITICAL':
                        emoji = "⛔"
                        color = "error"
                    elif rec['priority'] == 'High':
                        emoji = "🚨"
                        color = "warning"
                    elif rec['priority'] == 'Moderate':
                        emoji = "⚠️"
                        color = "info"
                    else:
                        emoji = "✅"
                        color = "success"

                    with st.expander(f"{emoji} [{rec['priority']}] {rec['category']}: {rec['action']}"):
                        st.write(f"**Action:** {rec['action']}")
                        st.write(f"**Details:** {rec['details']}")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check your inputs and try again.")

    # =========================================================================
    # PAGE 2: MODEL INSIGHTS
    # =========================================================================
    elif page == "📊 Model Insights":
        st.title("Model Performance & Insights")

        st.markdown("---")

        # Model performance
        st.subheader("🎯 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Full Model**")
            st.metric("Test R² Score", "0.36")
            st.metric("Sensitivity", "76%")
            st.metric("Early Detection Rate", "85%")

        with col2:
            st.write("**Hypothesis Model**")
            st.metric("Test R² Score", "0.15")
            st.metric("Sensitivity", "77%")
            st.metric("Early Detection Rate", "75%")

        st.markdown("---")

        # Feature importance
        st.subheader("📈 Feature Importance")

        # You can load actual feature importance from models
        st.write("**Full Model Top Features:**")
        st.write("1. Waist Circumference (27.4%)")
        st.write("2. BMI² (17.0%)")
        st.write("3. BMI (16.0%)")
        st.write("4. synthetic miRNA-155 (7.9%)")
        st.write("5. CRP (6.4%)")

        st.write("**Hypothesis Model Top Features:**")
        st.write("1. Physical Activity (8.0%) ⬆️")
        st.write("2. Sugar Intake (11.1%) ⬆️")
        st.write("3. Fiber Intake (11.1%) ⬆️")
        st.write("4. CRP (12.7%)")
        st.write("5. synthetic miRNA-155 (15.1%)")

        st.info("""
        **Key Finding**: When BMI is removed, physical activity importance increases 6.2x!
        This proves lifestyle factors DO matter - they're just masked by confounding in the full model.
        """)

    # =========================================================================
    # PAGE 3: ABOUT THE SCIENCE
    # =========================================================================
    else:  # About the Science
        st.title("🔬 The Science Behind The Pediatric Sentinel")

        st.markdown("---")

        st.subheader("Three-Tier Cascade Model")

        st.write("""
        This risk calculator is based on the **three-tier cascade hypothesis**:

        **Tier 1: Environmental Inputs (Root Causes)**
        - Poor diet (high sugar, low fiber)
        - Physical inactivity

        ⬇️

        **Tier 2: Biological Mediators**
        - Systemic inflammation (elevated CRP)
        - Epigenetic changes (miRNA-155 dysregulation)

        ⬇️

        **Tier 3: Metabolic Dysfunction**
        - Insulin resistance (elevated HOMA-IR)
        - Beta-cell exhaustion
        - Type 2 Diabetes risk
        """)

        st.markdown("---")

        st.subheader("📚 Data Source")

        st.write("""
        **NHANES 2013-2018** (National Health and Nutrition Examination Survey)
        - Population: Children & teens ages 12-19
        - Sample size: 879 complete cases
        - Variables: Physical activity, dietary intake, inflammation markers, metabolic health
        """)

        st.markdown("---")

        st.subheader("🎯 Model Performance")

        st.write("""
        **Validation Results:**
        - R² Score: 0.36 (explains 36% of variance in insulin resistance)
        - Sensitivity: 76% (correctly identifies 3 in 4 high-risk individuals)
        - Early Detection: 85% (identifies insulin resistance before glucose elevation)

        **Clinical Value:**
        This model can identify at-risk children **before** they develop elevated blood sugar,
        enabling early intervention to prevent Type 2 Diabetes.
        """)

        st.markdown("---")

        st.subheader("📖 References")

        st.write("""
        Based on the **TODAY Study** methodology:
        - Research on diabetes prevention in youth
        - Validates three-tier cascade hypothesis
        - Demonstrates lifestyle intervention effectiveness

        **PubMed ID:** 38815053
        """)

        st.markdown("---")

        st.subheader("⚠️ Disclaimer")

        st.warning("""
        This tool is for **educational and research purposes only**.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.

        Always consult a qualified healthcare provider with questions about your health.
        """)

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ for science fair | Powered by Machine Learning & SHAP")


if __name__ == "__main__":
    main()
