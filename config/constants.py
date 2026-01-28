"""
Project-wide constants and configurations for The Pediatric Sentinel.

This module defines paths, dataset configurations, model parameters,
and thresholds used throughout the project.
"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ============================================================================
# NHANES Dataset Configuration
# ============================================================================

# Multiple NHANES cycles configuration for larger dataset
NHANES_CYCLES = {
    "2013-2014": {
        "suffix": "_H",
        "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/",
        "year": "2013"
    },
    "2015-2016": {
        "suffix": "_I",
        "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/",
        "year": "2015"
    },
    "2017-2018": {
        "suffix": "_J",
        "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/",
        "year": "2017"
    },
    "2019-2020": {
        "suffix": "_K",
        "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2019/DataFiles/",
        "year": "2019"
    }
}

# Legacy single-cycle support (for backwards compatibility)
NHANES_CYCLE = "2015-2016"
NHANES_BASE_URL = NHANES_CYCLES["2015-2016"]["base_url"]

# Dataset base names (without cycle suffix)
DATASET_NAMES = {
    "DEMO": "Demographics",
    "HSCRP": "High-sensitivity CRP",
    "GLU": "Glucose",
    "INS": "Insulin",
    "PAQ": "Physical Activity",
    "DR1TOT": "Dietary Interview"
}

ENHANCED_DATASET_NAMES = {
    "GHB": "Glycohemoglobin (HbA1c)",
    "TRIGLY": "Triglycerides & Cholesterol",
    "BPX": "Blood Pressure",
    "BMX": "Body Measures",
    "MCQ": "Medical Conditions",
    "SLQ": "Sleep Disorders"
}

# Generate dataset codes for each cycle
def generate_datasets_for_cycle(cycle_name):
    """Generate dataset dictionary for a specific cycle."""
    suffix = NHANES_CYCLES[cycle_name]["suffix"]

    core = {f"{name}{suffix}": f"{name}{suffix}.xpt"
            for name in DATASET_NAMES.keys()}
    enhanced = {f"{name}{suffix}": f"{name}{suffix}.xpt"
                for name in ENHANCED_DATASET_NAMES.keys()}

    return core, enhanced, {**core, **enhanced}

# Current cycle datasets (for backwards compatibility)
DATASETS = {
    "DEMO_I": "DEMO_I.xpt",
    "HSCRP_I": "HSCRP_I.xpt",
    "GLU_I": "GLU_I.xpt",
    "INS_I": "INS_I.xpt",
    "PAQ_I": "PAQ_I.xpt",
    "DR1TOT_I": "DR1TOT_I.xpt",
}

ENHANCED_DATASETS = {
    "GHB_I": "GHB_I.xpt",
    "TRIGLY_I": "TRIGLY_I.xpt",
    "BPX_I": "BPX_I.xpt",
    "BMX_I": "BMX_I.xpt",
    "MCQ_I": "MCQ_I.xpt",
    "SLQ_I": "SLQ_I.xpt",
}

ALL_DATASETS = {**DATASETS, **ENHANCED_DATASETS}

# All datasets across all cycles
ALL_CYCLES_DATASETS = {}
for cycle_name in NHANES_CYCLES.keys():
    _, _, cycle_datasets = generate_datasets_for_cycle(cycle_name)
    ALL_CYCLES_DATASETS.update(cycle_datasets)

# NHANES variable names (column mappings)
NHANES_VARIABLES = {
    # Merge key
    "SEQN": "Respondent Sequence Number",

    # Demographics (P_DEMO)
    "RIDAGEYR": "Age in years at screening",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    "BMXBMI": "Body Mass Index (kg/m²)",
    "RIDRETH3": "Race/Hispanic origin w/ NH Asian",

    # CRP (P_HSCRP)
    "LBXHSCRP": "High-sensitivity C-Reactive Protein (mg/L)",

    # Biochemistry (P_BIOPRO)
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBXIN": "Fasting Insulin (μU/mL)",

    # Physical Activity (P_PAQ)
    "PAD680": "Minutes sedentary activity - weekday",
    "PAQ706": "Days physically active at least 60 min",
    "PAD733": "Minutes vigorous-intensity activity - weekday",

    # Dietary (P_DR1TOT)
    "DR1TSUGR": "Total sugars (gm)",
    "DR1TFIBE": "Dietary fiber (gm)",

    # === ENHANCED PARAMETERS ===

    # Glycohemoglobin (P_GHB)
    "LBXGH": "Glycohemoglobin (HbA1c) (%)",

    # Lipid Panel (P_TRIGLY)
    "LBXTR": "Triglycerides (mg/dL)",
    "LBDLDL": "LDL-cholesterol (mg/dL)",
    "LBDHDD": "HDL-cholesterol (mg/dL)",
    "LBXTC": "Total cholesterol (mg/dL)",

    # Blood Pressure (P_BPXO)
    "BPXOSY1": "Systolic blood pressure (mmHg) - 1st reading",
    "BPXODI1": "Diastolic blood pressure (mmHg) - 1st reading",

    # Body Measures (P_BMX)
    "BMXWAIST": "Waist circumference (cm)",
    "BMXHT": "Standing height (cm)",
    "BMXWT": "Weight (kg)",

    # Medical Conditions (P_MCQ)
    "MCQ300C": "Family history of diabetes",
    "MCQ160E": "Ever told you had diabetes",

    # Sleep Disorders (P_SLQ)
    "SLD012": "Sleep hours - weekdays",
    "SLD013": "Sleep hours - weekends",

    # Screen Time/Sedentary (P_SXQY or similar)
    # Note: Screen time variables may be in P_PAQ or separate dataset
}

# ============================================================================
# Age Filter for Pediatric Population
# ============================================================================

MIN_AGE = 12  # Minimum age (inclusive)
MAX_AGE = 19  # Maximum age (inclusive)

# ============================================================================
# HOMA-IR Calculation and Thresholds
# ============================================================================

# HOMA-IR = (Fasting Glucose [mg/dL] × Fasting Insulin [μU/mL]) / 405
HOMA_IR_DIVISOR = 405

# Clinical thresholds for insulin resistance
HOMA_IR_NORMAL = 2.5      # Below this: Normal insulin sensitivity
HOMA_IR_ELEVATED = 2.5    # 2.5-5.0: Insulin resistance
HOMA_IR_SEVERE = 5.0      # Above this: Severe insulin resistance

# Normal fasting glucose threshold (mg/dL)
# Used for early detection: high HOMA-IR with normal glucose
NORMAL_GLUCOSE_THRESHOLD = 100

# ============================================================================
# Risk Classification (for ROC/AUC Analysis)
# ============================================================================

# Convert continuous HOMA-IR to binary classification for ROC/AUC curves
# This allows us to evaluate the model's ability to distinguish between
# "at-risk" and "not at-risk" individuals

# Binary classification threshold
RISK_THRESHOLD = HOMA_IR_NORMAL  # 2.5 (insulin resistance threshold)

# Multi-class risk categories (for detailed analysis)
RISK_CATEGORIES = {
    "Low Risk": (0, 2.5),           # HOMA-IR < 2.5 (normal)
    "Moderate Risk": (2.5, 5.0),    # HOMA-IR 2.5-5.0 (insulin resistant)
    "High Risk": (5.0, float('inf'))  # HOMA-IR > 5.0 (severe)
}

# Alternative classification based on combined criteria
# (for more nuanced risk assessment)
COMBINED_RISK_CRITERIA = {
    "homa_ir_threshold": 2.5,
    "glucose_threshold": 100,
    "crp_threshold": 3.0,
    "bmi_threshold": 25.0,  # Overweight cutoff
}

# ============================================================================
# Machine Learning Model Parameters
# ============================================================================

# Target variable
TARGET_COLUMN = "HOMA_IR"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/validation/test split ratios
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Cross-validation
N_FOLDS = 5

# Performance targets
TARGET_R2 = 0.70              # Minimum R² score on test set
MAX_CV_STD = 0.05             # Maximum standard deviation in CV scores
MIN_EARLY_DETECTION_RATE = 0.75  # Minimum early detection sensitivity

# ============================================================================
# Feature Names
# ============================================================================

# Required model inputs (4 core features)
REQUIRED_FEATURES = [
    "physical_activity_score",  # Composite activity metric
    "sugar_intake",             # Total sugars (g/day)
    "fiber_intake",             # Dietary fiber (g/day)
    "crp_level",                # High-sensitivity CRP (mg/L)
]

# Additional features for model enhancement
ADDITIONAL_FEATURES = [
    "age",                      # Age in years (12-19)
    "gender",                   # 1=Male, 2=Female
    "bmi",                      # Body Mass Index
    "synthetic_mirna",          # miRNA-155 proxy from CRP
    "nutritional_stress_index", # Sugar/fiber ratio (0-100)
    "sedentary_minutes",        # Daily sedentary time
    "glucose",                  # Fasting glucose (mg/dL)
    "insulin",                  # Fasting insulin (μU/mL)
]

# Enhanced features (from additional datasets)
ENHANCED_FEATURES = [
    "hba1c",                    # Glycohemoglobin (%)
    "triglycerides",            # Triglycerides (mg/dL)
    "ldl_cholesterol",          # LDL cholesterol (mg/dL)
    "hdl_cholesterol",          # HDL cholesterol (mg/dL)
    "systolic_bp",              # Systolic blood pressure (mmHg)
    "diastolic_bp",             # Diastolic blood pressure (mmHg)
    "waist_circumference",      # Waist circumference (cm)
    "family_history_diabetes",  # Binary: family history
    "sleep_hours",              # Average sleep hours
]

# All features combined
ALL_FEATURES = REQUIRED_FEATURES + ADDITIONAL_FEATURES

# ============================================================================
# Feature Engineering Parameters
# ============================================================================

# Synthetic miRNA-155 generation
MIRNA_METHOD = "log_linear"  # Options: "log_linear", "sigmoid"
MIRNA_WEIGHT_FACTOR = 1.2    # Weight for log-linear method
MIRNA_SIGMOID_K = 0.5        # Sigmoid steepness
MIRNA_SIGMOID_THRESHOLD = 3.0  # CRP threshold for sigmoid (mg/L)

# Nutritional Stress Index
NSI_SCALE_MIN = 0
NSI_SCALE_MAX = 100
NSI_LOW_RISK = 33      # Below this: Low dietary stress
NSI_MEDIUM_RISK = 66   # 33-66: Medium dietary stress
                       # Above 66: High dietary stress

# Physical Activity Categories (minutes/week)
INACTIVE_THRESHOLD = 150      # <150: Inactive
MODERATE_THRESHOLD = 300      # 150-300: Moderate
                              # >300: Active

# ============================================================================
# Random Forest Hyperparameters
# ============================================================================

RF_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  # Use all available cores
}

# Hyperparameter tuning grid
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# ============================================================================
# Evaluation Metrics Configuration
# ============================================================================

# Regression metrics (for continuous HOMA-IR prediction)
REGRESSION_METRICS = [
    "r2_score",           # R² - Variance explained by model
    "rmse",               # Root Mean Squared Error
    "mae",                # Mean Absolute Error
    "mape",               # Mean Absolute Percentage Error
    "median_ae",          # Median Absolute Error (robust to outliers)
]

# Classification metrics (for binary risk prediction: at-risk vs not at-risk)
CLASSIFICATION_METRICS = [
    "accuracy",           # Overall accuracy
    "precision",          # Positive predictive value
    "recall",             # Sensitivity (true positive rate)
    "f1_score",           # Harmonic mean of precision and recall
    "roc_auc",            # Area Under ROC Curve ⭐ KEY METRIC
    "pr_auc",             # Precision-Recall AUC (better for imbalanced data)
    "specificity",        # True negative rate
    "npv",                # Negative predictive value
]

# ROC Curve configuration
ROC_CONFIG = {
    "plot_title": "ROC Curve - Diabetes Risk Classification",
    "xlabel": "False Positive Rate (1 - Specificity)",
    "ylabel": "True Positive Rate (Sensitivity)",
    "optimal_threshold_method": "youden",  # Options: "youden", "closest_to_topleft", "f1"
    "save_path": "models/metadata/roc_curve.png",
}

# Precision-Recall Curve configuration
PR_CONFIG = {
    "plot_title": "Precision-Recall Curve - Diabetes Risk",
    "xlabel": "Recall (Sensitivity)",
    "ylabel": "Precision",
    "save_path": "models/metadata/pr_curve.png",
}

# Confusion Matrix configuration
CM_CONFIG = {
    "normalize": "true",  # Options: None, "true", "pred", "all"
    "labels": ["Not At Risk", "At Risk"],
    "save_path": "models/metadata/confusion_matrix.png",
}

# Performance targets for classification
CLASSIFICATION_TARGETS = {
    "min_roc_auc": 0.80,      # Minimum ROC AUC score
    "min_sensitivity": 0.75,   # Minimum recall (catch 75% of at-risk cases)
    "min_specificity": 0.70,   # Minimum specificity (avoid false alarms)
    "min_f1_score": 0.75,      # Minimum F1 score
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============================================================================
# Streamlit App Configuration
# ============================================================================

APP_TITLE = "The Pediatric Sentinel"
APP_ICON = "🏥"
APP_LAYOUT = "wide"

# Risk category colors
RISK_COLORS = {
    "Low Risk": "green",
    "Moderate Risk": "orange",
    "High Risk": "red",
}

# Default input values for Streamlit app
DEFAULT_INPUTS = {
    "activity": 150,      # Physical activity (minutes/week)
    "sugar": 50.0,        # Sugar intake (grams/day)
    "fiber": 15.0,        # Fiber intake (grams/day)
    "crp": 1.0,           # CRP level (mg/L)
    "age": 15,            # Age (years)
    "gender": "Male",     # Gender
    "bmi": 22.0,          # BMI
}

# ============================================================================
# Data Quality Thresholds
# ============================================================================

# Minimum number of valid samples required after filtering
MIN_SAMPLE_SIZE = 500

# Maximum allowed percentage of missing values per feature
MAX_MISSING_PCT = 0.20  # 20%

# Outlier detection (IQR method)
IQR_MULTIPLIER = 1.5  # Standard: Q1 - 1.5×IQR, Q3 + 1.5×IQR

# ============================================================================
# File Paths for Saved Artifacts
# ============================================================================

MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_data.csv"
FEATURE_ENGINEERED_FILE = PROCESSED_DATA_DIR / "feature_engineered.csv"
FINAL_DATASET_FILE = PROCESSED_DATA_DIR / "final_dataset.csv"

TRAINED_MODEL_FILE = FINAL_MODEL_DIR / "pediatric_sentinel_model.pkl"
MODEL_METADATA_FILE = MODEL_METADATA_DIR / "model_info.json"
FEATURE_IMPORTANCE_FILE = MODEL_METADATA_DIR / "feature_importance.csv"

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR,
        MODELS_DIR, CHECKPOINTS_DIR, FINAL_MODEL_DIR, MODEL_METADATA_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("All project directories created successfully.")


if __name__ == "__main__":
    # Test: Print configuration summary
    print("=" * 70)
    print("THE PEDIATRIC SENTINEL - PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"\nNHANES Cycle: {NHANES_CYCLE}")
    print(f"Datasets to download: {len(DATASETS)}")
    for code, filename in DATASETS.items():
        print(f"  - {code}: {filename}")
    print(f"\nAge Range: {MIN_AGE}-{MAX_AGE} years")
    print(f"Target Variable: {TARGET_COLUMN}")
    print(f"Required Features: {len(REQUIRED_FEATURES)}")
    print(f"  - {', '.join(REQUIRED_FEATURES)}")
    print(f"\nPerformance Targets:")
    print(f"  - R² Score: >{TARGET_R2}")
    print(f"  - Early Detection Rate: >{MIN_EARLY_DETECTION_RATE}")
    print("=" * 70)

    # Ensure directories exist
    ensure_directories()
