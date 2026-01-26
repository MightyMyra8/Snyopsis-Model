# Manual NHANES Data Download Instructions

## Why Manual Download?

The CDC NHANES website doesn't support direct URL downloads. You need to download files through their web interface.

---

## Quick Start: Download All 12 Datasets by Component

### NHANES 2015-2016 Cycle - Component Pages

**Demographics:** https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2015-2016

**Laboratory:** https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016

**Questionnaire:** https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016

**Examination:** https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016

---

## Step-by-Step Instructions

### 1. Core Datasets (6 files) - REQUIRED

| Dataset | Full Name | Component Page | Find Dataset |
|---------|-----------|----------------|--------------|
| **DEMO_I** | Demographics | [Demographics](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2015-2016) | Search "DEMO_I" → Click dataset name → Download "Data [XPT]" |
| **GLU_I** | Plasma Glucose | [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016) | Search "GLU_I" or "Glucose" → Click "Plasma Fasting Glucose" → Download "Data [XPT]" |
| **INS_I** | Insulin | [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016) | Search "INS_I" or "Insulin" → Click "Insulin" → Download "Data [XPT]" |
| **HSCRP_I** | High-Sensitivity CRP | [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016) | Search "HSCRP_I" or "CRP" → Click "High-Sensitivity C-Reactive Protein" → Download "Data [XPT]" |
| **PAQ_I** | Physical Activity | [Questionnaire](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016) | Search "PAQ_I" or "Physical Activity" → Click dataset → Download "Data [XPT]" |
| **DR1TOT_I** | Dietary Interview | [Examination](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016) | Search "DR1TOT_I" or "Dietary" → Click "Dietary Interview Total Nutrients, First Day" → Download "Data [XPT]" |

**Save these files to:** `C:\Users\Myra Saxena\Documents\SourceCode\ScienceFair\data\raw\`

---

### 2. Enhanced Datasets (6 files) - OPTIONAL (for better model)

| Dataset | Full Name | Component Page | Find Dataset |
|---------|-----------|----------------|--------------|
| **GHB_I** | Glycohemoglobin (HbA1c) | [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016) | Search "GHB_I" or "Glycohemoglobin" → Download "Data [XPT]" |
| **TRIGLY_I** | Triglycerides & Cholesterol | [Laboratory](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2015-2016) | Search "TRIGLY_I" or "Cholesterol" → Click "Cholesterol - Total" → Download "Data [XPT]" |
| **BPX_I** | Blood Pressure | [Examination](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016) | Search "BPX_I" or "Blood Pressure" → Download "Data [XPT]" |
| **BMX_I** | Body Measures | [Examination](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2015-2016) | Search "BMX_I" or "Body Measures" → Download "Data [XPT]" |
| **MCQ_I** | Medical Conditions | [Questionnaire](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016) | Search "MCQ_I" or "Medical Conditions" → Download "Data [XPT]" |
| **SLQ_I** | Sleep Disorders | [Questionnaire](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2015-2016) | Search "SLQ_I" or "Sleep" → Download "Data [XPT]" |

**Save these files to:** `C:\Users\Myra Saxena\Documents\SourceCode\ScienceFair\data\raw\`

---

## Visual Guide

### Method 1: Component-Specific Search

1. **Go to NHANES homepage**
   ```
   https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
   ```

2. **Select Survey: "2015-2016"**
   - Click dropdown at top
   - Choose "2015-2016"

3. **Browse by Component**
   - **Demographics**: Click "Demographics" → Find DEMO_I → Click "Data [XPT]"
   - **Laboratory**: Click "Laboratory" → Find dataset name → Click "Data [XPT]"
   - **Questionnaire**: Click "Questionnaire" → Find dataset name → Click "Data [XPT]"
   - **Examination**: Click "Examination" → Find dataset name → Click "Data [XPT]"

4. **Download each .XPT file**
   - Right-click "Data [XPT]" link
   - Choose "Save Link As..."
   - Save to `data/raw/` folder

---

### Method 2: Direct Search (Faster!)

1. **Go to NHANES Search Page**
   ```
   https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx
   ```

2. **Search for each dataset by code**
   - Type dataset code (e.g., "DEMO_I") in search box
   - Filter by "2015-2016" cycle
   - Click on the result
   - Download "Data [XPT]" file

3. **Repeat for all 12 datasets**

---

## Direct Download Links (FASTEST METHOD!)

**Core Datasets (Right-click → Save As...):**
- Demographics: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt
- Glucose: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.xpt
- Insulin: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/INS_I.xpt
- CRP: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HSCRP_I.xpt
- Physical Activity: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.xpt
- Dietary: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.xpt

**Enhanced Datasets (Right-click → Save As...):**
- HbA1c: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.xpt
- Triglycerides: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/TRIGLY_I.xpt
- Blood Pressure: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.xpt
- Body Measures: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.xpt
- Medical Conditions: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt
- Sleep: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.xpt

**How to Download:**
1. Right-click each link above
2. Choose "Save Link As..." or "Save Target As..."
3. Save to `C:\Users\Myra Saxena\Documents\SourceCode\ScienceFair\data\raw\`
4. Make sure filename keeps the `.xpt` extension

---

## File Checklist

After downloading, you should have these files in `data/raw/`:

```
✓ DEMO_I.XPT      (~3.5 MB)
✓ GLU_I.XPT       (~500 KB)
✓ INS_I.XPT       (~500 KB)
✓ HSCRP_I.XPT     (~500 KB)
✓ PAQ_I.XPT       (~2 MB)
✓ DR1TOT_I.XPT    (~4 MB)

Enhanced (optional):
✓ GHB_I.XPT       (~500 KB)
✓ TRIGLY_I.XPT    (~500 KB)
✓ BPX_I.XPT       (~2 MB)
✓ BMX_I.XPT       (~2 MB)
✓ MCQ_I.XPT       (~3 MB)
✓ SLQ_I.XPT       (~1 MB)
```

**Total size:** ~20 MB for all 12 datasets

---

## Verify Downloads

Once files are in `data/raw/`, run:

```bash
python scripts/download_data.py
```

This will:
1. Skip download (files already exist)
2. Load all .XPT files
3. Clean data (filter ages 12-19)
4. Merge datasets
5. Save to `data/processed/merged_data.csv`

---

## Troubleshooting

### Problem: "File not found" error
**Solution:** Make sure filenames match exactly (case-sensitive!):
- Correct: `DEMO_I.XPT` (uppercase, underscore I)
- Wrong: `demo_i.xpt` or `DEMO-I.XPT`

### Problem: "Header record is not an XPORT file"
**Solution:** You downloaded an HTML page instead of the XPT file
- Right-click the "Data [XPT]" link
- Choose "Save Link As..." (not just clicking it)

### Problem: Files are all ~20 KB
**Solution:** These are error pages, not data files
- Use the direct links above
- On each page, download the "Data [XPT]" file
- Expected file sizes: 500 KB - 4 MB

---

## Alternative: Use Smaller Sample

If downloading is difficult, I can create a script to generate **synthetic sample data** for testing the pipeline. This lets you:
- ✅ Build and test the full pipeline
- ✅ Develop features and model
- ✅ Create visualizations and ROC curves
- ❌ Won't have real scientific results

Let me know if you want the sample data option instead!

---

## After Manual Download

Once files are downloaded to `data/raw/`, run the full pipeline:

```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Run data pipeline (load, clean, merge)
python scripts/download_data.py

# You'll see:
# ✓ Loaded: 12 datasets
# ✓ Filtered ages 12-19
# ✓ Merged dataset saved
```

Expected output:
```
Final dataset: 1,200-2,000 rows × 150+ columns
Age range: 12-19 years
Ready for feature engineering!
```

---

## Questions?

- **Can't find a dataset?** Use the search function at the top of NHANES website
- **Wrong cycle?** Make sure you selected "2015-2016" in the dropdown
- **File won't download?** Try a different browser (Chrome works best)

**Need help?** The NHANES website can be confusing - let me know which dataset you're stuck on!
