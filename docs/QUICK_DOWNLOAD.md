# Quick Download Guide - All 12 NHANES Datasets

## Direct Download Links (FASTEST!)

**Core Datasets (6 files) - REQUIRED**

Right-click each link → Save As... → Save to `data\raw\`

1. [DEMO_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt) - Demographics (~3.5 MB)
2. [GLU_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.xpt) - Glucose (~500 KB)
3. [INS_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/INS_I.xpt) - Insulin (~500 KB)
4. [HSCRP_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HSCRP_I.xpt) - CRP (~500 KB)
5. [PAQ_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.xpt) - Physical Activity (~2 MB)
6. [DR1TOT_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1TOT_I.xpt) - Dietary (~4 MB)

---

**Enhanced Datasets (6 files) - OPTIONAL**

7. [GHB_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.xpt) - HbA1c (~500 KB)
8. [TRIGLY_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/TRIGLY_I.xpt) - Lipids (~500 KB)
9. [BPX_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.xpt) - Blood Pressure (~2 MB)
10. [BMX_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.xpt) - Body Measures (~2 MB)
11. [MCQ_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt) - Medical Conditions (~3 MB)
12. [SLQ_I.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.xpt) - Sleep (~1 MB)

---

## Download Instructions

### Option 1: Right-Click Method (Easiest)
1. Right-click each link above
2. Choose "Save Link As..." or "Save Target As..."
3. Navigate to: `C:\Users\Myra Saxena\Documents\SourceCode\ScienceFair\data\raw\`
4. Make sure filename keeps `.xpt` extension
5. Click Save

### Option 2: Click & Download
1. Click each link
2. Browser will automatically download
3. Move files from Downloads folder to `data\raw\`

### Option 3: Automated (Try Now!)
```bash
cd "C:\Users\Myra Saxena\Documents\SourceCode\ScienceFair"
venv\Scripts\activate.bat
python scripts\download_data.py
```

The automated script should now work with the corrected URLs!

---

## Verify Downloads

After downloading, check you have all files:

```bash
cd data\raw
dir *.xpt
```

You should see:
- DEMO_I.XPT (~3.5 MB)
- GLU_I.XPT (~500 KB)
- INS_I.XPT (~500 KB)
- HSCRP_I.XPT (~500 KB)
- PAQ_I.XPT (~2 MB)
- DR1TOT_I.XPT (~4 MB)
- GHB_I.XPT (~500 KB)
- TRIGLY_I.XPT (~500 KB)
- BPX_I.XPT (~2 MB)
- BMX_I.XPT (~2 MB)
- MCQ_I.XPT (~3 MB)
- SLQ_I.XPT (~1 MB)

**Total:** ~20 MB

---

## Next Steps

Once all files are downloaded:

```bash
# Load, clean, and merge data
python scripts\download_data.py

# Expected output:
# - Loaded: 12 datasets
# - Filtered ages 12-19
# - Merged dataset saved to data/processed/merged_data.csv
```

---

## Troubleshooting

**File won't download?**
- Try a different browser (Chrome works best)
- Check internet connection
- Make sure `data\raw\` folder exists

**Downloaded file is tiny (~20 KB)?**
- This is an HTML error page, not the data
- Use the direct links above (not the search interface)

**Can't find raw folder?**
- Create it: `mkdir data\raw`
- Or run: `python -c "from config.constants import ensure_directories; ensure_directories()"`
