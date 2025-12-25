# Quick Start: Auto-Generated Report Feature

## What Was Implemented

A complete auto-generated final report system that captures all AutoML pipeline data and generates professional reports in multiple formats.

## 6 New/Modified Files

### Services (2 new)
1. **`app/services/issues.py`** - Issue detection service
2. **`app/services/report.py`** - Report generation service

### Routers (3 modified)
3. **`app/routers/preprocess_router.py`** - Added metadata persistence
4. **`app/routers/models_router.py`** - Added model results persistence  
5. **`app/routers/report_router.py`** - Added 3 new endpoints

### Documentation (1 new)
6. **`REPORT_IMPLEMENTATION.md`** - Complete implementation guide

---

## How It Works

### Automatic Data Collection
```
┌──────────────────────────────────────────┐
│ When you call /api/preprocess/preprocess │
│ ↓                                        │
│ • Detects data quality issues           │
│ • Saves preprocessing_metadata.json     │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ When you call /api/models/models         │
│ ↓                                        │
│ • Trains all selected models             │
│ • Saves model_results.json              │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ When you call /api/report/generate       │
│ ↓                                        │
│ • Loads all metadata                     │
│ • Generates report in requested format   │
│ • Returns file for download              │
└──────────────────────────────────────────┘
```

---

## 3 New API Endpoints

### 1. Generate Report (Download as File)
```
GET /api/report/generate?session_id=xxx&format=markdown
```
- **Formats**: `markdown`, `html`, `pdf`
- **Returns**: Downloadable file
- **Saves to**: `sessions/{session_id}/reports/`

### 2. Preview Report (View Content)
```
GET /api/report/preview?session_id=xxx&format=html
```
- **Returns**: JSON with report content
- **No file download**
- **Good for**: Web preview

### 3. Check Report Status
```
GET /api/report/status?session_id=xxx
```
- **Returns**: Available data & generation capability
- **Tells you**: What metadata is ready

---

## What Data is Stored

For each session, these JSON files are automatically created:

### `preprocessing_metadata.json`
- Before/after dataset statistics
- All preprocessing parameters used
- Detected issues (missing values, outliers, duplicates, etc.)
- Rows removed, features modified
- Recommendations for fixes

### `model_results.json`
- All model performance metrics
- Training time per model
- Hyperparameters (if tuned)
- Best model selection with reasoning

---

## Report Contains 7 Sections

1. **Dataset Overview** - Rows, columns, data types
2. **EDA Findings** - Statistics for all features
3. **Detected Issues** - Data quality problems & recommendations
4. **Preprocessing Decisions** - Methods applied & impact
5. **Model Configurations** - Hyperparameters & settings
6. **Model Comparison** - Performance table for all models
7. **Best Model Summary** - Selected model with justification

---

## Output Formats

### Markdown
- Simple text format
- Good for: Documentation, version control
- Extension: `.md`

### HTML
- Styled with CSS
- Good for: Email, web preview, sharing
- Extension: `.html`
- Includes: Tables, formatting, responsive design

### PDF
- Professional format
- Good for: Reports, archiving
- Extension: `.pdf`
- Currently: HTML with PDF metadata (use browser print to PDF)

---

## Installation Requirements

No new packages needed! Already using:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `fastapi` - API framework
- `sklearn` - ML metrics

**Optional for better PDF:**
```bash
pip install weasyprint reportlab
```

---

## Example Workflow

```
1. POST /api/dataset/upload → Upload CSV
2. GET /api/eda/analyze?session_id=xxx → Run EDA
3. POST /api/preprocess/preprocess → Preprocess data
   ↓ Automatically saves preprocessing_metadata.json
4. GET /api/models/models → Train models
   ↓ Automatically saves model_results.json
5. GET /api/report/generate?session_id=xxx&format=html → Get report
   ↓ Downloads: report_YYYYMMDD_HHMMSS.html
```

---

## What Gets Detected Automatically

### Data Quality Issues
- ✅ Missing values (per column)
- ✅ Duplicate rows
- ✅ Outliers (IQR method)
- ✅ Type inconsistencies
- ✅ Class imbalance (in target)
- ✅ High cardinality features

### Issue Severity
- 🔴 **High**: > 30% missing, > 10% imbalance
- 🟡 **Medium**: 10-30% missing, 3-10x imbalance
- 🟢 **Low**: < 10% missing, < 3x imbalance

---

## Session Data Structure

```
sessions/{session_id}/
├── dataset.csv                    (Original uploaded file)
├── data_cleaned.csv              (After preprocessing)
├── preprocessing_metadata.json    (Auto-saved after preprocess)
├── model_results.json            (Auto-saved after models)
├── plots/                        (EDA visualizations)
├── models/                       (Trained model files)
└── reports/                      (Generated reports)
    ├── report_20251220_103000.md
    ├── report_20251220_104500.html
    └── ...
```

---

## Testing the Feature

### Test 1: Preprocess with metadata
```bash
POST /api/preprocess/preprocess
{
  "session_id": "test-session",
  "missing_strategy": "Mean",
  "outlier_method": "Remove",
  "scaling_method": "Standard",
  "encoding_method": "OneHot",
  "test_size": 0.2,
  "target": "target_column"
}
```
**Result**: `preprocessing_metadata.json` created ✅

### Test 2: Train models with persistence
```bash
GET /api/models/models?session_id=test-session&target=target_column&optimize=false
```
**Result**: `model_results.json` created ✅

### Test 3: Generate markdown report
```bash
GET /api/report/generate?session_id=test-session&format=markdown
```
**Result**: Download `report_YYYYMMDD_HHMMSS.md` ✅

### Test 4: Generate HTML report
```bash
GET /api/report/generate?session_id=test-session&format=html
```
**Result**: Download styled HTML report ✅

### Test 5: Preview report
```bash
GET /api/report/preview?session_id=test-session&format=html
```
**Result**: JSON with HTML content ✅

### Test 6: Check report status
```bash
GET /api/report/status?session_id=test-session
```
**Result**: Shows available data ✅

---

## Troubleshooting

### Report is empty
- ✅ Check that preprocessing was run
- ✅ Check that model training was completed
- ✅ Check session folder has metadata files

### Session not found
- ✅ Verify session_id is correct
- ✅ Check that session folder exists in `/app/static/sessions/`

### Missing sections in report
- ✅ Some sections only appear if data is available
- ✅ Run full pipeline for all 7 sections

---

## Next Steps

1. ✅ Test all 3 endpoints
2. ✅ Verify metadata files are created
3. ✅ Download a sample report
4. ✅ Check report quality and content
5. 🔄 Integrate with frontend
6. 🔄 Add PDF generation (optional)
7. 🔄 Customize report styling (optional)

---

## Questions?

Refer to `REPORT_IMPLEMENTATION.md` for:
- Complete API documentation
- Data structure details
- Implementation architecture
- Future enhancement ideas
