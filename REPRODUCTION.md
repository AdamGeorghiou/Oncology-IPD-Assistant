# Reproduction Guide

This document provides instructions for reproducing the evaluation results reported in the dissertation "Oncology IPD Assistant: A Semi-Automated Pipeline for Reconstructing Individual Patient Data from Kaplan–Meier Survival Curves".

---

## 1. Environment Setup

### 1.1 Python Environment

**Requirements:** Python 3.11+ (earlier versions not supported)

```bash
# Clone repository
git clone https://github.com/AdamGeorghiou/Oncology-IPD-Assistant.git
cd Oncology-IPD-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.2 R Environment

**Requirements:** R 4.0+

The IPD reconstruction uses the `IPDfromKM` R package via `rpy2`. Install in R:

```R
install.packages("IPDfromKM")
```

Verify installation:

```R
library(IPDfromKM)
```

### 1.3 API Configuration (Optional)

The AI-powered features (Vision LLM at-risk table extraction, clinical interpretation) require a Google Gemini API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Alternatively, create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

**Note:** The core IPD reconstruction pipeline works without an API key. Only the Vision LLM features require it.

---

## 2. Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 3. Reproducing Evaluation Results

### 3.1 Test Figures

The evaluation used Kaplan-Meier figures from three published Phase III oncology trials:

| Trial            | Source                  | Figure          |
| ---------------- | ----------------------- | --------------- |
| RELATIVITY-047   | Tawbi et al., NEJM 2022 | Figure 2A (PFS) |
| DESTINY-Breast04 | Modi et al., NEJM 2022  | Figure 2A (OS)  |
| CheckMate 816    | Forde et al., NEJM 2022 | Figure 2A (EFS) |

Test figures can be obtained from the original publications via the New England Journal of Medicine.

### 3.2 Step-by-Step Reproduction

For each trial:

1. **Step 1 (Ingest):** Upload the KM curve image and crop to isolate the plot area
2. **Step 2 (Extract):** Use the appropriate extraction method:
   - RELATIVITY-047: Automatic extraction (clear colour separation)
   - DESTINY-Breast04: Manual tracing (low contrast, similar colours)
   - CheckMate 816: Manual tracing (partial auto extraction possible for orange arm)
3. **Step 3 (At-Risk):** Enter the number-at-risk table using Vision LLM extraction (recommended) or manual entry
4. **Step 4 (Validate):** Select reference arm and run reconstruction
5. **Step 5 (Analyse):** Review statistical outputs

### 3.3 Expected Results

#### Curve Reconstruction Accuracy (MAE)

| Trial            | Arm                      | MAE (%) | Accuracy |
| ---------------- | ------------------------ | ------- | -------- |
| RELATIVITY-047   | Relatlimab–nivolumab     | 0.29    | High     |
| RELATIVITY-047   | Nivolumab                | 0.16    | High     |
| DESTINY-Breast04 | Trastuzumab deruxtecan   | 0.16    | High     |
| DESTINY-Breast04 | Physician's choice       | 0.22    | High     |
| CheckMate 816    | Nivolumab + chemotherapy | 0.36    | High     |
| CheckMate 816    | Chemotherapy alone       | 0.22    | High     |

All arms achieved MAE well below the 2% threshold (mean 0.24%).

#### Clinical Statistics Agreement

| Trial            | Statistic            | Reconstructed | Published | Agreement           |
| ---------------- | -------------------- | ------------- | --------- | ------------------- |
| RELATIVITY-047   | Hazard Ratio         | 0.75          | 0.75      | Exact match         |
|                  | 95% CI               | 0.59–0.94     | 0.62–0.92 | –                   |
|                  | Cox p-value          | 0.012         | 0.006     | Both significant    |
|                  | Median (Rela-nivo)   | 10.0 mo       | 10.1 mo   | Within tolerance    |
|                  | Median (Nivo)        | 5.1 mo        | 4.6 mo    | Marginal (Δ 0.5 mo) |
| DESTINY-Breast04 | Hazard Ratio         | 0.66          | 0.64      | Within tolerance    |
|                  | 95% CI               | 0.52–0.84     | 0.49–0.84 | –                   |
|                  | Cox p-value          | 0.0007        | 0.001     | Both significant    |
|                  | Median (T-DXd)       | 19.3 mo       | 23.4 mo   | Δ 4.1 mo            |
|                  | Median (Phys Choice) | 14.5 mo       | 16.8 mo   | Δ 2.3 mo            |
| CheckMate 816    | Hazard Ratio         | 0.59          | 0.57      | Within tolerance    |
|                  | 95% CI\*             | 0.39–0.88     | 0.30–1.07 | –                   |
|                  | Cox p-value          | 0.009         | 0.008     | Both significant    |
|                  | Median (Both)        | NR            | NR        | Match               |

\*Published CI was 99.67% per interim analysis plan; reconstructed CI is 95%.

#### At-Risk Table Extraction Accuracy

| Trial            | Enhanced OCR | Vision LLM |
| ---------------- | ------------ | ---------- |
| RELATIVITY-047   | 95.5%        | 100%       |
| DESTINY-Breast04 | ~71%         | 100%       |
| CheckMate 816    | 97%          | 100%       |
| **Overall**      | **~83%**     | **100%**   |

### 3.4 Saved Trial Data

Pre-extracted trial data (for verification) is stored in `data/projects/` as JSON files containing:

- Digitised curve coordinates
- At-risk tables
- Reconstructed IPD
- Statistical outputs

These can be loaded via the Knowledge Base sidebar in the application.

---

## 4. Project Structure

```
Oncology-IPD-Assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── REPRODUCTION.md             # This file
├── src/
│   └── core/
│       ├── vision.py           # Axis detection and cropping
│       ├── extractor.py        # Auto curve extraction (colour-based)
│       ├── seeded_extractor.py # Click-based curve extraction
│       ├── tracer.py           # Manual curve tracing
│       ├── ocr_table.py        # At-risk table OCR + Vision LLM
│       ├── reconstructor.py    # Guyot algorithm (via IPDfromKM)
│       ├── analysis.py         # Cox PH, Log-rank, KM estimation
│       ├── llm_bridge.py       # Gemini API integration
│       ├── project_io.py       # Save/load functionality
│       └── table_manager.py    # At-risk table UI management
├── pages/
│   └── 🤖_LLM_Analysis.py      # Multi-trial comparison page
└── data/
    ├── uploads/                # Uploaded images (temporary)
    └── projects/               # Saved analyses (JSON)
```

---

## 5. Key Dependencies

| Package                   | Version | Purpose                          |
| ------------------------- | ------- | -------------------------------- |
| streamlit                 | ≥1.28   | Web application framework        |
| opencv-python             | ≥4.8    | Image processing, axis detection |
| numpy                     | ≥1.24   | Numerical operations             |
| pandas                    | ≥2.0    | Data manipulation                |
| scikit-learn              | ≥1.3    | Clustering for curve extraction  |
| lifelines                 | ≥0.27   | Kaplan-Meier estimation, Cox PH  |
| scipy                     | ≥1.11   | Interpolation, statistics        |
| rpy2                      | ≥3.5    | R integration                    |
| easyocr                   | ≥1.7    | OCR for at-risk tables           |
| google-generativeai       | ≥0.3    | Gemini API (optional)            |
| streamlit-drawable-canvas | ≥0.9    | Manual curve tracing             |
| matplotlib                | ≥3.7    | Plotting                         |

Full dependency list with pinned versions in `requirements.txt`.

---

## 6. Methodological Notes

### 6.1 IPD Reconstruction Algorithm

The system implements the algorithm described in:

> Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. BMC Med Res Methodol. 2012;12:9.

Key assumptions:

- Constant censoring rate within intervals
- Monotonically decreasing survival function
- At-risk numbers are accurate

### 6.2 Validation Approach

Reconstruction accuracy is validated by:

1. Recalculating Kaplan-Meier curves from reconstructed IPD
2. Comparing to original digitised coordinates
3. Computing Mean Absolute Error (MAE)

Quality thresholds:

- MAE < 2%: High confidence
- MAE 2-5%: Moderate confidence
- MAE > 5%: Poor fit (review inputs)

### 6.3 Hazard Ratio Calculation

HR is calculated using Cox Proportional Hazards regression:

```
HR = hazard(experimental) / hazard(reference)
```

The reference arm is user-selectable in Step 4 to ensure HR direction matches published convention (HR < 1 indicates benefit for experimental arm).

---

## 7. Troubleshooting

### R Package Not Found

```
RuntimeError: Failed to load IPDfromKM R package
```

**Solution:** Ensure R is installed and `IPDfromKM` package is available:

```R
install.packages("IPDfromKM")
```

### rpy2 Context Error

```
Conversion rules for `rpy2.robjects` appear to be missing
```

**Solution:** Restart the Streamlit application. This is a threading issue that resolves on restart.

### API Quota Exceeded

```
⚠️ Quota limit reached
```

**Solution:** Wait 60 seconds (rate limit) or add billing to your Google Cloud project.

### Python Version Errors

```
SyntaxError or ImportError on startup
```

**Solution:** Ensure you are using Python 3.11 or higher. Earlier versions (3.9, 3.10) may cause compatibility errors.

---

## 8. Citation

If reproducing this work, please cite:

```
Georghiou, A. (2025). Oncology IPD Assistant: A Semi-Automated Pipeline for
Reconstructing Individual Patient Data from Kaplan–Meier Survival Curves.
MSc Data Science Dissertation, University of Manchester.
```

And the underlying algorithm:

```
Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival
data: reconstructing the data from published Kaplan-Meier survival curves.
BMC Med Res Methodol. 2012;12:9.
```
