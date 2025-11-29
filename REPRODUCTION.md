# Reproduction Guide

This document provides instructions for reproducing the evaluation results reported in the dissertation "Oncology IPD Assistant: A Semi-Automated Pipeline for Reconstructing Individual Patient Data from Kaplan–Meier Survival Curves".

---

## 1. Environment Setup

### 1.1 Python Environment

**Requirements:** Python 3.9+

```bash
# Clone repository
git clone https://github.com/yourusername/Oncology-IPD-Assistant.git
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

The AI-powered features (at-risk table extraction, clinical interpretation) require a Google Gemini API key:

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

Test figures are located in `data/test_figures/` (if included) or can be obtained from the original publications.

### 3.2 Step-by-Step Reproduction

For each trial:

1. **Step 1 (Ingest):** Upload the KM curve image and crop to isolate the plot area
2. **Step 2 (Extract):** Use the appropriate extraction method:
   - RELATIVITY-047: Seeded extraction (clear color separation)
   - DESTINY-Breast04: Manual tracing (overlapping curves)
   - CheckMate 816: Seeded extraction
3. **Step 3 (At-Risk):** Enter the number-at-risk table from the figure
4. **Step 4 (Validate):** Select reference arm and run reconstruction
5. **Step 5 (Analyse):** Review statistical outputs

### 3.3 Expected Results

#### RELATIVITY-047 (Melanoma, PFS)

| Metric          | Arm 1 (Rela-Nivo) | Arm 2 (Nivo) |
| --------------- | ----------------- | ------------ |
| Reconstructed n | 355               | 359          |
| MAE             | < 2%              | < 2%         |

| Statistic    | Reconstructed | Published |
| ------------ | ------------- | --------- |
| Hazard Ratio | ~0.75         | 0.75      |
| 95% CI       | ~0.62-0.92    | 0.62-0.92 |
| p-value      | < 0.01        | 0.0055    |

#### DESTINY-Breast04 (Breast Cancer, OS)

| Metric          | Arm 1 (T-DXd) | Arm 2 (Physician's Choice) |
| --------------- | ------------- | -------------------------- |
| Reconstructed n | 373           | 184                        |
| MAE             | < 2%          | < 2%                       |

| Statistic    | Reconstructed | Published |
| ------------ | ------------- | --------- |
| Hazard Ratio | ~0.64         | 0.64      |
| 95% CI       | ~0.49-0.84    | 0.49-0.84 |
| p-value      | < 0.01        | 0.001     |

#### CheckMate 816 (Lung Cancer, EFS)

| Metric          | Arm 1 (Nivo+Chemo) | Arm 2 (Chemo) |
| --------------- | ------------------ | ------------- |
| Reconstructed n | 179                | 179           |
| MAE             | < 2%               | < 2%          |

| Statistic    | Reconstructed | Published |
| ------------ | ------------- | --------- |
| Hazard Ratio | ~0.63         | 0.63      |
| 95% CI       | ~0.43-0.91    | 0.43-0.91 |
| p-value      | < 0.01        | 0.005     |

### 3.4 Saved Trial Data

Pre-extracted trial data (for verification) is stored in `data/projects/` as JSON files containing:

- Digitised curve coordinates
- At-risk tables
- Reconstructed IPD
- Statistical outputs

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
│       ├── extractor.py        # Auto curve extraction (color-based)
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
    ├── projects/               # Saved analyses (JSON)
    └── test_figures/           # Evaluation figures (if included)
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
HR = hazard(comparator) / hazard(reference)
```

The reference arm is user-selectable in Step 4 to ensure HR direction matches published convention.

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

---

## 8. Citation

If reproducing this work, please cite:

```
Georghiou, A. (2025). Oncology IPD Assistant: A Semi-Automated Pipeline for
Reconstructing Individual Patient Data from Kaplan–Meier Survival Curves.
MSc Data Science Dissertation, [University Name].
```

And the underlying algorithm:

```
Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival
data: reconstructing the data from published Kaplan-Meier survival curves.
BMC Med Res Methodol. 2012;12:9.
```
