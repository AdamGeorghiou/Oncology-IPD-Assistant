# Oncology IPD Assistant

A semi-automated pipeline for reconstructing Individual Patient Data (IPD) from published Kaplan-Meier survival curves.

## Overview

Researchers conducting meta-analyses often only have access to published survival curves, not raw patient data. This tool bridges that gap by implementing the Guyot algorithm to reconstruct IPD from digitised Kaplan-Meier curves.

## Features

### 5-Step Pipeline

1. **Ingest & Crop** - Upload KM curve image, auto-detect or manually crop plot area
2. **Curve Extraction** - Extract survival curves via auto (color detection), seeded (click-based), or manual tracing
3. **At-Risk Table** - Enter number-at-risk data via manual input, OCR, or Vision LLM extraction
4. **Validation** - Reconstruct IPD using Guyot algorithm, validate against digitised curves (MAE), select reference arm for HR calculation
5. **Analysis** - Cox PH, Log-rank test, median survival, forest plot, AI-powered interpretation

### Additional Features

- **Knowledge Base** - Save and load completed analyses
- **Multi-Trial Comparison** - Compare multiple trials with AI assistant
- **Export** - Download reconstructed IPD as CSV, analysis reports as TXT
- **AI Chat** - Ask clinical questions about your results (Gemini-powered)
- **Reference Arm Selection** - Choose control arm to ensure HR matches published convention

## Tech Stack

- **Frontend**: Streamlit
- **Curve Extraction**: OpenCV, NumPy, scikit-learn
- **OCR**: EasyOCR, Google Gemini Vision
- **Statistics**: lifelines, scipy
- **IPD Reconstruction**: IPDfromKM (R package via rpy2)
- **LLM**: Google Gemini API

## Installation

### Prerequisites

- Python 3.9+
- R 4.0+ with `IPDfromKM` package installed

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Oncology-IPD-Assistant.git
cd Oncology-IPD-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install R package (in R console)
# install.packages("IPDfromKM")

# Set up environment variables
export GOOGLE_API_KEY="your-gemini-api-key"

# Run the app
streamlit run app.py
```

## Usage

1. Upload a Kaplan-Meier curve image (PNG/JPG)
2. Crop to isolate the plot area
3. Extract survival curves using your preferred method
4. Enter at-risk numbers (from table below the curve)
5. Select reference (control) arm for correct HR direction
6. Run reconstruction and validate accuracy
7. Analyse results and export IPD

## Project Structure

```
Oncology-IPD-Assistant/
├── app.py                 # Main Streamlit application
├── src/
│   └── core/
│       ├── vision.py          # Axis detection and cropping
│       ├── extractor.py       # Auto curve extraction
│       ├── seeded_extractor.py    # Click-based extraction
│       ├── tracer.py          # Manual curve tracing
│       ├── ocr_table.py       # At-risk table OCR + LLM
│       ├── reconstructor.py   # Guyot algorithm wrapper
│       ├── analysis.py        # Statistical analysis (Cox PH, Log-rank)
│       ├── llm_bridge.py      # Gemini API integration
│       └── project_io.py      # Save/load functionality
├── pages/
│   └── LLM_Analysis.py    # Multi-trial comparison page
├── data/
│   ├── uploads/           # Uploaded images
│   └── projects/          # Saved analyses
└── requirements.txt
```

## Methods

### IPD Reconstruction

This tool implements the algorithm described in:

> Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. BMC Med Res Methodol. 2012;12:9.

### Validation

Reconstructed curves are validated against digitised input by calculating Mean Absolute Error (MAE):

- MAE < 2%: High confidence
- MAE 2-5%: Moderate confidence
- MAE > 5%: Poor fit (check inputs)

### Statistical Analysis

- **Cox Proportional Hazards**: Hazard ratio with 95% CI
- **Log-rank Test**: Non-parametric comparison of survival curves
- **Median Survival**: Kaplan-Meier estimated median for each arm

## Limitations

- **Approximation**: Reconstructed IPD is an estimate, not original trial data
- **Input quality**: Accuracy depends on digitisation quality and at-risk numbers
- **Assumptions**: Guyot algorithm assumes constant censoring within intervals
- **Two-arm comparison**: Statistical analysis currently supports pairwise comparisons
- **Curve coverage**: At-risk data is auto-trimmed to match curve time range

## Reproducing Results

See [REPRODUCTION.md](REPRODUCTION.md) for detailed instructions.

## Citation

If you use this tool in your research:

```
Data were reconstructed from published Kaplan-Meier curves using the Guyot algorithm
(Guyot et al., 2012) implemented via the IPDfromKM R package. Digitisation and analysis
performed using Oncology IPD Assistant.

Reference:
Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data:
reconstructing the data from published Kaplan-Meier survival curves.
BMC Med Res Methodol. 2012;12:9.
```

## License

MIT License

## Author

Adam Georghiou - MSc Data Science Dissertation Project
