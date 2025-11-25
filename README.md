# Oncology IPD Assistant 🧬

A "Human-in-the-Loop" AI system for reconstructing Individual Patient Data (IPD) from Kaplan-Meier survival curves in medical literature.

## 🚀 Features
1. **Ingest:** Auto-detection of graph axes using Computer Vision.
2. **Extract:** Hybrid digitization (Auto-Cluster, Seeded-Click, or Manual-Trace).
3. **Reconstruct:** Statistical engine (R parity) to regenerate IPD.
4. **Analyze:** Instant Cox Proportional Hazards & Median Survival calculation.
5. **Reason:** Integration with Google Gemini to interpret clinical results.

## 🛠️ Installation

**Prerequisites:**
- Python 3.10 or 3.11
- R (Language) installed and added to PATH
- R Package `IPDfromKM` installed (`install.packages("IPDfromKM")`)

**Setup:**
```bash
git clone https://github.com/YOUR_USERNAME/Oncology-IPD-Assistant.git
cd Oncology-IPD-Assistant
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🤖 Usage

```bash
export GOOGLE_API_KEY="your_key"
streamlit run app.py
```

## 📁 Project Structure
```
Oncology-IPD-Assistant/
├── app.py                 # Main entry point
├── pages/                 # Streamlit multipage UI
├── src/core/              # Core logic modules
├── data/examples/         # Sample images
├── requirements.txt
└── README.md
```
