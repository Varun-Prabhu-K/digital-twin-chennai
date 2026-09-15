# AI-Based Prototype Digital Twin for Predictive Smart City Planning and Management
A prototype AI-powered Digital Twin for smart city planning, using selected Chennai urban data.

## Features
- Traffic prediction
- Pollution prediction
- Electricity-demand prediction
- What-if scenario simulation
- Road-closure rerouting
- Population-growth simulation
- New-hospital scenario
- Interactive Streamlit dashboard
- Multi-criteria scenario recommendation

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly

## Run Locally
bash
pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py

Then open:
http://localhost:8501

Project Structure
digital-twin-chennai/
├── dashboard/
├── models/
├── data/
└── engine/

Project Scope
This is an academic prototype. It uses selected urban datasets and a configured subset of Chennai roads and zones rather than attempting to model the entire city in real time.
