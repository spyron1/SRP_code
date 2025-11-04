# Satellite SRP Hybrid Modeling Project

Simple end‑to‑end pipeline to:
1. Download TLE data
2. Compute geometric & physical SRP features
3. Generate Cannonball and ML SRP accelerations
4. Run orbit propagation validation (SGP4 vs Custom + SRP vs Custom + ML SRP)

## 1. Install

```bash
git clone https://github.com/spyron1/SRP_code.git


```
 create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.\.venv\Scripts\activate       # Windows
pip install -r requirements.txt
cd  validation
streamlit run streamlit_orbit_app.py
```

## 2. Space-Track Credentials

Create `.env` in project root:
```env
SPACETRACK_USERNAME=your_username
SPACETRACK_PASSWORD=your_password
```

## 3. Directory Overview

```
data_acquisition/        # TLE download client
preprocessing/           # Position + shadow + SRP features + SRP acceleration
feature_engineering/     # consoliddated data for traning
model/                   # ML training artifacts
validation/              # Comparison workflow (SGP4 vs Cannonball vs ML SRP)

```





