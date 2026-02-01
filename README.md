# Heart Disease Prediction Project

A small Flask web app for predicting heart disease risk from patient features.

**What it is**
- **Project:** Heart Disease Prediction (final year project)
- **UI:** Simple form at the web root that posts patient values to the model
- **Templates:** [templates/index.html](templates/index.html)

**Repository layout**
- `app.py`, `main.py`: application entrypoints
- `templates/`: HTML templates (including the form)
- `static/`: CSS and static assets
- `data/`: datasets used (`heart.csv`, `dataset.csv`)
- `src/`: data loading, preprocessing, evaluation and model code
  - `src/models/`: model implementations

Prerequisites
- Python 3.8+ recommended
- Git (optional)

Quick setup
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install common dependencies (add a `requirements.txt` if you want):

```bash
pip install flask pandas numpy scikit-learn joblib
```

Run the app

```bash
python3 main.py
# or
python3 app.py
```

Open http://127.0.0.1:5000/ in your browser and use the form on the home page to submit patient values.

Notes
- Form fields are defined in [templates/index.html](templates/index.html). The app returns a prediction message and an optional probability value when available.
- Datasets live in [data/](data/). If you retrain models, put output models under `src/models/` or update loading code in `src/` accordingly.

If you want me to:
- add a `requirements.txt` or `venv`-friendly setup, or
- add a script to train and persist a model from `data/`,
tell me and I will add it.


