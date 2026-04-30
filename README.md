# Predicting Maternal Health Risk

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Kedro-based machine learning pipeline for predicting maternal health risk from routinely collected vital signs, paired with a lightweight Flask web interface for entering patient data and reviewing risk estimates.

> ⚠️ **Clinical Disclaimer**  
> This is a **prototype decision support tool** intended to assist, not replace, clinical judgement. Before any real-world use, the model must be validated on local data, integrated into clinical protocols, and reviewed for performance, bias, and safety. Always defer to local guidelines and qualified clinical professionals.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running the Pipelines](#running-the-pipelines)
- [Running the Web App](#running-the-web-app)
- [Running with Docker](#running-with-docker)
- [Deploying to Railway](#deploying-to-railway)
- [Configuration & Data](#configuration--data)
- [Testing](#testing)
- [Notebooks](#notebooks)
- [Limitations](#limitations)

---

## Overview

The project predicts whether a patient is at **Low**, **Medium**, or **High** maternal health risk based on six vital sign inputs:

| Input | Description |
|---|---|
| Age | Patient age in years |
| Systolic BP | Systolic blood pressure (mmHg) |
| Diastolic BP | Diastolic blood pressure (mmHg) |
| Blood Sugar | Blood glucose level |
| Body Temperature | Temperature (°F) |
| Heart Rate | Resting heart rate (bpm) |

An XGBoost classifier is trained via Kedro pipelines and served through a Flask web app with a simple form interface.

---

## Project Structure

```
.
├── src/predicting_maternal_health_risk/
│   ├── pipelines/          # Data processing, training, and evaluation pipelines
│   └── serving/            # Model loading and prediction helpers
├── conf/
│   ├── base/               # Catalog and parameter config
│   └── local/              # Local secrets and overrides (untracked)
├── data/
│   ├── 01_raw/             # Raw input data (do not commit)
│   └── 06_models/          # Saved trained model
├── templates/              # HTML templates for the Flask UI
├── tests/                  # Automated tests
├── app.py                  # Flask web application entry point
├── Dockerfile              # Container image definition
└── pyproject.toml          # Project metadata and dependencies
```

Kedro handles **data engineering, training, and evaluation**. Flask serves the trained model for interactive use.

---

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd predicting-maternal-health-risk

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt  # if present

# 4. Run all Kedro pipelines
kedro run

# 5. Start the Flask app
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Running the Pipelines

Run the full project with:

```bash
kedro run
```

Or run individual stages:

```bash
kedro run --pipeline dp   # Data processing
kedro run --pipeline mt   # Model training
kedro run --pipeline me   # Model evaluation
```

The pipelines load raw data, preprocess features, train an XGBoost model, evaluate performance, and save the trained model to `data/06_models/`.

---

## Running the Web App

Once the model has been trained and saved, start the Flask app:

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000). The **Maternal Health Risk Predictor** form allows you to enter patient vitals and will return:

- The predicted risk category: **Low**, **Medium**, or **High**
- The associated class probabilities

Inputs are preprocessed identically to the training pipeline before inference.

---

## Running with Docker

Build and run the containerised app:

```bash
docker build -t maternal-risk-app .
docker run -p 5000:5000 maternal-risk-app
```

Then visit [http://localhost:5000](http://localhost:5000).

The Docker image installs all dependencies from `pyproject.toml`, copies the Kedro project, trained model, and Flask app into `/app`, and runs `python app.py` as the default command.

---

## Deploying to Railway

This project is ready to deploy as a web service on [Railway](https://railway.app).

### Steps

1. **Push to GitHub** — include `app.py`, `templates/`, `pyproject.toml`, and `Dockerfile`.
2. **Create a Railway project** — select *Deploy from GitHub* and point it at your repository.
3. **Choose a build method**:
   - **Docker** (recommended): Railway will detect and use the `Dockerfile` automatically.
   - **Python buildpack**: Set the start command to `python app.py`.
4. **Configure the port**: Railway injects a `PORT` environment variable. Your `app.py` must respect it:

```python
import os
from flask import Flask
from predicting_maternal_health_risk.serving.predict import FEATURES, predict_risk

app = Flask(__name__)

# ... your routes ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

Once deployed, Railway provides a public URL you can share directly.

---

## Configuration & Data

- **Raw data** must not be committed. Place local datasets under `data/01_raw/` and ensure that path is listed in `.gitignore`.
- **Base configuration** (data catalog, model parameters) lives under `conf/base/`.
- **Local overrides** (credentials, environment-specific paths) go under `conf/local/` — keep these untracked as well.

Kedro's [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention) is followed throughout: `raw → intermediate → primary → models → reporting`.

---

## Testing

Run the test suite with:

```bash
pytest
```

Coverage settings are configured in `pyproject.toml` under `[tool.coverage.report]`.

---

## Notebooks

For interactive exploration of data and models:

```bash
pip install jupyter jupyterlab
kedro jupyter lab
# or
kedro jupyter notebook
```

Launching via `kedro jupyter` pre-loads `catalog`, `context`, `pipelines`, and `session` into the notebook environment. To avoid committing large notebook outputs, consider using [`nbstripout`](https://github.com/kynan/nbstripout).

---

## Limitations

- This is a **research prototype**, not a production-grade clinical tool.
- The model requires **local validation** before being used on any real patient cohort.
- Predictions depend on the quality and representativeness of training data; performance may degrade on populations not reflected in that data.
- Risk thresholds are configurable but must be reviewed by clinical experts before deployment.
- The tool does not replace a full clinical assessment and must be embedded in a governed clinical workflow.

---

For more on Kedro, see the [official documentation](https://docs.kedro.org).
