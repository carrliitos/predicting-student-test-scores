# Predicting Student Test Scores
> Playground Series - Season 6 Episode 1

Link: [Predicting Student Test Scores](https://www.kaggle.com/competitions/playground-series-s6e1/overview)

## Overview

**Welcome to the 2026 Kaggle Playground Series!** We plan to continue in the spirit of previous playgrounds, providing 
interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a 
competition each month.

**Your Goal**: Predict students' test scores.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
````

## Structure

* `src/` - project code
  * `main.py` - entry point
  * `scripts/` - runnable scripts (e.g., `extraction.py`)
  * `utils/` - shared helpers (`context.py`, `logger.py`, `query.py`)
* `src/data/` - generated datasets (**gitignored**)
* `src/data-raw/` - raw inputs (**gitignored**)
* `src/logs/` - logs (**gitignored**)
* `src/notebooks/` - notebooks
* `requirements.txt` - dependencies
