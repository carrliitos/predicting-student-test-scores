# Project/Competition Name

Template repo for Kaggle projects.

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
