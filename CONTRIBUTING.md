# Contributing

Thanks for contributing to `Robot Vision Data Studio`.

## Development Setup

```bash
python scripts/setup_venv.py
source .venv/bin/activate
python -m app.gui
```

On Windows PowerShell:

```powershell
python scripts/setup_venv.py
.venv\Scripts\Activate.ps1
python -m app.gui
```

## Project Conventions

- keep the robot camera position consistent during data collection
- store reusable model weights in `models/`
- keep raw field captures in `data/raw/`
- use the GUI `Review / Label` tab before exporting datasets
- use `scripts/export_yolo_dataset.py --clear-out` when rebuilding a dataset

## Validation

Before opening a pull request, run:

```bash
python -m compileall app scripts
python scripts/export_yolo_dataset.py --help
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/test_model.py --help
```

## Pull Requests

- keep changes focused
- update docs when behavior changes
- avoid committing `.venv`, raw captures, generated datasets, or training outputs

