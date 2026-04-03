# Release Checklist

Use this checklist before tagging or publishing a new version of `Robot Vision Data Studio`.

## Code

- confirm the GUI starts with `python -m app.gui`
- run `python -m compileall app scripts`
- run CLI help checks for export, train, evaluate, and test scripts
- confirm `.gitignore` still excludes generated data and local environments

## Data Workflow

- collect a small real webcam sample
- review at least one sample in `Review / Label`
- export a clean dataset with `--clear-out`
- verify corrected labels are used during export

## Training Workflow

- confirm `configs/dataset.yaml` points to the intended dataset
- train with the selected base model
- validate `best.pt`
- test the trained model on live webcam or recorded video

## Documentation

- update `README.md` if setup or workflow changed
- update Thai and English docs if behavior changed
- document any new model files or deployment notes

## GitHub

- confirm GitHub Actions smoke workflow passes
- review open issues or pull requests
- add release notes summarizing the main changes
- tag the version only after verification is complete
