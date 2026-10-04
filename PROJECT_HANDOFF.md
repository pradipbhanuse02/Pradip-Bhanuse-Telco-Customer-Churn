# Telco Customer Churn Project Handoff

## Summary

The project now has reusable preprocessing, model, and CSV-inference modules. The preprocessing and modeling notebooks were updated to use those modules. Preprocessing is fitted as part of the model pipeline, so encoders and imputers learn from training data only.

## Work Completed

- Added [src/preprocessing.py](src/preprocessing.py) for CSV loading, target/feature separation, churn label encoding, stratified splitting, feature engineering, imputation, scaling, and categorical encoding.
- Added [src/model.py](src/model.py) for building, training, and evaluating Logistic Regression, KNN, SVM, Decision Tree, and Random Forest pipelines.
- Added [src/utils.py](src/utils.py) for predicting churn labels and probabilities from a separate CSV.
- Added [tests/test_pipeline.py](tests/test_pipeline.py) with four unit tests covering target encoding, stratified splits, prediction on a separate CSV with unseen categories, and evaluation metrics.
- Added `pytest` to [requirements.txt](requirements.txt).
- Updated [ipynb file/03_preprocessing.ipynb](ipynb%20file/03_preprocessing.ipynb) and [ipynb file/04_model.ipynb](ipynb%20file/04_model.ipynb) to use the source modules.
- Documented each function in the new source and test files with a docstring.

## Issues and Resolutions

| Step | Issue encountered | Resolution |
|---|---|---|
| 1. Initial test command | The system `python` command was not on the terminal PATH. | Used the project's interpreter at `myenv\\python.exe`. |
| 2. Test setup | The project environment did not have pytest installed. | Added pytest to `requirements.txt` and installed it in the project environment. |
| 3. Real-data inference | The Telco dataset's `TotalCharges` column was read as text because it contains blank values. Unseen text values then caused the encoder to fail on the real holdout CSV. | Converted `TotalCharges` to numeric with invalid/blank values treated as missing; added median imputation for numeric features and most-frequent imputation for categorical features. Added regression coverage for blank and unseen charge values. |
| 4. Notebook imports | The notebook kernel started in the notebook folder and could not import the root-level `src` package (`ModuleNotFoundError: No module named 'src'`). | Added project-root discovery and inserted the root into `sys.path` before importing source modules. |
| 5. Modeling notebook execution | The notebook execution tool reported that the model-comparison cell did not finish, despite showing a comparison table. The tool did not confirm completion, so that output is not treated as verified. | No fix is claimed. The full modeling notebook run remains unverified; the source pipeline and separate CSV inference were validated independently. |

## Validation Results by Step

| Validation step | Result |
|---|---|
| Initial `python -m pytest -q` | Did not start: system Python was not found. |
| Project interpreter before pytest setup | Did not start: pytest was not installed in the environment. |
| Unit tests after pytest setup | `4 passed in 13.47s`. |
| Unit tests after fixing `TotalCharges` preprocessing | `4 passed in 3.46s`. |
| Real Telco data and separate CSV inference | Trained on 5,634 training rows; predicted 25 rows from a separate unlabeled CSV. Produced `No`/`Yes` labels and probabilities from 0.015 to 0.901. |
| Preprocessing notebook | Ran the data-loading and preprocessing cells successfully: 7,043 rows, 19 raw features, 5,634 training rows, and 1,409 test rows. Stratification preserved both classes. |
| Python compile check | `python -m compileall -q src tests` completed without errors. |
| Final pytest run on current workspace | `4 passed in 3.59s`. |
| Workspace diagnostics | No errors reported for `src/` or `tests/`. |

## Remaining Notes

- The complete modeling notebook cell, including all model comparisons, was not confirmed as finished by the notebook runner. Its full execution and later tuning/SHAP steps should be rerun in the notebook when validating the end-to-end notebook workflow.
- No Git commit was created. The source, tests, requirements, and notebook edits remain in the working tree for review.
