# Student Performance Indicator

This project predicts a student's math score from background and study-related features such as gender, race/ethnicity group, parental education, lunch type, test preparation status, reading score, and writing score.

The application has two parts:

1. A Flask web app with a form-based UI and a JSON prediction API.
2. A machine learning training pipeline that prepares the data, trains multiple regression models, tunes them, and saves the best performing artifacts.

## What the project does

The goal is to estimate the expected math score for a student using information that is available before the exam result is known. This is a supervised regression problem because the target value, `math_score`, is continuous.

The project does the following:

1. Reads the dataset from `notebook/data/stud.csv`.
2. Splits the data into train and test sets.
3. Builds a preprocessing pipeline for numeric and categorical features.
4. Trains and tunes several regression algorithms.
5. Selects the best model using validation performance.
6. Saves the fitted preprocessor and model into `artifacts/`.
7. Serves predictions through a Flask UI and API.

## Why these methods were used

### Data ingestion

`DataIngestion` reads the source CSV and creates reproducible train/test splits. This is necessary so the model can be evaluated on data it did not see during training.

### Data transformation

`DataTransformation` uses a `ColumnTransformer` with two branches:

1. Numerical features are imputed with the median and scaled with `StandardScaler`.
2. Categorical features are imputed with the most frequent value, one-hot encoded, and then scaled with `StandardScaler(with_mean=False)`.

This approach is used because the dataset contains both numeric and categorical inputs. The model expects a purely numeric matrix, so preprocessing converts the raw features into a format that can be learned from consistently.

### Model training

`ModelTrainer` compares multiple regression models:

1. Random Forest Regressor
2. Decision Tree Regressor
3. Gradient Boosting Regressor
4. Linear Regression
5. K-Neighbors Regressor
6. XGB Regressor
7. CatBoost Regressor
8. AdaBoost Regressor

These models were chosen to compare a mix of linear, tree-based, boosting, and neighborhood-based approaches. That makes it easier to find a model that handles the mix of categorical and numerical signal in the dataset.

### Hyperparameter tuning

Each model is tuned with `RandomizedSearchCV` and `KFold` cross-validation. This is used because:

1. It searches a useful subset of each model's hyperparameter space without an expensive full grid search.
2. Cross-validation gives a more stable estimate than a single validation split.
3. The selected model is based on validation R2, not on the held-out test set.

### Model selection

The best tuned model is chosen by cross-validated R2 score and then evaluated on the held-out test set. The final model is only saved if its test R2 is at least 0.6.

## Results

The current training pipeline achieves an R2 score of approximately `0.88` on the held-out test set.

That means the model explains about 88 percent of the variance in the math score target on the test split. In practical terms, the model is capturing a large portion of the patterns present in the dataset.

For the current saved artifacts, the in-process Flask prediction check returns a math score of `66.11` for the default sample input used in validation.

## Project structure

```text
spi/
	artifacts/               Saved training outputs
	notebook/                EDA and training notebooks
	src/spi/
		app.py                 Flask app and routes
		main.py                Application entrypoint
		logger.py              Logging setup
		utils.py               Serialization, tuning, evaluation helpers
		components/
			data_ingestion.py    Raw data loading and split creation
			data_transform.py    Preprocessing pipeline
			model_trainer.py     Model comparison and selection
		pipeline/
			predict_pipeline.py  Artifact loading and prediction helpers
			train_pipeline.py    End-to-end training pipeline
		templates/index.html   Web UI template
```

## How to run the project

### Prerequisites

This project uses Python 3.13 and `uv`.

Make sure you have:

1. Python 3.13 or compatible installed.
2. `uv` installed.
3. A virtual environment synced for the project.

### Install dependencies

From the project root:

```bash
uv sync
```

### Run the Flask app

The app listens on port 5000.

```bash
uv run spi
```

Then open:

```text
http://localhost:5000/
```

If you prefer plain Python, run:

```bash
PYTHONPATH=src python -m spi.main
```

### Run the training pipeline

To rebuild the preprocessing and model artifacts:

```bash
uv run train
```

Or with plain Python:

```bash
PYTHONPATH=src python -m spi.pipeline.train_pipeline
```

## How the app works

### Web UI

`GET /` renders the form in `src/spi/templates/index.html`.

### Form prediction

`POST /predict` accepts form data, validates the inputs, creates a `CustomData` record, loads the saved preprocessor and model, and returns a predicted math score.

### JSON API prediction

`POST /api/predict` accepts a JSON object with the same fields as the form and returns:

```json
{
	"predicted_math_score": 66.11
}
```

### Health check

`GET /health` returns:

```json
{
	"status": "ok"
}
```

## Logging

Logging is initialized when the app starts and when the training pipeline starts. Log files are written to the `logs/` directory with a timestamped filename.

This helps with:

1. Debugging prediction failures.
2. Tracking model training runs.
3. Keeping a record of runtime behavior.

## Reproducing the results

To reproduce the current state from scratch:

1. Clone the repository.
2. Install dependencies with `uv sync`.
3. Run the training pipeline with `uv run train`.
4. Confirm that `artifacts/model.pkl` and `artifacts/preprocessor.pkl` are created.
5. Start the app with `uv run spi`.
6. Open `http://localhost:5000/` and submit a sample prediction.

If the dataset and environment are unchanged, you should see a similar R2 score and a similar prediction for the same sample input.

## Important files

1. [src/spi/main.py](src/spi/main.py)
2. [src/spi/app.py](src/spi/app.py)
3. [src/spi/pipeline/train_pipeline.py](src/spi/pipeline/train_pipeline.py)
4. [src/spi/pipeline/predict_pipeline.py](src/spi/pipeline/predict_pipeline.py)
5. [src/spi/components/data_ingestion.py](src/spi/components/data_ingestion.py)
6. [src/spi/components/data_transform.py](src/spi/components/data_transform.py)
7. [src/spi/components/model_trainer.py](src/spi/components/model_trainer.py)
8. [pyproject.toml](pyproject.toml)

## Conclusion

This project is a complete small machine learning application: it includes data ingestion, preprocessing, model comparison, artifact persistence, logging, and a Flask interface for prediction.

The main strengths of the current implementation are:

1. A repeatable training pipeline.
2. A saved preprocessing and prediction flow that matches training.
3. A working Flask app and API.
4. A strong test-set R2 score of about `0.88`.

## Author

GitHub: [github.com/yash27007](https://github.com/yash27007)

## Contributions

Contributions are welcome. If you have suggestions for improvements or find an issue, feel free to open a pull request or start a discussion.