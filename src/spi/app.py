import logging
from typing import Mapping

from flask import Flask, jsonify, render_template, request

from spi.pipeline.predict_pipeline import CustomData, PredictPipeline
from spi.logger import configure_logging


logger = logging.getLogger(__name__)


def _predict_from_data(data: Mapping[str, object]) -> float:
    """Validate request values and return one math-score prediction."""
    required_fields = (
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "reading_score",
        "writing_score",
    )
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    student = CustomData(
        gender=str(data["gender"]),
        race_ethnicity=str(data["race_ethnicity"]),
        parental_level_of_education=str(data["parental_level_of_education"]),
        lunch=str(data["lunch"]),
        test_preparation_course=str(data["test_preparation_course"]),
        reading_score=float(data["reading_score"]),
        writing_score=float(data["writing_score"]),
    )
    prediction = PredictPipeline().predict(student.get_data_as_data_frame())
    return round(float(prediction[0]), 2)


def create_app() -> Flask:
    configure_logging()
    app = Flask(__name__)

    @app.get("/")
    def home():
        return render_template("index.html", prediction=None, form_data={})

    @app.post("/predict")
    def predict_from_form():
        form_data = request.form.to_dict()
        try:
            prediction = _predict_from_data(form_data)
            return render_template(
                "index.html", prediction=prediction, form_data=form_data, error=None
            )
        except (ValueError, TypeError) as error:
            return render_template(
                "index.html", prediction=None, form_data=form_data, error=str(error)
            ), 400
        except Exception:
            logger.exception("Prediction failed")
            return render_template(
                "index.html",
                prediction=None,
                form_data=form_data,
                error="Prediction could not be completed. Check that model artifacts exist.",
            ), 500

    @app.post("/api/predict")
    def predict_from_api():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Send a JSON object in the request body."), 400
        try:
            return jsonify(predicted_math_score=_predict_from_data(data))
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error)), 400
        except Exception:
            logger.exception("API prediction failed")
            return jsonify(error="Prediction could not be completed."), 500

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    return app
