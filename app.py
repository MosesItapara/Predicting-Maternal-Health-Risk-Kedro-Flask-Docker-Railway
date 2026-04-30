# app.py

from flask import Flask, request, render_template
from predicting_maternal_health_risk.serving.predict import (
    FEATURES,
    predict_risk,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probs = None
    form_data = {}

    if request.method == "POST":
        # Collect raw form values
        for name in FEATURES.keys():
            form_data[name] = request.form.get(name, "")

        try:
            result = predict_risk(form_data)
            prediction = result["risk_label"]
            probs = result["risk_probs"]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        probs=probs,
        form_data=form_data,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)