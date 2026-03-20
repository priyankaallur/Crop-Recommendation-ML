from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# ── Load models ────────────────────────────────────────────────────────────────
with open("model/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/all_models.pkl", "rb") as f:
    all_models = pickle.load(f)   # {"Random Forest": (model, acc), ...}

# ── Input validation ranges ────────────────────────────────────────────────────
VALID_RANGES = {
    "N":           (0,    140,  "Nitrogen",     "mg/kg"),
    "P":           (5,    145,  "Phosphorus",   "mg/kg"),
    "K":           (5,    205,  "Potassium",    "mg/kg"),
    "temperature": (8.0,  44.0, "Temperature",  "°C"),
    "humidity":    (14.0, 100.0,"Humidity",     "%"),
    "ph":          (3.5,  9.5,  "pH",           ""),
    "rainfall":    (20.0, 300.0,"Rainfall",     "mm"),
}

def validate_inputs(data):
    """Validate and parse inputs. Returns (values_dict, errors_list)."""
    errors = []
    values = {}
    for key, (mn, mx, label, unit) in VALID_RANGES.items():
        raw = data.get(key, "")
        try:
            val = float(raw)
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid number.")
            continue
        if not (mn <= val <= mx):
            u = f" {unit}" if unit else ""
            errors.append(f"{label} must be between {mn} and {mx}{u}.")
        else:
            values[key] = val
    return values, errors

def predict_with_proba(m, features):
    """Return (crop_name, confidence_pct_or_None)."""
    crop = m.predict(features)[0]
    proba = None
    if hasattr(m, "predict_proba"):
        probs = m.predict_proba(features)[0]
        proba = round(float(max(probs)) * 100, 1)
    return crop, proba

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    error = None
    input_data = {}

    if request.method == "POST":
        values, errors = validate_inputs(request.form)
        input_data = dict(request.form)          # keep form values for repopulation
        if errors:
            error = errors
        else:
            features = [[values["N"], values["P"], values["K"],
                         values["temperature"], values["humidity"],
                         values["ph"], values["rainfall"]]]
            result, confidence = predict_with_proba(model, features)

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           error=error,
                           input_data=input_data,
                           ranges=VALID_RANGES)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    results = None
    error = None
    input_data = {}

    if request.method == "POST":
        values, errors = validate_inputs(request.form)
        input_data = dict(request.form)
        if errors:
            error = errors
        else:
            features = [[values["N"], values["P"], values["K"],
                         values["temperature"], values["humidity"],
                         values["ph"], values["rainfall"]]]
            results = {}
            for name, (m, acc) in all_models.items():
                crop, proba = predict_with_proba(m, features)
                results[name] = {
                    "crop":       crop,
                    "accuracy":   round(acc * 100, 2),
                    "confidence": proba,
                }

    best_name = max(results, key=lambda n: results[n]["accuracy"]) if results else None

    return render_template("compare.html",
                           results=results,
                           best_name=best_name,
                           error=error,
                           input_data=input_data,
                           ranges=VALID_RANGES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint — accepts JSON, returns JSON.

    Request body (JSON):
        { "N": 90, "P": 42, "K": 43, "temperature": 20,
          "humidity": 82, "ph": 6.5, "rainfall": 220 }

    Response (JSON):
        { "recommended_crop": "rice", "confidence": 97.3, "inputs": {...} }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided. Send Content-Type: application/json"}), 400

        values, errors = validate_inputs(data)
        if errors:
            return jsonify({"error": errors}), 422

        features = [[values["N"], values["P"], values["K"],
                     values["temperature"], values["humidity"],
                     values["ph"], values["rainfall"]]]

        crop, proba = predict_with_proba(model, features)

        return jsonify({
            "recommended_crop": crop,
            "confidence":       proba,
            "inputs":           values
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
