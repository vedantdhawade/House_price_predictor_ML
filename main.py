from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load cleaned data + model
data = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open("LinerModel.pkl", "rb"))

@app.route("/")
def index():
    locations = sorted(data['location'].unique())
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form.get("location")
        total_sqft = float(request.form.get("total_sqft"))
        bath = int(request.form.get("bath"))
        bedroom = int(request.form.get("bedroom"))

        # Create DataFrame for model
        input_data = pd.DataFrame([[location, total_sqft, bath, bedroom]],
                                  columns=['location', 'total_sqft', 'bath', 'bedroom'])

        prediction = pipe.predict(input_data)[0]

        return jsonify({"status": "ok", "prediction": round(prediction,2)})  # in Lakhs
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
