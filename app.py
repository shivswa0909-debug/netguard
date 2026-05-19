from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

@app.route("/")
def home():
    return render_template("index.html", results=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded!", results=None)

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="Please select a file.", results=None)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath, names=col_names)

        for col in ["protocol_type", "service", "flag"]:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        X = df.drop(columns=["label", "difficulty"], errors="ignore")
        predictions = model.predict(X)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "row": i + 1,
                "protocol": df["protocol_type"].iloc[i],
                "src_bytes": df["src_bytes"].iloc[i],
                "dst_bytes": df["dst_bytes"].iloc[i],
                "flag": df["flag"].iloc[i],
                "prediction": pred
            })

        total   = len(results)
        threats = sum(1 for r in results if r["prediction"] == "attack")
        safe    = total - threats

        return render_template("index.html",
            results=results, total=total,
            threats=threats, safe=safe, filename=file.filename)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}", results=None)

if __name__ == "__main__":
    app.run(debug=False)