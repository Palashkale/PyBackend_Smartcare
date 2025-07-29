from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import docx
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load training data
df = pd.read_csv('./Dataset/training.csv')
le = LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

symptom_index = X.columns.tolist()
disease_index = le.classes_.tolist()

models = {
    
    
    "naive_bayes": GaussianNB().fit(X, y),
    "knn": KNeighborsClassifier().fit(X, y),
    "random_forest": RandomForestClassifier().fit(X, y),
    "decision_tree": DecisionTreeClassifier().fit(X, y),
}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_symptoms_from_text(text):
    text = text.lower()
    return [symptom for symptom in symptom_index if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', text)]

def predict_disease(symptoms: list[str]):
    input_vector = [0] * len(symptom_index)
    for symptom in symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index.index(symptom)] = 1

    result = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([input_vector])[0]
            top_idx = proba.argmax()
            disease = disease_index[top_idx]
            confidence = round(proba[top_idx] * 100, 2)
            result[name] = {
                "disease": disease,
                "confidence": f"{confidence}%"
            }
        else:
            pred = model.predict([input_vector])[0]
            result[name] = {
                "disease": disease_index[pred],
                "confidence": "N/A"
            }
    return result

def get_ai_suggestion(symptoms: list[str], predicted_disease: dict):
    prompt = f"""

 "You are a licensed and knowledgeable medical assistant. "
Only answer questions strictly related to health or medical advice.
Do not entertain non-medical queries.
Based on the symptoms: {', '.join(symptoms)} and predictions: {predicted_disease}, provide a short 75-word actionable advice.

Structure response with:
- ✅ Main advice
- 📝 Notes
- 🚨 If urgent, mention emergency action
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a licensed doctor and medical assistant. You must only answer health-related questions. Politely reject any non-medical requests."
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Could not get AI suggestion: {str(e)}"

@app.route('/predict-from-report', methods=['POST'])
def predict_from_report():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.pdf', '.docx')):
        return jsonify({"status": "error", "message": "Only PDF and DOCX files are supported"}), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath) if filename.endswith(".pdf") else extract_text_from_docx(filepath)
        symptoms = extract_symptoms_from_text(text)

        os.remove(filepath)

        if not symptoms:
            return jsonify({
                "status": "error",
                "message": "No known symptoms found",
                "sample_text": text[:500] + "..." if len(text) > 500 else text,
                "available_symptoms": symptom_index
            }), 404

        prediction = predict_disease(symptoms)

        return jsonify({
            "status": "success",
            "source": "file",
            "symptoms": symptoms,
            "predicted_disease": prediction
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Processing error: {str(e)}"}), 500

@app.route('/predict-from-json', methods=['POST'])
def predict_from_json():
    data = request.get_json()
    symptoms = data.get("symptoms")

    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"status": "error", "message": "Invalid or missing 'symptoms' list"}), 400

    prediction = predict_disease(symptoms)

    return jsonify({
        "status": "success",
        "source": "json",
        "symptoms": symptoms,
        "predicted_disease": prediction
    }), 200

@app.route('/ai-suggestion', methods=['POST'])
def ai_suggestion():
    data = request.get_json()
    symptoms = data.get("symptoms")
    predicted_disease = data.get("predicted_disease")

    if not symptoms or not predicted_disease:
        return jsonify({"status": "error", "message": "Missing symptoms or predicted_disease"}), 400

    suggestion = get_ai_suggestion(symptoms, predicted_disease)

    return jsonify({
        "status": "success",
        "symptoms": symptoms,
        "predicted_disease": predicted_disease,
        "suggestion": suggestion
    }), 200

@app.route('/api/medical-chat', methods=['POST'])
def medical_chat():
    data = request.get_json()
    prompt = data.get("prompt", "")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a licensed and knowledgeable medical assistant. "
                    "You must answer only health, wellness, disease, treatment, or symptom-related questions. "
                    "If someone asks about non-medical topics, politely refuse. Keep answers clear and under 25 words with bullet points and emojis when helpful."
                )
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"status": "success", "reply": reply}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"AI response error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
