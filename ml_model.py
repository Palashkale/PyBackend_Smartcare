import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Load dataset
df = pd.read_csv('./Dataset/training.csv')

# Encode the target
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# Split features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Symptom and disease mapping
symptom_index = X.columns.tolist()
disease_index = le.classes_.tolist()

# Train models
models = {
    "Decision Tree": DecisionTreeClassifier().fit(X, y),
    "Random Forest": RandomForestClassifier().fit(X, y),
    "Naive Bayes": GaussianNB().fit(X, y),
    "KNN": KNeighborsClassifier().fit(X, y),
}

def predict_disease(symptoms: list[str]):
    input_vector = [0] * len(symptom_index)
    for symptom in symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index.index(symptom)] = 1

    predictions = []
    confidence_scores = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([input_vector])[0]
            top_index = probs.argmax()
            predicted_disease = disease_index[top_index]
            confidence = round(probs[top_index] * 100, 2)
        else:
            pred = model.predict([input_vector])[0]
            predicted_disease = disease_index[pred]
            confidence = 100.0  # no prob available

        predictions.append(predicted_disease)
        confidence_scores[name] = {
            "disease": predicted_disease,
            "confidence": confidence
        }

    # Majority vote
    most_common = Counter(predictions).most_common(1)[0][0]

    return {
        "final_prediction": most_common,
        "individual_model_predictions": confidence_scores
    }

# Example usage
if __name__ == "__main__":
    input_symptoms = ["fever", "cough", "fatigue"]
    result = predict_disease(input_symptoms)
    print("Final Predicted Disease:", result["final_prediction"])
    print("Model-wise Predictions:")
    for model, info in result["individual_model_predictions"].items():
        print(f"  {model}: {info['disease']} ({info['confidence']}%)")
