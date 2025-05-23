from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Feature names and labels
feature_names = [f"Q{i+1}" for i in range(50)]
labels = [
    "Depression", "Anxiety", "PTSD/OCD", "Psychosis",
    "Eating Disorder", "Substance Use", "Sleep Disorder",
    "Suicidal Risk", "Personality Disorder"
]

# Load models and scaler
model_names = [
    "logistic_regression", "random_forest", "svm",
    "knn", "naive_bayes", "gradient_boosting"
]
models = {name: pickle.load(open(f"{name}.pkl", "rb")) for name in model_names}
scaler = pickle.load(open("scaler.pkl", "rb"))

# Advice per condition
advice_dict = {
    "Depression": "Consider speaking with a licensed therapist. Daily routines and physical activity can help reduce symptoms.",
    "Anxiety": "Practice relaxation techniques like deep breathing or meditation. Seek support if symptoms persist.",
    "PTSD/OCD": "Avoid known triggers and try structured therapy like CBT. Consult a specialist if intrusive thoughts worsen.",
    "Psychosis": "This may require immediate medical attention. Consult a psychiatrist promptly.",
    "Eating Disorder": "Focus on balanced meals and avoid diet culture triggers. Speak to a nutritionist and therapist.",
    "Substance Use": "Seek help from addiction support groups or clinics. Avoid environments that promote usage.",
    "Sleep Disorder": "Maintain a regular sleep schedule. Reduce screen time before bed and consult a sleep specialist.",
    "Suicidal Risk": "This is critical — seek immediate professional help or call a mental health helpline.",
    "Personality Disorder": "Therapy, especially DBT, may help in emotional regulation. Consistency in care is key."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        answers = [int(request.form.get(f"q{i+1}", 0)) for i in range(50)]
        input_df = pd.DataFrame([answers], columns=feature_names)
        input_scaled = scaler.transform(input_df)

        results = {}
        for name, model in models.items():
            prediction = model.predict(input_scaled)[0]
            results[name] = dict(zip(labels, prediction))

        # Collect advice
        advice_output = {}
        for model, preds in results.items():
            advice_output[model] = {
                label: advice_dict[label] for label, value in preds.items() if value == 1
            }

        return render_template("result.html", results=results, advice_output=advice_output)

    return render_template("form.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").lower()

    responses = {
        "depression": "Depression involves persistent sadness, fatigue, and loss of interest. Therapy and medication can help. Want to know how to seek help?",
        "anxiety": "Anxiety often involves excessive worry and physical symptoms like restlessness or rapid heartbeat. Try breathing exercises or speak with a therapist.",
        "ptsd": "PTSD can cause flashbacks, nightmares, and emotional numbness. It often needs structured therapy. Would you like a grounding technique to try?",
        "ocd": "OCD involves intrusive thoughts and repetitive behaviors. Cognitive Behavioral Therapy (CBT) is a common treatment.",
        "psychosis": "Psychosis can cause hallucinations or delusions. This should be evaluated by a psychiatrist as soon as possible.",
        "eating disorder": "These involve an unhealthy relationship with food and body image. Therapy and nutritional support are key. Would you like to talk to someone?",
        "sleep": "For sleep problems, follow a consistent routine, avoid caffeine, and reduce screen time. If symptoms persist, talk to a specialist.",
        "suicide": "Please don't face this alone. Call a local mental health helpline or speak to someone you trust. Help is available.",
        "therapist": "You can find therapists via online directories or local hospitals. Let me know if you need resources.",
        "clinic": "Search 'mental health clinic near me' on Google Maps, or contact local hospitals for referrals.",
        "help": "I'm here to support you. Let me know if you're feeling overwhelmed, anxious, or just need guidance."
    }

    reply = "I'm here to help. Please ask about specific symptoms or conditions."

    for key, val in responses.items():
        if key in user_msg:
            reply = val
            break

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run()
