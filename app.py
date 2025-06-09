from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
# Load model and vectorizer (placed in the same folder as app.py)
model = joblib.load('best_complaint_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Create Flask app
app = Flask(__name__)
CORS(app)

label_map = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        text_vector = vectorizer.transform([text])
        pred_class = model.predict(text_vector)[0]
        pred_label = label_map.get(pred_class, "Unknown")
        return jsonify({'prediction': pred_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
