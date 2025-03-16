from flask import Flask, request, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from fpdf import FPDF

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure Gemini API with key from .env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('model.h5')
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Directory for storing reports
REPORT_DIR = os.path.join(os.getcwd(), 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

# Function to preprocess the uploaded image
def preprocess_image(image_file, image_size=128):
    image_path = os.path.join('static', 'uploaded_image.jpg')
    image_file.save(image_path)
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, image_path

# Function to get prediction result
def get_prediction_result(prediction):
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]
    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index].capitalize()}"
    return result, confidence_score

# Function to get short recommendations using Gemini API, including survey data
def get_recommendations(result, confidence, survey_data):
    try:
        client = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Role:
You are an AI medical assistant providing recommendations based on MRI brain tumor predictions and patient survey data.

Input:
Tumor Type Prediction: {result}
Confidence Score: {confidence}
Patient Survey Data:
- Smoking: {survey_data['smoking']}
- Alcohol Consumption: {survey_data['alcohol']}
- Exercise Frequency: {survey_data['exercise']}
- Stress Level: {survey_data['stress']}
- Family History of Brain Tumors: {survey_data['family_history']}
- Known Genetic Conditions: {survey_data['genetic_conditions']}
- Diet: {survey_data['diet']}
- Sleep: {survey_data['sleep']}
- Age Group: {survey_data['age']}
- Gender: {survey_data['gender']}

Provide a short, concise response in this exact format, prioritizing:
- Lifestyle changes if smoking='Yes', alcohol='Regular', exercise='None', stress='High', or sleep='Less than 6'.
- Genetic counseling if family_history='Yes' or genetic_conditions!='None'.
Otherwise, focus on standard medical steps:
1. Immediate Next Steps:
- Step 1
- Step 2
2. Potential Treatment Options:
- Option 1
- Option 2
- Option 3
3. Monitoring or Lifestyle Recommendations:
- Rec 1
- Rec 2
- Rec 3
"""
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 256,
            }
        )
        return response.text
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# Function to generate an attractive PDF report
def generate_report(result, confidence, recommendations, survey_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.pdf"
    report_path = os.path.join(REPORT_DIR, report_filename)

    # Create PDF with FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set margins
    pdf.set_margins(left=20, top=20, right=20)

    # Header
    pdf.set_fill_color(30, 60, 120)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "Brain Tumor Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    # Reset text color
    pdf.set_text_color(0, 0, 0)

    # Patient Survey Data Section with Table
    pdf.ln(10)
    pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Patient Survey Data", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    # Table for survey data
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(40, 8, "Field", 1, 0, 'C', 1)
    pdf.cell(50, 8, "Value", 1, 1, 'C', 1)
    pdf.set_fill_color(255, 255, 255)
    pdf.cell(40, 8, "Smoking", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['smoking'], 1, 1, 'L')
    pdf.cell(40, 8, "Alcohol", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['alcohol'], 1, 1, 'L')
    pdf.cell(40, 8, "Exercise", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['exercise'], 1, 1, 'L')
    pdf.cell(40, 8, "Stress", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['stress'], 1, 1, 'L')
    pdf.cell(40, 8, "Family History", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['family_history'], 1, 1, 'L')
    pdf.cell(40, 8, "Genetic Conditions", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['genetic_conditions'], 1, 1, 'L')
    pdf.cell(40, 8, "Diet", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['diet'], 1, 1, 'L')
    pdf.cell(40, 8, "Sleep", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['sleep'], 1, 1, 'L')
    pdf.cell(40, 8, "Age Group", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['age'], 1, 1, 'L')
    pdf.cell(40, 8, "Gender", 1, 0, 'L')
    pdf.cell(50, 8, survey_data['gender'], 1, 1, 'L')

    # Prediction Section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Prediction Details", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Result: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence Score: {confidence * 100:.2f}%", ln=True)

    # Recommendations Section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Recommendations", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, recommendations)

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "© 2025 xAI - Brain Tumor Detection", ln=True, align="C")

    pdf.output(report_path)
    return report_filename

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error="No image selected")

    try:
        # Collect survey data
        survey_data = {
            'smoking': request.form['smoking'],
            'alcohol': request.form['alcohol'],
            'exercise': request.form['exercise'],
            'stress': request.form['stress'],
            'family_history': request.form['family_history'],
            'genetic_conditions': request.form['genetic_conditions'],
            'diet': request.form['diet'],
            'sleep': request.form['sleep'],
            'age': request.form['age'],
            'gender': request.form['gender']
        }

        # Preprocess the image
        img_array, image_path = preprocess_image(image_file, image_size=128)

        # Make prediction
        prediction = model.predict(img_array)

        # Get result and confidence
        result, confidence = get_prediction_result(prediction)

        # Get recommendations (only for PDF, includes survey data)
        recommendations = get_recommendations(result, confidence, survey_data)

        # Generate the PDF report with recommendations and survey data
        report_filename = generate_report(result, confidence, recommendations, survey_data)

        # Pass result to template without recommendations
        return render_template('result.html',
                             result=result,
                             confidence=confidence * 100,
                             image_path='uploaded_image.jpg',
                             report_filename=report_filename)
    except Exception as e:
        return render_template('index.html', error=f"Error processing image: {str(e)}")

@app.route('/download/<filename>')
def download_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)