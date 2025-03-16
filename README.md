# 🧠 Brain Tumor Detection using AI



## 🚀 Overview
This AI-powered **Brain Tumor Detection** system utilizes **Deep Learning** and **Generative AI** to predict brain tumor types from MRI scans and provide personalized medical recommendations. It integrates **TensorFlow**, **Flask**, **Google Gemini AI**, and **FPDF** to create a comprehensive diagnostic tool.

---

## 🎯 Features
✅ **MRI Scan Classification** - Detects **Pituitary, Glioma, Meningioma**, or **No Tumor**.
✅ **Confidence Score** - Displays the model's confidence in the prediction.
✅ **Survey-Based AI Recommendations** - Uses **Google Gemini AI** to generate personalized lifestyle and treatment suggestions.
✅ **PDF Report Generation** - Generates a detailed **AI-powered medical report** with recommendations.
✅ **User-Friendly Web Interface** - Built using **Flask** & **Streamlit** for easy interaction.
✅ **Secure Authentication** - Supports **user authentication and profile management**.

---

## 🏗️ Tech Stack
🔹 **Deep Learning:** TensorFlow, Keras
🔹 **Backend:** Flask
🔹 **Frontend:** HTML,CSS,JS
🔹 **AI Recommendations:** Google Gemini AI
🔹 **Database:** MongoDB (for user data & reports)
🔹 **Containerization:** Docker
🔹 **Deployment:** AWS/GCP

---

## 🔧 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Add your GEMINI_API_KEY

# Run the application
python app.py
```

---

## 🚀 Usage
1. **Register/Login** to access personalized reports.
2. **Upload an MRI scan** of the brain.
3. **Fill out the patient survey** with lifestyle details.
4. **Receive AI predictions** with confidence scores.
5. **Download a detailed PDF report** with AI-generated recommendations.

---

## 📂 Project Structure
```
📦 brain-tumor-detection
 ┣ 📂 static                # Stores uploaded images
 ┣ 📂 templates             # HTML templates for Flask
 ┣ 📂 reports               # Generated PDF reports
 ┣ 📂 models                # AI model storage
 ┣ 📂 authentication        # User authentication module
 ┣ 📜 app.py                # Main Flask application
 ┣ 📜 model.h5              # Pretrained TensorFlow model
 ┣ 📜 requirements.txt      # Dependencies
 ┣ 📜 .env.example          # Environment variables
 ┗ 📜 README.md             # Project documentation
```

---

## 📊 AI-Powered PDF Report
- **Detailed prediction results**
- **Confidence level analysis**
- **AI-based recommendations** for lifestyle changes & treatments
- **User-friendly visualization** of medical insights
- **Secure user-based report storage & access**

---

## 🤖 AI Model
- Pretrained **TensorFlow CNN model** for MRI classification.
- Fine-tuned on **brain tumor datasets** for high accuracy.
- Integrates **Generative AI (Gemini API)** for intelligent recommendations.

---

## 📌 To-Do
- [ ] Improve model accuracy with **transfer learning**.
- [ ] Implement **real-time inference API**.
- [ ] Deploy to **AWS/GCP** for public access.
- [ ] Add **multi-disease detection support**.
- [ ] Enhance **security & authentication** features.

---

## 💡 Contributing
🚀 Contributions are welcome! Feel free to fork this repo, create a branch, and submit a **pull request**. 

---

## 📜 License
📜 This project is licensed under the **MIT License**.

---

## ✨ Credits
Developed by **Your Name** | Inspired by AI-driven Healthcare Innovations ❤️

