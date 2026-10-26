# AI-Based Early Sepsis Detection System

An AI-powered early warning system that analyzes patient vital signs and clinical parameters to estimate sepsis risk. The application provides a patient dashboard, risk probability, explainable AI visualizations, patient history, and clinical decision-support information.

## Features

- **Hospital Staff Login:** Role-based login interface for doctors, nurses, and administrators.
- **Patient Selection:** Retrieve existing patient records or enter data for a new patient.
- **Patient Data Upload:** Upload ICU patient data using CSV or Excel files.
- **Manual Data Entry:** Enter vital signs and clinical parameters manually.
- **AI-Based Risk Prediction:** Estimate sepsis risk using a trained machine learning model.
- **Risk Assessment Dashboard:** Display risk status, probability, severity level, and vital signs.
- **Explainable AI:** Visualize contributing features using SHAP.
- **Patient History:** Review prediction history, risk factors, and doctor notes.
- **Clinical Decision Support:** Display monitoring suggestions based on the assessment.
- **Report Generation:** Generate patient assessment reports.

## Application Screenshots

### 1. Hospital Staff Login

The login page provides access for different hospital staff roles.

![Hospital Staff Login](screenshots/login.png)

### 2. Patient Selection

Choose between retrieving an existing patient record and uploading data for a new patient.

![Patient Selection](screenshots/patient-selection.png)

### 3. Patient Data Upload

Upload ICU data in CSV or Excel format before running a prediction.

![Patient Data Upload](screenshots/patient-upload.png)

### 4. Manual Patient Data Entry

Enter patient parameters such as heart rate, temperature, respiratory rate, mean arterial pressure (MAP), oxygen saturation, creatinine, platelets, age, and ICU hours.

![Manual Patient Data Entry](screenshots/manual-entry.png)

### 5. Sepsis Risk Dashboard

View patient information, vital signs, estimated risk probability, severity, and prediction results.

![Sepsis Risk Dashboard](screenshots/dashboard.png)

### 6. Explainable AI and Clinical Decision Support

Explore SHAP-based feature contributions and the clinical monitoring suggestions displayed by the application.

![Explainable AI Dashboard](screenshots/explainable-ai.png)

### 7. Patient History and Doctor Notes

Review previous assessments, recorded notes, and the top contributing risk factors.

![Patient History and Doctor Notes](screenshots/history-notes.png)

## Technology Stack

- **Frontend:** React, Vite, JavaScript, CSS
- **Backend / ML:** Python, machine learning
- **Model Explainability:** SHAP
- **Data Processing:** CSV and Excel files
- **Deployment:** Vercel and Hugging Face Spaces (deployment configuration may vary)

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/LalithyaKonne/AI-Based-Early-Sepsis-Detection.git
   ```

2. Navigate to the frontend:

   ```bash
   cd AI-Based-Early-Sepsis-Detection/frontend
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open the local URL printed in the terminal.

Follow the backend setup instructions in the repository before using prediction features that require the Python API.

## Important Note

This project is an educational and research prototype. Its predictions and recommendations are not a substitute for professional medical judgment and should not be used as the sole basis for clinical decisions.
