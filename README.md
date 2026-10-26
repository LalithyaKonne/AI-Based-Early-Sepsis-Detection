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

<img width="978" height="848" alt="image" src="https://github.com/user-attachments/assets/56c7250f-1ced-4561-88dd-9f768a48da2e" />


### 2. Patient Selection

Choose between retrieving an existing patient record and uploading data for a new patient.

<img width="1323" height="822" alt="image" src="https://github.com/user-attachments/assets/55c32a93-17ba-4112-877c-4863c06fe379" />


### 3. Patient Data Upload

Upload ICU data in CSV or Excel format before running a prediction.

<img width="883" height="791" alt="image" src="https://github.com/user-attachments/assets/26d58dda-0e4f-406c-a848-6d57330bbbea" />


### 4. Manual Patient Data Entry

Enter patient parameters such as heart rate, temperature, respiratory rate, mean arterial pressure (MAP), oxygen saturation, creatinine, platelets, age, and ICU hours.

<img width="670" height="842" alt="image" src="https://github.com/user-attachments/assets/5eeb39cd-b153-4490-b8e1-cb822fabd1cb" />


### 5. Sepsis Risk Dashboard

View patient information, vital signs, estimated risk probability, severity, and prediction results.

<img width="1600" height="739" alt="image" src="https://github.com/user-attachments/assets/e2fecd27-c2a4-4d9f-bf3f-3418199842f0" />


### 6. Explainable AI and Clinical Decision Support

Explore SHAP-based feature contributions and the clinical monitoring suggestions displayed by the application.

<img width="1600" height="724" alt="image" src="https://github.com/user-attachments/assets/92854434-0691-4c6d-9c84-5b81580897f9" />


### 7. Patient History and Doctor Notes

Review previous assessments, recorded notes, and the top contributing risk factors.

<img width="1600" height="740" alt="image" src="https://github.com/user-attachments/assets/5c38901a-7aba-445a-99a1-f13f7eea3a3f" />


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
