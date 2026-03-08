# AI Disaster Prediction System

The **AI Disaster Prediction System** is a full-stack application that leverages Machine Learning to predict the risk of various natural disasters. It consists of a Python Flask backend for serving ML predictions and a React + Vite frontend for a modern, interactive user interface.

## 🌍 Supported Disasters
The system currently supports risk prediction for:
- **Flood**
- **Earthquake**
- **Forest Fire**
- **Landslide**
- **Tsunami**

---

## 🏗️ Project Structure

```text
disaster_prediction_system/
│
├── backend/            # Flask REST API handling ML predictions
│   ├── app.py          # Main application and routing
│   └── auth.py         # JWT authentication & user management
├── frontend/           # React + Vite application (UI)
│   └── src/pages/      # Login, Register, Dashboard & Prediction pages
├── models/             # Pre-trained ML models and scalers (.pkl files)
├── datasets/           # Raw and processed datasets (.csv)
├── scripts/            # Scripts for training models
├── utils/              # Utility functions (preprocessing, etc.)
├── requirements.txt    # Python dependencies
├── users.db            # SQLite database for storing registered users
└── start_project.bat   # Automated startup script
```

---

## 🚀 How to Run the Project

### Option 1: One-Click Start (Windows Recommended)
The easiest way to start the system is using the provided batch script.
1. Simply double-click the `start_project.bat` file in the root directory.
2. The script will automatically open two command prompts:
   - One for the **Flask Backend API** (running on port 5000 by default).
   - One for the **React Frontend** (running on port 5173).
3. Open your browser and go to: `http://localhost:5173`
4. Register a new account, and sign in to access the Dashboard.

### Option 2: Manual Start

**1. Start the Backend:**
```bash
# Install dependencies (only needed once)
pip install -r requirements.txt

# Run the Flask app
python backend/app.py
```
*The backend will be available at `http://127.0.0.1:5000`.*

**2. Start the Frontend:**
```bash
cd frontend

# Install Node dependencies (only needed once)
npm install

# Run the development server
npm run dev
```
*The frontend will be available at `http://localhost:5173`.*

---

## 🧠 Model Training (Run Query)

If you have updated the datasets in the `/datasets` folder or want to retrain the machine learning models from scratch, run the following pipeline script from the root directory:

```bash
python scripts/train_all_pipelines.py
```
This script will parse the `.csv` files from the `datasets/` directory, train the respective models (Random Forest, XGBoost, etc.), and save the updated `.pkl` models and scalers directly into the `models/` directory.

---

## 📡 API Endpoints

The system exposes a REST API that the frontend communicates with. 

### **Authentication**
- **Register:** `POST /api/register` (Requires JSON `{"username": "...", "password": "..."}`)
- **Login:** `POST /api/login` (Returns a JWT token on success)

### **1. Health Check**
Check if the API is running and which models are fully loaded.
- **Endpoint:** `GET /health`

### **2. Supported Features**
Get the necessary features expected for a specific disaster model.
- **Endpoint:** `GET /features/<disaster>`
- **Example:** `GET /features/flood`

### **3. Predict Disaster Risk**
Get a prediction for a specific disaster type.
- **Endpoint:** `POST /predict/<disaster>`
- **Parameters:** `<disaster>` can be `flood`, `earthquake`, `forestfire`, `landslide`, or `tsunami`.
- **Payload Example:**
  ```json
  {
    "data": {
      "feature1": 5.4,
      "feature2": 12.1,
      "feature3": 0.8
    }
  }
  ```
- **Response Example:**
  ```json
  {
    "label": "Flood Risk",
    "prediction": 1,
    "probability": 0.885,
    "risk_level": "High"
  }
  ```

---

## 🛠️ Technology Stack
- **Frontend:** React, Vite, Tailwind CSS, Lucide React (for premium dashboard UI)
- **Backend:** Python, Flask, Flask-CORS
- **Authentication:** PyJWT (JSON Web Tokens), SQLite, Werkzeug Security
- **Machine Learning:** scikit-learn, pandas, numpy, joblib
