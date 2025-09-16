# 🚦 Bangalore Traffic Prediction API

A robust Flask-based API server that leverages Machine Learning to provide real-time traffic predictions for Bangalore. This service uses a Random Forest Regressor model to predict traffic volume and average speed based on various input parameters.

## ✨ Features

-   **📈 Traffic Volume Prediction**: Predicts the total traffic volume for a given location and time.
-   **🚗 Average Speed Prediction**: Estimates the average vehicle speed.
-   **📊 Derived Congestion Level**: Automatically calculates the congestion level (Low, Medium, High, Severe) based on the predicted traffic volume.
-   **⚡️ Real-Time API**: Designed for fast, on-the-fly predictions.
-   **🩺 Health Check**: Includes a `/health` endpoint to monitor API status.

## 📂 Project Structure

```
Random-Forest/
├── app/
│   ├── __init__.py
│   ├── api.py              # Flask API endpoints
│   ├── preprocessing.py    # Data preprocessing logic
│   ├── train_rf.py         # Training script for the traffic volume model
│   └── train_rf_speed.py   # Training script for the average speed model
├── data/
│   └── bangalore_traffic.csv # Training dataset
├── models/
│   └── .gitkeep            # Trained models will be saved here
├── venv/                   # Virtual environment
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## ⚙️ Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

-   Python 3.10+
-   `pip` and `venv`

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd Random-Forest
```

### 3. Set Up Virtual Environment

Create and activate a virtual environment to manage project dependencies.

-   **Windows**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
-   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 4. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` file, create one with the following content:)*
```
flask
pandas
scikit-learn>=1.4.0
numpy
joblib
```

## 🚀 Usage

Before running the API, you must train the models.

### 1. Train the Models

The models are trained using the provided dataset. Run the training scripts in the following order. This process will create `rf_traffic.pkl`, `rf_speed.pkl`, and `preprocessor.joblib` inside the `models/` directory.

```bash
# First, train the traffic volume model and create the preprocessor
python app/train_rf.py

# Second, train the average speed model using the same preprocessor
python app/train_rf_speed.py
```

### 2. Run the API Server

Once the models are trained and saved, you can start the Flask API server.

```bash
python app/api.py
```

The server will start on `http://0.0.0.0:5000`. You can now send requests to the API.

## 📡 API Endpoints

### Health Check

-   **Endpoint**: `/health`
-   **Method**: `GET`
-   **Description**: Checks if the API is running and responsive.
-   **Success Response (200)**:
    ```json
    {
      "status": "ok"
    }
    ```

### Traffic Prediction

-   **Endpoint**: `/predict`
-   **Method**: `POST`
-   **Description**: Predicts traffic volume and average speed. `Average_Speed` and `Congestion_Level` are calculated on the server and should **not** be sent in the request.

-   **Request Body**:
    ```json
    {
      "Date": "2025-09-09 08:00:00",
      "Area_Name": "Indiranagar",
      "Road_Intersection_Name": "100ft road",
      "Weather_Conditions": "Clear",
      "Roadwork_and_Construction_Activity": "No"
    }
    ```

-   **Success Response (200)**:
    ```json
    {
      "traffic_count": 9500,
      "average_speed": 25.5,
      "congestion_level": "Severe"
    }
    ```

-   **Error Response (400)**:
    *If the JSON is missing or invalid.*
    ```json
    {
      "error": "Invalid JSON"
    }
    ```

## 🛠️ Technology Stack

-   **Backend**: Flask
-   **Machine Learning**: Scikit-learn
-   **Data Handling**: Pandas, NumPy

---
