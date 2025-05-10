
## Machine Condition Prediction Using Random Forest

### Sanjay Raj J

**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

### Project Overview

This project focuses on predicting the condition of industrial machines using machine learning. By analyzing important parameters such as temperature, vibration, oil quality, RPM, and others, we use a **Random Forest Classifier** to determine whether a machine is operating normally or showing signs of faults.

---

### Setup Instructions

To begin using this project, you need to install the required Python libraries. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

---

### Important Files for Prediction

Make sure the following files are available in your working directory:

* `random_forest_model.pkl` – Contains the trained Random Forest model.
* `scaler.pkl` – StandardScaler used during training for feature normalization.
* `selected_features.pkl` – A list of features used to train the model (important for correct input order).

These files are essential for making accurate predictions.

---

### How the Prediction Works

1. **Model and Scaler Loading**

   * Load the trained model using `joblib.load('random_forest_model.pkl')`.
   * Load the feature scaler using `joblib.load('scaler.pkl')`.
   * Load the list of selected features using `joblib.load('selected_features.pkl')`.

2. **Input Preparation**

   * Create a `pandas.DataFrame` that contains all the required features in one row.
   * Ensure that the column names exactly match those used during training.

3. **Feature Scaling**

   * Use the scaler to normalize your input data so it matches the format the model expects.

4. **Making Predictions**

   * Use `.predict()` to find out the predicted condition.
   * Use `.predict_proba()` to see the confidence level of each class prediction.

---

### Sample Prediction Script

You can use the following template to make predictions:

```python
import joblib
import pandas as pd

# Load model components
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Align input data with training features
new_data = new_data[selected_features]

# Normalize the input
scaled_data = scaler.transform(new_data)

# Predict machine condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

### Things to Keep in Mind

* Your input data must include all the same features used during model training.
* Input values should be within the typical range the model was trained on.
* The feature order matters. Do not rearrange the columns.

---

### If You Want to Retrain the Model

If you plan to retrain the model:

* Use the same steps for preprocessing and scaling.
* Keep the feature selection and order consistent.
* Save your new model, scaler, and feature list using `joblib`.

---

### Real-Life Applications

This project can be applied in:

* Monitoring machines in manufacturing environments.
* Identifying early signs of mechanical failure.
* Supporting maintenance teams using sensor data for smarter diagnostics.


