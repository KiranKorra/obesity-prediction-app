import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and label encoder
model = joblib.load('best_obesity_model.pkl')
le = joblib.load('label_encoder.pkl')

# Define the expected feature columns (this is the fix)
# These must match exactly what was used during training
feature_columns = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'bmi',
    'Gender_Male',
    'family_history_with_overweight_yes',
    'FAVC_yes',
    'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'SMOKE_yes',
    'SCC_yes',
    'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

def predict_obesity(input_data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # Align columns with training features (Critical fix)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probabilities = model.predict_proba(input_encoded)[0]
    
    predicted_class = le.inverse_transform([prediction])[0]
    confidence = max(probabilities) * 100
    
    return predicted_class, confidence, probabilities

# ====================== Streamlit App ======================
st.title("🩺 Obesity Level Prediction System")
st.write("Enter your details below to predict your obesity level based on lifestyle factors.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 14, 65, 25)
    height = st.slider("Height (meters)", 1.45, 1.98, 1.70, step=0.01)
    weight = st.slider("Weight (kg)", 39.0, 173.0, 80.0, step=0.1)
    
with col2:
    family_history = st.selectbox("Family History of Overweight", ["no", "yes"])
    favc = st.selectbox("Frequent High Caloric Food Consumption", ["no", "yes"])
    fcvc = st.slider("Frequency of Vegetable Consumption (1-3)", 1.0, 3.0, 2.5, step=0.1)
    ncp = st.slider("Number of Main Meals (1-4)", 1.0, 4.0, 3.0, step=0.1)
    caec = st.selectbox("Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Do you smoke?", ["no", "yes"])
    ch2o = st.slider("Daily Water Intake (liters)", 1.0, 3.0, 2.0, step=0.1)
    scc = st.selectbox("Monitor Calories", ["no", "yes"])
    faf = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, 1.0, step=0.1)
    tue = st.slider("Time Using Technology (hours/day)", 0.0, 2.0, 0.8, step=0.1)
    calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently"])
    mtrans = st.selectbox("Main Transportation", 
                         ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Calculate BMI
bmi = weight / (height ** 2)

# Prepare input data
input_data = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': family_history,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': mtrans,
    'bmi': bmi
}

if st.button("🔮 Predict Obesity Level", type="primary"):
    with st.spinner("Predicting..."):
        predicted_class, confidence, probs = predict_obesity(input_data)
        
        st.success(f"**Predicted Obesity Level:** {predicted_class}")
        st.metric("Confidence Score", f"{confidence:.1f}%")
        
        # Show probabilities
        prob_df = pd.DataFrame({
            'Obesity Level': le.classes_,
            'Probability (%)': np.round(probs * 100, 2)
        }).sort_values('Probability (%)', ascending=False)
        
        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # BMI Insight
        st.subheader("Your BMI")
        st.info(f"**{bmi:.2f}**")

        if bmi < 18.5:
            st.caption("Underweight range")
        elif bmi < 25:
            st.caption("Normal weight range")
        elif bmi < 30:
            st.caption("Overweight range")
        else:
            st.caption("Obesity range")

st.caption("")
