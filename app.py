import streamlit as st
import numpy as np
import onnxruntime as rt
import pandas as pd

# Title
st.title("Are you an Introvert or an Extrovert ?")

# Sidebar for user input
st.sidebar.header("Personal Information")

Time_spent_Alone = st.sidebar.slider("Spent Alone Time per Day (in Hours)", 0,11,0)
Stage_fear = st.sidebar.selectbox("Do you have Fear of getting on Stage?", ['Yes', 'No'])
Social_event_attendance = st.sidebar.slider("How Often You Attend Social Events?", 0,10,0)
Going_outside = st.sidebar.slider("How Many Hours a Day You Spent Outside?", 0,7,0)
Drained_after_socializing = st.sidebar.selectbox("Do you get Drained After Socializing", ['Yes', 'No'])
Friends_circle_size = st.sidebar.slider("How Many Friends Do You Have?", 0,15,0)
Post_frequency = st.sidebar.slider("How Often You Post on Internet?", 0,10,0)


Stage_fear = 1 if Stage_fear == "Yes" else 0
Drained_after_socializing = 1 if Drained_after_socializing == "Yes" else 0

# --- Final input as 10 features in correct order ---
input_data = pd.DataFrame([{
    "Time_spent_Alone": Time_spent_Alone,
    "Stage_fear": Stage_fear,
    "Social_event_attendance": Social_event_attendance,
    "Going_outside": Going_outside,
    "Drained_after_socializing": Drained_after_socializing,
    "Friends_circle_size": Friends_circle_size,
    "Post_frequency": Post_frequency,
}])



# Convert to numpy float32
input_np = input_data.to_numpy().astype(np.float32)

# Load and run ONNX model
try:
    sess = rt.InferenceSession("model/lgb_model.onnx")
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: input_np})
    prediction = int(pred[0][0])

    # Output
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.success("🎉 You're an **Extrovert**")
    else:
        st.error("👌 You're an **Introvert**")
except Exception as e:
    st.error(f"Model could not be loaded or run. Error: {e}")