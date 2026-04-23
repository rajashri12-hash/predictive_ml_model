import streamlit as st
from project13_predictive_app.predict import predict_from_dict, predict_subscription_from_dict


st.set_page_config(page_title="Predictive Analytics App", layout="centered")

st.title("Predictive Analytics — Purchase & Subscription")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

task = st.radio("Task", ["Predict Amount", "Predict Subscription"])

with st.form("input_form"):
    age = st.number_input("Age", min_value=16, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    category = st.selectbox("Category", ["Clothing", "Accessories", "Footwear", "Outerwear"])
    location = st.selectbox("Location", ["CA", "NY", "TX", "FL", "WA", "IL"])
    color = st.selectbox("Color", ["Blue", "Red", "Black", "Green", "Yellow", "Purple"])
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    rating = st.slider("Review Rating", 2, 5, 4)
    # Only ask for subscription status when predicting amount (not when predicting it)
    if task == "Predict Amount":
        subscription = st.selectbox("Subscription Status", ["Yes", "No"])
    else:
        subscription = None
    shipping = st.selectbox("Shipping Type", ["Standard", "Express", "Free Shipping"])
    discount = st.selectbox("Discount Applied", ["Yes", "No"])
    promo = st.selectbox("Promo Code Used", ["Yes", "No"])
    prev = st.number_input("Previous Purchases", min_value=0, max_value=100, value=5)
    payment = st.selectbox("Payment Method", ["Credit Card", "Cash", "Venmo"])
    freq = st.selectbox("Frequency of Purchases", ["Monthly", "Weekly", "Fortnightly"])

    submitted = st.form_submit_button("Predict")

if submitted:
    data = {
        "Age": age,
        "Gender": gender,
        "Category": category,
        "Location": location,
        "Color": color,
        "Season": season,
        "Review Rating": rating,
    }
    # include subscription only if provided (Predict Amount task)
    if subscription is not None:
        data["Subscription Status"] = subscription
    else:
        # ensure key absent when predicting subscription
        pass
    # continue adding remaining fields
    data.update({
        "Shipping Type": shipping,
        "Discount Applied": discount,
        "Promo Code Used": promo,
        "Previous Purchases": prev,
        "Payment Method": payment,
        "Frequency of Purchases": freq,
    })
    try:
        st.session_state.prediction_error = None
        if task == "Predict Amount":
            val = predict_from_dict(data)
            st.session_state.prediction_result = {
                "task": task,
                "amount": float(val),
            }
        else:
            res = predict_subscription_from_dict(data)
            st.session_state.prediction_result = {
                "task": task,
                "probability": float(res["probability"]),
                "label": res["label"],
            }
    except Exception as e:
        st.session_state.prediction_result = None
        st.session_state.prediction_error = str(e)

if st.session_state.prediction_error:
    st.error(f"Prediction error: {st.session_state.prediction_error}")

if st.session_state.prediction_result:
    if st.session_state.prediction_result["task"] == "Predict Amount":
        st.metric(
            label="Predicted Purchase Amount (USD)",
            value=f"${st.session_state.prediction_result['amount']:,.2f}",
        )
    else:
        st.metric(
            label="Subscription Probability (Yes)",
            value=f"{st.session_state.prediction_result['probability']:.2%}",
        )
        st.write(
            f"Predicted Subscription Status: **{st.session_state.prediction_result['label']}**"
        )
