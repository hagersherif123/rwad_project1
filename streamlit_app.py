import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# تأكد من أن st.set_page_config هو أول دالة يتم استدعاؤها
st.set_page_config(page_title="Vehicle Price Prediction", page_icon="🚗", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #e1e4e8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to download and load the model
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        response = requests.get(url)
        model = pickle.load(BytesIO(response.content))
        if isinstance(model, RandomForestRegressor):
            return model
        else:
            st.error("Loaded model is not a RandomForestRegressor.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data, model):
    input_df = pd.DataFrame(data, index=[0])
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    model_features = model.feature_names_in_
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)
    return input_df_encoded

# Main Streamlit app
def main():
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020)
        used_or_new = st.selectbox("Used or New 🏷️", ["Used", "New"])
        transmission = st.selectbox("Transmission ⚙️", ["Manual", "Automatic"])
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1)
        drive_type = st.selectbox("Drive Type 🛣️", ["FWD", "RWD", "AWD"])
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"])

    with col2:
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1)
        kilometres = st.number_input("Kilometres 🛣️", min_value=0, value=50000, step=1000)
        cylinders_in_engine = st.number_input("Cylinders in Engine 🔢", min_value=1, value=4)
        body_type = st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"])
        doors = st.selectbox("Number of Doors 🚪", [2, 3, 4, 5])

    if st.button("Predict Price 💰"):
        with st.spinner("Calculating..."):
            file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
            model = load_model_from_drive(file_id)
            if model is not None:
                input_data = {
                    'Year': year,
                    'UsedOrNew': used_or_new,
                    'Transmission': transmission,
                    'Engine': engine,
                    'DriveType': drive_type,
                    'FuelType': fuel_type,
                    'FuelConsumption': fuel_consumption,
                    'Kilometres': kilometres,
                    'CylindersinEngine': cylinders_in_engine,
                    'BodyType': body_type,
                    'Doors': doors
                }
                input_df = preprocess_input(input_data, model)
                try:
                    prediction = model.predict(input_df)
                    st.success("Prediction successful!")
                    st.markdown(f"<div class='prediction-box'>Predicted Price: ${prediction[0]:,.2f}</div>", unsafe_allow_html=True)
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': model.feature_names_in_,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)

                    # Plotting feature importance using plotly
                    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                 title='Top 10 Important Features', labels={'importance': 'Importance', 'feature': 'Feature'})
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)

                    # Displaying input data and prediction as a table
                    st.subheader("Input Data and Prediction")
                    input_data['Predicted Price'] = f"${prediction[0]:,.2f}"
                    input_df_display = pd.DataFrame(input_data, index=[0])
                    st.dataframe(input_df_display)

                    # Plotting categorical distributions using plotly
                    st.subheader("Categorical Feature Distributions")
                    fig_used_new = px.pie(input_df_display, names='UsedOrNew', title='Used or New')
                    fig_transmission = px.pie(input_df_display, names='Transmission', title='Transmission')
                    fig_drive_type = px.pie(input_df_display, names='DriveType', title='Drive Type')
                    fig_fuel_type = px.pie(input_df_display, names='FuelType', title='Fuel Type')

                    st.plotly_chart(fig_used_new)
                    st.plotly_chart(fig_transmission)
                    st.plotly_chart(fig_drive_type)
                    st.plotly_chart(fig_fuel_type)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Failed to load the model.")

if __name__ == "__main__":
    main()
