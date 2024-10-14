import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Create a function to generate plots
def create_dashboard(df):
    # Scatter plot for Fuel Consumption vs. Price
    scatter = px.scatter(df, x='FuelConsumption', y='Price', color='FuelType',
                         title='Fuel Consumption vs Price', 
                         labels={'FuelConsumption': 'Fuel Consumption (L/100km)', 'Price': 'Price ($)'})

    # Histogram for Price Distribution
    histogram = px.histogram(df, x='Price', nbins=30, 
                             title='Distribution of Vehicle Prices', 
                             labels={'Price': 'Price ($)'})

    # Box Plot for Price by Transmission Type
    box = px.box(df, x='Transmission', y='Price', 
                 title='Price Distribution by Transmission Type', 
                 labels={'Transmission': 'Transmission Type', 'Price': 'Price ($)'})

    # Dashboard Layout using Plotly
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Fuel Consumption vs Price', 'Price Distribution', 'Price by Transmission'),
                        specs=[[{"type": "scatter"}, {"type": "histogram"}], [{"type": "box"}, None]])

    # Adding traces to the subplots
    fig.add_trace(go.Scatter(x=df['FuelConsumption'], y=df['Price'], mode='markers',
                             marker=dict(color=df['FuelType'].apply(lambda x: 'blue' if x == 'Petrol' else 'red')), name='Fuel vs Price'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['Price'], nbinsx=30, name='Price Distribution'), row=1, col=2)
    fig.add_trace(go.Box(y=df['Price'], x=df['Transmission'], name='Price by Transmission'), row=2, col=1)

    # Update layout for interactivity and aesthetics
    fig.update_layout(height=800, width=1200, title_text="Vehicle Prices Dashboard", showlegend=False)

    return fig

# Main Streamlit app
def main():
    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price.")

    # Load data for visualization
    url = "https://drive.google.com/uc?id=1FjZWfVGrIIdtQVXu4g89lcVgQRBg8h1j"
    df = pd.read_csv(url)  # تأكد من تحميل بياناتك
    df = pd.DataFrame()  # استبدل هذا بجلب البيانات الخاصة بك.

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year 📅", min_value=1900, max_value=2024, value=2020, key="year")
        used_or_new = st.selectbox("Used or New 🏷️", ["Used", "New"], key="used_or_new")
        transmission = st.selectbox("Transmission ⚙️", ["Manual", "Automatic"], key="transmission")
        engine = st.number_input("Engine Size (L) 🔧", min_value=0.0, value=2.0, step=0.1, key="engine")
        drive_type = st.selectbox("Drive Type 🛣️", ["FWD", "RWD", "AWD"], key="drive_type")
        fuel_type = st.selectbox("Fuel Type ⛽", ["Petrol", "Diesel", "Electric", "Hybrid"], key="fuel_type")

    with col2:
        fuel_consumption = st.number_input("Fuel Consumption (L/100km) ⛽", min_value=0.0, value=8.0, step=0.1, key="fuel_consumption")
        kilometres = st.number_input("Kilometres 🛣️", min_value=0, value=50000, step=1000, key="kilometres")
        cylinders_in_engine = st.number_input("Cylinders in Engine 🔢", min_value=1, value=4, key="cylinders_in_engine")
        body_type = st.selectbox("Body Type 🚙", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"], key="body_type")
        doors = st.selectbox("Number of Doors 🚪", [2, 3, 4, 5], key="doors")

    # Load model only once and store in session state
    if 'model' not in st.session_state:
        file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
        st.session_state.model = load_model_from_drive(file_id)

    # Make prediction automatically based on inputs
    if st.session_state.model is not None:
        input_data = {
            'Year': st.session_state.year,
            'UsedOrNew': st.session_state.used_or_new,
            'Transmission': st.session_state.transmission,
            'Engine': st.session_state.engine,
            'DriveType': st.session_state.drive_type,
            'FuelType': st.session_state.fuel_type,
            'FuelConsumption': st.session_state.fuel_consumption,
            'Kilometres': st.session_state.kilometres,
            'CylindersinEngine': st.session_state.cylinders_in_engine,
            'BodyType': st.session_state.body_type,
            'Doors': st.session_state.doors
        }
        input_df = preprocess_input(input_data, st.session_state.model)

        try:
            prediction = st.session_state.model.predict(input_df)
            st.markdown(f"<div class='prediction-box'>Predicted Price: ${prediction[0]:,.2f}</div>", unsafe_allow_html=True)

            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': st.session_state.model.feature_names_in_,
                'importance': st.session_state.model.feature_importances_
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

            # Create and display the dashboard
            st.subheader("Vehicle Prices Dashboard")
            dashboard_fig = create_dashboard(df)
            st.plotly_chart(dashboard_fig)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Failed to load the model.")

if __name__ == "__main__":
    main()
