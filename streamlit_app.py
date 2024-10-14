import streamlit as st
import pickle
import requests
from io import BytesIO

# Function to download model from Google Drive
def download_model(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

# Main Streamlit app
def main():
    st.title("My Streamlit App")

    # Download model
    model_url = "https://drive.google.com/file/d/1tm26hgFqH6jgquktn3ZosbTuRV_Yoepq/view?usp=sharing"  # Replace with your actual file ID
    
    try:
        model = download_model(model_url)
        st.success("Model loaded successfully!")

        # Your Streamlit app logic here
        # Use the model for predictions, etc.
        # For example:
        # user_input = st.text_input("Enter some text for prediction")
        # if user_input:
        #     prediction = model.predict([user_input])
        #     st.write(f"Prediction: {prediction}")

    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

if __name__ == "__main__":
    main()
