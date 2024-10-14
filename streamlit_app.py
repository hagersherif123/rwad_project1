import streamlit as st
import requests
import pickle
from io import BytesIO

# Add this import
import sklearn

def download_model(url):
    response = requests.get(url)
    return BytesIO(response.content)

def main():
    st.title("My Streamlit App")

    model_url = "https://drive.google.com/uc?id=1tm26hgFqH6jgquktn3ZosbTuRV_Yoepq"  # Make sure this is your correct file ID

    
    try:
        model_file = download_model(model_url)
        model = pickle.load(model_file)
        st.success("Model loaded successfully!")
        
        # Your model usage code here
        # For example:
        # if isinstance(model, sklearn.base.BaseEstimator):
        #     st.write("Loaded a scikit-learn model")
        
    except ModuleNotFoundError as e:
        st.error(f"Error: {str(e)}")
        st.write("It looks like you're missing some required libraries. Please install them using:")
        st.code("pip install scikit-learn")
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

if __name__ == "__main__":
    main()
