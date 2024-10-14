import streamlit as st
import pickle
import requests
from io import BytesIO

def download_model(url):
    response = requests.get(url)
    content = response.content
    
    # Check the first few bytes of the content
    print("First 100 bytes of content:", content[:100])
    
    if content.startswith(b'<'):
        raise ValueError("Received HTML instead of model file. Check your download link.")
    
    try:
        return pickle.loads(content)
    except pickle.UnpicklingError:
        raise ValueError("The downloaded content is not a valid pickle file.")

def main():
    st.title("My Streamlit App")

    model_url = "https://drive.google.com/uc?id=1tm26hgFqH6jgquktn3ZosbTuRV_Yoepq"  # Replace with your actual file ID
  
    
    try:
        model = download_model(model_url)
        st.success("Model loaded successfully!")
        
        # Your model usage code here
        
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        if "Received HTML" in str(e):
            st.write("It looks like the download link might be incorrect. Please check your Google Drive sharing settings and URL.")

if __name__ == "__main__":
    main()
