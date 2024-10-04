import requests
import gradio as gr

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/trans"

# Function to call the FastAPI backend
def predict(user_input):
    # Prepare the data to send to the FastAPI API
    payload = {"query": user_input}
    
    # Make a request to the FastAPI backend
    response = requests.post(API_URL, json=payload)
    
    # Get the response JSON
    result = response.json()
    
    # Extract the answer 
    return " ".join(result["response"])
    

# Launch the Gradio interface
if __name__ == "__main__":
    gr.Interface(predict,
                 inputs=['textbox'],
                 outputs=['text']).launch(share=True)
