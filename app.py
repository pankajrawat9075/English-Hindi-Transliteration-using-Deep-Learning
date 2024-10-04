import gradio as gr
from inference.language import Language
from inference.utility import Encoder, Decoder, encoderBlock, decoderBlock, MultiHeadAttention, Head, FeedForward
from inference.transformer import generate

# Function to call the FastAPI backend
def predict(user_input):
    # Prepare the data to send to the FastAPI API
    input = user_input.split(" ")

    result = generate(input)
    
    # Extract the answer 
    return " ".join(result)
    

# Launch the Gradio interface
if __name__ == "__main__":
    gr.Interface(predict,
                 inputs=gr.Textbox(placeholder="Your Hinglish text"),
                 outputs=gr.Textbox(placeholder="Output Hindi text"),
                 description="A English to Hindi Transliteration app",
                 examples=["namaste aapko", "kese ho aap, sab badiya"]).launch(share=False)
