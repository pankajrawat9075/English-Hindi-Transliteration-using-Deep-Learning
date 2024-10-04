from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.language import Language
from inference.utility import Encoder, Decoder, encoderBlock, decoderBlock, MultiHeadAttention, Head, FeedForward
from inference.transformer import generate
from typing import List
import uvicorn


# Initialize FastAPI app
app = FastAPI()

# Create a request model to define the input for the transliteration pipeline
class TransRequest(BaseModel):
    query: str

# Create a response model to define the output of the RAG pipeline
class TransResponse(BaseModel):
    response: List[str]

# Define a FastAPI endpoint for transliteration pipeline
@app.post("/trans", response_model=TransResponse)
async def get_transliteration(request: TransRequest):
    try:
        # Call the RAG pipeline function with the query
        input = request.query.split(" ")
        result = generate(input)

        return TransResponse(
            response=result
        )
    except Exception as e:
        # In case of an error, return an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
