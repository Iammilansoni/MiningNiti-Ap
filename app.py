# app.py
import os
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from src.model import MyChatBot
from src.utils import extract_text_from_pdf
from src.db import store_pdf_text, search_pdf_text

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Load configuration from YAML file
try:
    with open("conf/variables.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
except yaml.YAMLError as exc:
    print(f"Error loading YAML configuration: {exc}")
    raise

# Initialize an instance of MyChatBot and set the prompt
try:
    model = MyChatBot(api_key=api_key, temperature=config['temperature'])
    model.set_prompt(template=config['MiningNitiTemplate'], input_variables=["input"])
except KeyError as e:
    print(f"Configuration key error: {e}")
    raise

app = FastAPI()

# CORS settings
origins = [
    "https://miningniti.vercel.app",
    "https://miningniti.vercel.app/chatting",
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/chatting",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://miningniti.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    input_query: str
    source: str  # "database", "internet", or "both"

@app.post("/chat")
def run(item: Item, model: MyChatBot = Depends(lambda: model)):
    """
    Endpoint to handle chat requests.
    """
    try:
        if item.source == "database":
            results = search_pdf_text(item.input_query)
            response = "\n".join([f"PDF: {res['pdf_name']}\nText: {res['text']}" for res in results])
        elif item.source == "internet":
            response = model.run(input=item.input_query)
        else:  # both
            results = search_pdf_text(item.input_query)
            db_response = "\n".join([f"PDF: {res['pdf_name']}\nText: {res['text']}" for res in results])
            internet_response = model.run(input=item.input_query)
            response = f"Database Results:\n{db_response}\n\nInternet Results:\n{internet_response}"
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_text_from_pdf")
async def extract_text_from_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to handle PDF uploads and text extraction.
    """
    try:
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        text = extract_text_from_pdf(file_location)
        store_pdf_text(file.filename, text)
        os.remove(file_location)
        return JSONResponse(content={"message": "Text extracted and stored successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
