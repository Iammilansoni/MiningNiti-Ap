#app.py 
import os
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from src.model import MyChatBot

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
    allow_headers=["*"],  # Adjust this line based on the headers your frontend sends
)

class Item(BaseModel):
    input_query: str

# In-memory storage for chat history
chat_history = []
@app.post("/chat")
def run(item: Item, model: MyChatBot = Depends(lambda: model)):
    """
    Endpoint to handle chat requests.
    """
    try:
        response = model.run(input=item.input_query)
        chat_history.append({"message": item.input_query, "response": response})
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    """
    Endpoint to retrieve chat history.
    """
    return JSONResponse(content={"history": chat_history})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)

