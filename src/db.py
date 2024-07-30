# db.py
from pymongo import MongoClient
import os

client = MongoClient(os.getenv('MONGODB_URI'))
db = client['miningniti']
collection = db['pdf_texts']

def store_pdf_text(pdf_name: str, text: str):
    collection.insert_one({"pdf_name": pdf_name, "text": text})

def search_pdf_text(query: str):
    results = collection.find({"text": {"$regex": query, "$options": "i"}})
    return list(results)

