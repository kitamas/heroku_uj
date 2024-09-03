 # embedding text, metadata text, filename, title

import os
from openai import OpenAI

# Load environment variables from .env file
# env example: HUGGINGFACEHUB_API_TOKEN = <your-hugging-face-api-key>
# from dotenv import load_dotenv
# load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ['HUGGINGFACEHUB_API_TOKEN'] =  "my_key"
# os.environ['HF_TOKEN']  =  "my_key"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import chromadb
from chromadb.config import Settings

import jsonlines

# Replace with the actual path to your data file
data_file = './kumamoto_text.jsonl'

def text_embedding(text) -> None:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding
def insert_to_vectordb(data_file):
    client = chromadb.PersistentClient(path="chromadb")
    collection = client.get_or_create_collection(
        name="collection",
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
    )
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    with jsonlines.open(data_file) as reader:
        for entry in reader:
            text = entry.get("text", "") 
            documents.append(text)
            embeddings.append(text_embedding(text)) # Embedding text
            metadatas.append({
                "filename": entry.get("filename", "") ,
                "title": entry.get("title", "") ,
                "text": text  # Store the original text
            })
            ids.append(str(len(documents)))  # Simple sequential IDs
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("ChromaDB data inserted successfully!")
insert_to_vectordb(data_file)

if __name__ == "__main__":

    app.run()
#    app.debug = True