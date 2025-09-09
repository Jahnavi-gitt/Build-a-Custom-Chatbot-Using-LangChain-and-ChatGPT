import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")

documents = []

with open('output.jsonl', 'r') as file:
    for line in file:
        
        documents.append(json.loads(line))


load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=api_key
)

vectorstore = Chroma(
    collection_name="Jlangchain_store",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

for doc in tqdm(documents):
    vectorstore.add_texts(texts=[doc["content"]])

query = input("Enter your question: ")

docs = vectorstore.similarity_search(query)

print("\nMost relevant result:")
print(docs[0].page_content)
