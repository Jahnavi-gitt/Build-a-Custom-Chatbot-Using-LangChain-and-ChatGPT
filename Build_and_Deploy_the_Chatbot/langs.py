from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

model_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=api_key
)

vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(openai_api_key=api_key),
    chain_type="map_reduce",
    retriever=retriever,
    memory=memory
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using LangChain's Runnable interfaces",
)

add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
