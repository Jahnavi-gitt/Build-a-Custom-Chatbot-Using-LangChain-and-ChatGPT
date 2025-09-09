from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

retriever=vectorstore.as_retriever()

qa=RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), 
    chain_type="map_reduce",
    retriever=retriever)

query=print("Enter your question: ")
response=qa.run(query)

print(response)