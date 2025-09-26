# Build-a-Custom-Chatbot-Using-LangChain-and-ChatGPT

## Overview

This project is a **full AI-powered chatbot system** that demonstrates the complete workflow of building a conversational agent. The chatbot can:

- Extract and preprocess text data from multiple sources like HTML, PDFs, and plain text.  
- Augment the text and create embeddings for efficient knowledge retrieval.  
- Store and query information using a vector database.  
- Integrate with **LangChain** to manage conversation logic and memory.  
- Provide an interactive chat interface via **Streamlit**.  
- Deploy online for global accessibility.  

The main goal is to combine **natural language processing, vector-based retrieval, and an interactive UI** to build a functional chatbot that answers questions based on a custom knowledge base.

---

## Project Workflow

### 1. Text Extraction & Preprocessing
- Extracted text from HTML, PDFs, and plain text files.  
- Cleaned the text by removing unnecessary spaces, special characters, and formatting issues.  
- Split long text into **contextual chunks** suitable for embeddings.  

### 2. Data Augmentation & Tokenization
- Augmented text to increase dataset size and variety for better model understanding.  
- Tokenized text and ensured chunks are **semantically meaningful** for retrieval.  

### 3. Embeddings & Vector Store
- Converted text chunks into embeddings using **OpenAI `text-embedding-ada-002`**.  
- Stored embeddings in **Chroma**, enabling fast similarity-based retrieval.  

### 4. LangChain Backend
- Built a **ConversationalRetrievalChain** to integrate the vector store with a language model.  
- Used **ConversationBufferMemory** to track chat history and maintain context.  
- Developed a **FastAPI server** to handle queries from the frontend.  

### 5. Streamlit Frontend
- Created an interactive chat interface using **Streamlit**.  
- Captures user input and displays AI responses in real time.  
- Communicates with the backend to fetch relevant answers.  

### 6. Deployment
- Tested locally using `localhost` and optionally **ngrok** for temporary exposure.  
- Deployed to **Streamlit Cloud** for global accessibility.  
- Ensures anyone can interact with the chatbot from any device.  

---

## Tools & Technologies

- **Python 3.10+**  
- **OpenAI GPT models** (`text-embedding-ada-002`)  
- **LangChain** (conversation management & memory)  
- **Chroma** (vector store for embeddings)  
- **FastAPI** (backend API server)  
- **Streamlit** (frontend chat interface)  
- **Git & GitHub** (version control & deployment)  
- **Streamlit Cloud / ngrok** (deployment & testing)  

---

## Key Learnings

- Extracting, cleaning, and preprocessing text for AI tasks.  
- Data augmentation to improve model understanding.  
- Creating and querying vector embeddings efficiently.  
- Managing conversational context with LangChain memory.  
- Integrating backend and frontend for interactive AI applications.  
- Deploying AI projects to the cloud for public access.  

---

## Usage

1. Open the deployed **Streamlit app** or run locally.  
2. Type your query in the chat box.  
3. The AI retrieves relevant information and responds in context.  
4. Chat history is maintained for smoother conversations during the session.  
