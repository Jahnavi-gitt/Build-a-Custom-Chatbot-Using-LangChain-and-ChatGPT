from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import json

loader = TextLoader("Computer Configuration.txt")
docs = loader.load()

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=tiktoken_len,
    separators = ['\n\n', '\n', ' ', ''] #default
)

chunks = splitter.split_documents(docs)

with open("output.jsonl", "w") as f:
    for chunk in chunks:
        f.write(json.dumps({"content": chunk.page_content}) + "\n")

print("Done! Your text is now in output.jsonl")

