import os
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from html_to_text import extract_text_from_html, split_text_into_chunks


def get_embeddings_for_chunks(chunk_list):
    """Guaranteed embedding generation with automatic fallback"""
    
    api_key = "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"  
    model = "openai/text-embedding-ada-002"  
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "input": chunk_list  
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return [item["embedding"] for item in response.json()["data"]]
    
    except Exception as api_error:
        print("API failed: " + str(api_error)[:200])

    model = SentenceTransformer('all-MiniLM-L6-v2')  
    return model.encode(chunk_list).tolist()

def get_chunks_and_embeddings_dict(directory, max_tokens=4096):
    chunks_dict = {}
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".html"):
                file_path = os.path.join(dirpath, filename)
                try:
                    text = extract_text_from_html(file_path)
                    chunks = split_text_into_chunks(text, max_tokens)
                    if chunks:
                        embeddings = get_embeddings_for_chunks(chunks)
                        chunks_dict[file_path] = list(zip(chunks, embeddings))
                except Exception as e:
                    print(f"Skipped {filename}: {str(e)[:100]}")
    return chunks_dict

if __name__ == "__main__":
    folder_path = "C:/Users/Jahnavi/OneDrive/Desktop/I"
    
    print("\nPROCESSING FILES...")
    result_dict = get_chunks_and_embeddings_dict(folder_path)

    data=[]

    for path, pairs in result_dict.items():
        for chunk, emb in pairs:
            data.append((path, chunk, emb))

    df = pd.DataFrame(data, columns=["file_path", "text", "embedding"])
    print("\nRESULTS:")
    print(df)  
    print("Total chunks processed:", len(df))

    df.to_pickle("my_chunks_embeddings.pkl")
    
    print(" Saved to my_chunks_embeddings.pkl")