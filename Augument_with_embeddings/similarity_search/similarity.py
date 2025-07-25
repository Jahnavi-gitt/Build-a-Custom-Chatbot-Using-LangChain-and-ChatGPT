from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from embedding import get_embeddings_for_chunks


def search_similar_strings(query,df,limit=5):
    query_embedding= get_embeddings_for_chunks([query])
    all_embeddings=np.array(df["embedding"].tolist())
    similarities=cosine_similarity(query_embedding,all_embeddings)[0]
    
    df["similarity"] = similarities
    top_matches = df.sort_values("similarity", ascending=False).head(limit)
    
    return top_matches[["text", "similarity"]]

def query_message(query,df,limit=5):
    top_chunks=search_similar_strings(query,df,limit)
    intro = "Use the following information to answer the question:\n\n"
    context= "\n\n".join(top_chunks["text"].tolist())
    return intro + context + "\n\nQuestion: " + query + "\nAnswer:"

import openai
import time

openai.api_key="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"
openai.api_base="https://openrouter.ai/api/v1"

def generate_chatgpt_response(prompt):
    try:
        response=openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()
    
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please wait for a bit and try.")
        time.sleep(30)
        return None
    
    except Exception as e:
        print(f"Error message: {e}")
        time.sleep(30)
        return None

if __name__ == "__main__":
    df = pd.read_pickle("my_chunks_embeddings.pkl")

    print("Welcome to ChatGPT CLI! Type 'exit' to quit.")
    while True:
        user_input = input("Ask ChatGPT: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        prompt = query_message(user_input, df)

        chatgpt_response = generate_chatgpt_response(prompt)
        
        if chatgpt_response:
            print(f"\nChatGPT says: {chatgpt_response}\n")
        else:
            print("\nNo response from ChatGPT.\n")
