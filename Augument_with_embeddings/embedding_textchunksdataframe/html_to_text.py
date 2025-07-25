from bs4 import BeautifulSoup
import tiktoken
 
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')

    for tag in soup(['script', 'style']):
        tag.decompose()

    return soup.get_text(separator=' ', strip=True)

def split_text_into_chunks(text,max_token=300):
    encoding= tiktoken.encoding_for_model("gpt-3.5-turbo")
    words=text.split()
    chunks=[]
    curr_chunk=[]
    curr_token_count=0
    for word in words:
        word_token_count=len(encoding.encode(word))
        if curr_token_count+word_token_count > max_token:
            chunks.append(" ".join(curr_chunk))
            curr_chunk=[word]
            curr_token_count=word_token_count
        else:
            curr_chunk.append(word)
            curr_token_count+=word_token_count

    if curr_chunk:
            chunks.append(" ".join(curr_chunk))

    return chunks

text = extract_text_from_html("C:/Users/Jahnavi/OneDrive/Desktop/I/res.html")
chunks = split_text_into_chunks(text, max_token=10)
print(chunks)
