from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from urllib.parse import urljoin, urlparse
import os
import logging
import time

app = Flask(__name__, static_folder='sit-web - Copy/New SIT WEBPAGE')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the chatbot components
GOOGLE_API_KEY = "AIzaSyATsK-lvEoDkK0xhD3o7BU44O622ZoxTHo"
UNIVERSITY_URL = "https://www.sittechno.org/"

@app.route('/')
def serve_base():
    return send_file('base.html')

@app.route('/aboutus')
def serve_aboutus():
    return send_file('sit-web - Copy/New SIT WEBPAGE/aboutus.html')

@app.route('/academics')
def serve_academics():
    return send_file('sit-web - Copy/New SIT WEBPAGE/academics.html')

@app.route('/admission')
def serve_admission():
    return send_file('sit-web - Copy/New SIT WEBPAGE/admission.html')

def scrape_entire_site(start_url, max_pages=50):
    visited = set()
    to_visit = [start_url]
    all_text = []
    domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or urlparse(url).netloc != domain:
            continue
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            visited.add(url)
            all_text.append(soup.get_text(separator="\n", strip=True))
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            continue

    return "\n\n".join(all_text)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_text(text)

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return model, embeddings

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index

def load_gemini_model():
    return GoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_output_tokens=512
    )

prompt_template = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
    You are a helpful assistant for university-related queries. Answer based on the context and previous conversation.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer in a helpful tone:
    """
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

def query_bot(question, embedder, index, chunks, llm):
    q_embed = embedder.encode([question])
    _, I = index.search(np.array(q_embed), k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = prompt_template.format(context=context, chat_history=memory.buffer, question=question)
    response = llm.invoke(prompt)
    memory.save_context({"question": question}, {"output": response})
    return response

# Initialize chatbot
text = scrape_entire_site(UNIVERSITY_URL)
chunks = chunk_text(text)
embedder, embeddings = embed_chunks(chunks)  # Corrected line
index = build_faiss_index(embeddings)  # Ensure this function is called correctly
llm = load_gemini_model()

@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()
    try:
        data = request.json
        user_message = data.get("question", "")
        bot_response = query_bot(user_message, embedder, index, chunks, llm)
        logging.debug(f"Response time: {time.time() - start_time} seconds")
        return jsonify({"response": bot_response})
    except Exception as e:
        logging.error(f"Error in chat route: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
