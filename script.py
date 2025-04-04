# 🧠 STEP 2: Import libraries
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
import time

# 📍 CONFIG
UNIVERSITY_URL = "https://www.sittechno.org/"  # Full site scrape
GOOGLE_API_KEY = "AIzaSyATsK-lvEoDkK0xhD3o7BU44O622ZoxTHo"  # Replace with your actual key

# 🔎 STEP 3: Full Site Scraper
def scrape_entire_site(start_url, max_pages=50, delay=1):
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
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, "html.parser")
            visited.add(url)

            # Extract visible text
            page_text = soup.get_text(separator="\n", strip=True)
            text_snippet = page_text[:300].replace('\n', ' ')
            print(f"✅ Scraped: {url}\n📄 Snippet: {text_snippet}\n{'-'*80}")

            all_text.append(page_text)

            # Find internal links
            for link_tag in soup.find_all("a", href=True):
                href = link_tag['href']
                full_url = urljoin(url, href)
                if full_url not in visited and urlparse(full_url).netloc == domain:
                    to_visit.append(full_url)

            time.sleep(delay)

        except Exception as e:
            print(f"⚠️ Skipping {url}: {e}")
            continue

    print(f"🎉 Done. Scraped {len(visited)} pages from {domain}")
    return "\n\n".join(all_text)

# 🧩 STEP 4: Chunk + Embed
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_text(text)

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return model, embeddings

# 🔎 STEP 5: FAISS Setup
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index

#  STEP 6: Load Gemini (via LangChain)
def load_gemini_model():
    # Ensure the Google API key is valid
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is not set.")
        
    return GoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_output_tokens=512
    )

#  LangChain memory + prompt
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

# 🧠 STEP 7: Chat Function
def query_bot(question, embedder, index, chunks, llm):
    q_embed = embedder.encode([question])
    _, I = index.search(np.array(q_embed), k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = prompt_template.format(context=context, chat_history=memory.buffer, question=question)
    if "greeting" in question.lower():
        response = random.choice(["Hello! How can I assist you today?", "Hi there! What can I help you with?"])
    elif "farewell" in question.lower():
        response = random.choice(["Goodbye! Have a great day!", "See you later! Take care!"])
    elif "university_info" in question.lower():
        response = random.choice(["The university is known for its excellent programs and research opportunities.", "Our university offers a variety of courses and has a vibrant campus life."])
    elif "admissions_info" in question.lower():
        response = random.choice(["You can find the admission requirements on our website under the admissions section.", "To apply for admission, please visit our admissions page for detailed instructions."])
    elif "courses_info" in question.lower():
        response = random.choice(["We offer a variety of courses across different fields. Please check our courses page for more details.", "Our university has programs in engineering, arts, sciences, and more."])
    else:
        response = random.choice(["I'm sorry, I didn't quite catch that. Can you please rephrase your question?", "I’m not sure how to respond to that. Could you ask something else?"])
    memory.save_context({"question": question}, {"output": response})
    return response

# 🚀 STEP 8: RUN THE BOT
text = scrape_entire_site(UNIVERSITY_URL)
chunks = chunk_text(text)
embedder, embeddings = embed_chunks(chunks)
index = build_faiss_index(embeddings)
llm = load_gemini_model()

print("✅ Chatbot ready! Type your question or 'exit' to quit.\n")

# 💬 Interactive Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    response = query_bot(user_input, embedder, index, chunks, llm)
    print("Bot:", response, "\n")
