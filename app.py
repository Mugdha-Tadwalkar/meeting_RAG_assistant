# -------------------------------
# Imports
# -------------------------------
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

# -------------------------------
# Step 1 : Document Loading
# -------------------------------
loader = TextLoader("meeting.txt")
documents = loader.load()

# -------------------------------
# Step 2 : Chunking
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# -------------------------------
# Step 3 : Embedding Model
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -------------------------------
# Step 4 : Create Chroma DB
# -------------------------------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


# -------------------------------
# Step 5 : Groq LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------------
# Step 6 : Chat Loop
# -------------------------------
while True:

    query = input("\nAsk me a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("👋 Thanks for using Meeting Assistant!")
        break

    # Retrieve relevant chunks
    results = vector_store.similarity_search(
        query=query,
        k=3
    )

    if not results:
        print("No relevant information found.")
        continue

    context = ""


    for doc in results:


        context += doc.page_content + "\n\n"

 

    prompt = f"""
You are an AI Meeting Assistant.

Answer ONLY using the meeting notes provided below.

Rules:
- Use only the meeting notes.
- If the answer is available, answer clearly.
- If the answer is NOT available, reply exactly:
Sorry, I can only answer questions related to this meeting.
- Do not make up information.

Meeting Notes:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n==============================")
    print("AI Answer")
    print("==============================")
    print(response.content)