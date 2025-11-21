import streamlit as st
import cohere
import pinecone
import fitz  # PyMuPDF
import time
import os

from pinecone import Pinecone
os.environ["PINECONE_API_KEY"] = "fb805f6d-9378-4124-958e-0618bbff6030"

from pinecone import Pinecone, Index  # Import Pinecone and Index

# Get the API key and environment from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Get the API key from the env var
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")

# Check if the API key is set
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

# Initialize Pinecone using Pinecone()
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "myindex"

# Get the index using pc.Index()
index = pc.Index(index_name)


api_key = os.environ.get("fb805f6d-9378-4124-958e-0618bbff6030")

pc = Pinecone(api_key=api_key)

index_name = 'myindex'
index = pc.Index(index_name)

# Set your API keys
cohere_api_key = "YH63QkCnizd7e1nvuq3uceQAhuzdNJiwdgGvpABk"  # Replace with your Cohere API key
#openai.api_key = "your-openai-api-key"  # If needed, replace with your OpenAI API key

# Initialize clients
co = cohere.Client(cohere_api_key)


def split_into_chunks(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to generate a response based on retrieved texts

st.title("Interactive PDF QA Bot")
st.subheader("Upload your PDF document and ask questions!")

# File uploader and query input
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your question:")

# Helper function to split text into chunks
def split_into_chunks(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to generate response using Cohere LLM
def generate_response(retrieved_texts, query):
    context = " ".join(retrieved_texts)

    prompt = f"""
You are a helpful question-answering assistant.
Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = co.chat(
        model="c4ai-aya-expanse-8b",
        message=prompt,
        temperature=0.7,
    )

    return response.text



# Main logic triggered on button click
if st.button("Submit"):
    if uploaded_file is not None:
        # Step 1: Extract text from the uploaded PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([doc.load_page(page).get_text() for page in range(doc.page_count)])

        # Step 2: Split text into chunks and generate embeddings
        chunks = split_into_chunks(text)
        embeddings = []

        #st.info("Generating embeddings. This may take some time...")

        # Generate embeddings in batches to avoid rate limits
        for i in range(0, len(chunks), 10):
            batch = chunks[i:i + 10]
            response = co.embed(texts=batch, model="embed-english-v3.0")
            embeddings.extend(response.embeddings)
            time.sleep(5)  # Avoid hitting API rate limits

        # Step 3: Upsert embeddings into Pinecone index
        batch_size = 32
        for i in range(0, len(embeddings), batch_size):
            batch = [
                (f"chunk-{j}", embeddings[j], {"text": chunks[j]})
                for j in range(i, min(i + batch_size, len(embeddings)))
            ]
            index.upsert(vectors=batch)
        #st.success("PDF data has been successfully indexed!")

    else:
        st.warning("Please upload a PDF document.")

# Step 4: Query the index when the user submits a query
if query:
    query_embedding = co.embed(texts=[query], model="embed-english-v3.0").embeddings[0]
    response = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if response['matches']:
        # Retrieve the top matching chunks from Pinecone
        relevant_texts = [match['metadata']['text'] for match in response['matches']]

        # Generate the answer using RAG (retrieved context + LLM)
        generated_answer = generate_response(relevant_texts, query)

        # Display the answer and retrieved segments
        st.success(f"Answer: {generated_answer}")
        # for i, text in enumerate(relevant_texts):
        #     st.text_area(f"Retrieved Segment {i + 1}:", text, height=150)

    else:
        st.warning("No relevant results found.")
