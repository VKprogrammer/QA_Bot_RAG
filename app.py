import streamlit as st
import cohere
from pinecone import Pinecone
import fitz  # PyMuPDF
import time

# Set your API keys
PINECONE_API_KEY = "fb805f6d-9378-4124-958e-0618bbff6030"
cohere_api_key = "YH63QkCnizd7e1nvuq3uceQAhuzdNJiwdgGvpABk"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index = pc.Index("myindex")
co = cohere.Client(cohere_api_key)

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

# Streamlit UI
st.title("Interactive PDF QA Bot")
st.subheader("Upload your PDF document and ask questions!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Main logic for uploading and indexing PDF
if st.button("Upload and Index PDF"):
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Delete existing records ONLY when uploading a new file
            index.delete(delete_all=True)
            
            # Step 1: Extract text from the uploaded PDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join([doc.load_page(page).get_text() for page in range(doc.page_count)])
            
            # Step 2: Split text into chunks and generate embeddings
            chunks = split_into_chunks(text)
            embeddings = []
            
            st.info("Generating embeddings. This may take some time...")
            
            # Generate embeddings in batches to avoid rate limits
            for i in range(0, len(chunks), 10):
                batch = chunks[i:i + 10]
                response = co.embed(texts=batch, model="embed-english-v2.0")
                embeddings.extend(response.embeddings)
                time.sleep(1)  # Reduced sleep time
            
            # Step 3: Upsert embeddings into Pinecone index
            batch_size = 32
            for i in range(0, len(embeddings), batch_size):
                batch = [
                    (f"chunk-{j}", embeddings[j], {"text": chunks[j]})
                    for j in range(i, min(i + batch_size, len(embeddings)))
                ]
                index.upsert(vectors=batch)
            
            st.success(f"PDF successfully indexed! {len(chunks)} chunks added to the database.")
    else:
        st.warning("Please upload a PDF document first.")

# Query section
st.markdown("---")
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        # Check if there are records in the index
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            st.error("No documents indexed yet. Please upload and index a PDF first.")
        else:
            with st.spinner("Searching for answer..."):
                # Generate query embedding
                query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
                
                # Query the index
                response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
                
                if response['matches']:
                    # Retrieve the top matching chunks from Pinecone
                    relevant_texts = [match['metadata']['text'] for match in response['matches']]
                    
                    # Generate the answer using RAG
                    generated_answer = generate_response(relevant_texts, query)
                    
                    # Display the answer
                    st.success("Answer:")
                    st.write(generated_answer)
                    
                    # Optionally show retrieved segments
                    with st.expander("View Retrieved Context"):
                        for i, text in enumerate(relevant_texts):
                            st.text_area(f"Segment {i + 1}:", text, height=150)
                else:
                    st.warning("No relevant results found in the document.")
    else:
        st.warning("Please enter a question.")
