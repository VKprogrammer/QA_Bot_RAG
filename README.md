# QA_Bot_RAG
Technologies Used:

Streamlit: Interactive web app interface
Cohere API: Text embedding and generation
Pinecone: Vector database for efficient similarity search
PyMuPDF: PDF text extraction
LocalTunnel: For exposing the app to the web during development
📖 Overview
This project implements a Retrieval-Augmented Generation (RAG)-based Q&A bot that allows users to:

Upload PDF documents.
Ask questions based on the document’s content.
Retrieve relevant document segments using Pinecone for similarity search.
Generate human-like answers with the Cohere LLM based on the retrieved segments.
The application is built with Streamlit for the frontend, and Cohere API for both embedding and text generation.

🛠️ Features
Upload PDF documents: Extract content dynamically from any PDF.
Ask questions: Type your question in a text box.
Real-time similarity search: Use Pinecone to search through the document embeddings.
LLM-based answer generation: Get context-aware answers using Cohere’s xlarge model.
Interactive interface: Simple, responsive UI built with Streamlit.
⚙️ Project Setup
Follow these steps to set up the project locally.

1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/llm-rag-streamlit-bot.git
cd llm-rag-streamlit-bot
2. Create a Virtual Environment (Optional but Recommended)
bash
Copy code
python3 -m venv env
source env/bin/activate  # For Linux/macOS
env\Scripts\activate  # For Windows
3. Install Required Packages
bash
Copy code
pip install streamlit cohere pinecone-client PyMuPDF
4. Set Up API Keys
Make sure you have:

Cohere API key: Get it from Cohere.
Pinecone API key: Get it from Pinecone.
Create a .env file in the project directory to store your API keys:

makefile
Copy code
COHERE_API_KEY=your-cohere-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment  # Example: us-east1-gcp
5. Initialize Pinecone Index
Before running the app, ensure that you have:

Created a Pinecone index from the Pinecone dashboard.
Specified the index name correctly in the app code (index_name = 'myindex').
🚀 Running the Application
Start the Streamlit app:

bash
Copy code
streamlit run app.py
Expose the app to the web (for development):

bash
Copy code
npx localtunnel --port 8501
Access the app:
You will receive a URL from LocalTunnel (e.g., https://funny-pianos.loca.lt). Open it in your browser to use the app.

🖥️ Application Flow
Upload a PDF:

Use the file uploader to upload any PDF document.
The app extracts text using PyMuPDF.
Ask a Question:

Enter your question in the text input box.
The app generates Cohere embeddings for both the document and the query.
Similarity Search with Pinecone:

The top matching document segment is retrieved using Pinecone’s vector database.
LLM-Based Answer Generation:

The matching text segment is used to generate an answer with the Cohere LLM.
Display:

The app shows both the retrieved segment and the generated answer.
