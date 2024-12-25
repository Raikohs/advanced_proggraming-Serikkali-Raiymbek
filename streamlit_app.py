import os
import streamlit as st
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Constants
LLM_MODEL_NAME = "llama3.2"
CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "rag_collection_demo_1"
COLLECTION_DESCRIPTION = "A collection for RAG with Ollama - Demo1"

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Embedding Function Class
class CustomEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.embedding_model.embed_documents(input)

# Set up embedding model
embedding_function = CustomEmbeddingFunction(
    OllamaEmbeddings(
        model=LLM_MODEL_NAME,
        base_url="http://localhost:11434"
    )
)

# Create or retrieve collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": COLLECTION_DESCRIPTION},
    embedding_function=embedding_function
)

# Function to add documents to the collection
def add_documents_to_collection(doc_texts, doc_ids):
    try:
        collection.add(documents=doc_texts, ids=doc_ids)
    except Exception as e:
        raise ValueError(f"Failed to add documents: {str(e)}")

# Query ChromaDB
def query_chromadb(query, top_k=1):
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        return results.get("documents", []), results.get("metadatas", [])
    except Exception as e:
        return [], []

# Query LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    return llm.invoke(prompt)

# RAG Pipeline
def run_rag_pipeline(user_query):
    retrieved_docs, _ = query_chromadb(user_query)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    prompt = f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"
    print("[DEBUG] Generated Prompt:", prompt)
    return query_ollama(prompt)

# Streamlit App UI
st.title("Enhanced RAG Pipeline with Ollama")

# Sidebar with a simple description
st.sidebar.header("Navigation")
menu_options = ["View Documents", "Add Document", "Ask a Question"]
user_choice = st.sidebar.radio("Select an option", menu_options)

# View Documents
if user_choice == "View Documents":
    st.header("Documents in the Collection")
    docs = collection.get(include=["documents", "metadatas"])
    if docs and docs.get("documents"):
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            st.markdown(f"**Content:** {doc}")
            st.markdown(f"**Metadata:** {meta}")
            st.markdown("---")
    else:
        st.write("No documents available.")

# Add Document
elif user_choice == "Add Document":
    st.header("Add a New Document")
    doc_content = st.text_area("Enter the document content")
    doc_identifier = st.text_input("Enter a unique document ID")
    if st.button("Add Document"):
        if doc_content and doc_identifier:
            try:
                add_documents_to_collection([doc_content], [doc_identifier])
                st.success("Document successfully added!")
            except ValueError as e:
                st.error(str(e))
        else:
            st.error("Both document content and ID are required.")

# Ask a Question
elif user_choice == "Ask a Question":
    st.header("Ask a Question to the Knowledge Base")
    question = st.text_area("What do you want to know?")
    if st.button("Submit Query"):
        if question:
            response = run_rag_pipeline(question)
            st.subheader("Response from the Knowledge Base")
            st.write(response)
        else:
            st.error("Please enter a question to get a response.")
