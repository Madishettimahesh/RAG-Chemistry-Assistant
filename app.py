import os
import gdown
import zipfile
import streamlit as st
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from groq import Groq


# ---------------------------------
# STREAMLIT TITLE
# ---------------------------------
st.title("🧪 LLM MnSol ΔG Solvation Assistant (Hybrid Search)")


# ---------------------------------
# DOWNLOAD FAISS INDEX
# ---------------------------------
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"

if not os.path.exists(INDEX_FOLDER):

    with st.spinner("Downloading vector database..."):

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")


# ---------------------------------
# LOAD CSV
# ---------------------------------
df = pd.read_csv("new_dataset_with_predictions.csv")


# ---------------------------------
# LOAD VECTOR STORE
# ---------------------------------
@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectorstore = FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


vectorstore = load_vectorstore()


# ---------------------------------
# CREATE HYBRID RETRIEVERS
# ---------------------------------
@st.cache_resource
def create_retrievers(vectorstore):

    # Semantic search (FAISS)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Keyword search (BM25)
    docs = list(vectorstore.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    return vector_retriever, bm25_retriever


vector_retriever, bm25_retriever = create_retrievers(vectorstore)


# ---------------------------------
# HYBRID SEARCH FUNCTION ⭐
# ---------------------------------
def hybrid_search(query):

    semantic_results = vector_retriever.invoke(f"query: {query}")
    keyword_results = bm25_retriever.invoke(query)

    # Remove duplicates
    combined_docs = {
        doc.page_content: doc
        for doc in semantic_results + keyword_results
    }

    return list(combined_docs.values())[:5]


# ---------------------------------
# GROQ API
# ---------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------------------------
# USER QUERY
# ---------------------------------
query = st.text_input("Ask your solvation question")


# ---------------------------------
# QUERY EXECUTION
# ---------------------------------
if query:

    with st.spinner("Searching MnSol dataset..."):

        docs = hybrid_search(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a dataset-driven chemistry assistant.

RULES:
- Answer ONLY using dataset context.
- Do NOT use external chemistry knowledge.
- Extract DeltaG directly from dataset.
- Keep answer short.

Output format:

DeltaG: <value>

Explanation: <short dataset-based explanation>
"""
                },
                {
                    "role": "user",
                    "content": f"""
Dataset Context:
{context}

User Question:
{query}
"""
                }
            ]
        )

        st.success("Result")
        st.write(completion.choices[0].message.content)
