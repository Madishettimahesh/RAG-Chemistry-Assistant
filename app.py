import os
import gdown
import zipfile
import streamlit as st
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from groq import Groq

from rapidfuzz import process, fuzz


# ---------------------------------
# STREAMLIT TITLE
# ---------------------------------
st.title("🧪 MnSol ΔG Solvation Assistant (Hybrid + Auto Matcher)")


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
# LOAD DATASET
# ---------------------------------
df = pd.read_csv("new_dataset_with_predictions.csv")

solute_list = df["solute"].astype(str).unique().tolist()
solvent_list = df["solvent"].astype(str).unique().tolist()


# ---------------------------------
# AUTO MOLECULE MATCHER
# ---------------------------------
def match_name(user_input, choices):

    if not user_input:
        return user_input

    match, score, _ = process.extractOne(
        user_input,
        choices,
        scorer=fuzz.WRatio
    )

    if score > 75:
        return match

    return user_input


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

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = list(vectorstore.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    return vector_retriever, bm25_retriever


vector_retriever, bm25_retriever = create_retrievers(vectorstore)


# ---------------------------------
# HYBRID SEARCH
# ---------------------------------
def hybrid_search(query):

    semantic_docs = vector_retriever.invoke(f"query: {query}")
    keyword_docs = bm25_retriever.invoke(query)

    combined = {
        doc.page_content: doc
        for doc in semantic_docs + keyword_docs
    }

    return list(combined.values())[:5]


# ---------------------------------
# GROQ API
# ---------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------------------------
# USER INPUT
# ---------------------------------
st.subheader("Enter Molecule Details")

formula = st.text_input("Solute / Molecular Formula")
charge = st.text_input("Charge")
solvent = st.text_input("Solvent Name")

search_button = st.button("Predict DeltaG")


# ---------------------------------
# QUERY EXECUTION
# ---------------------------------
if search_button:

    if not formula or not solvent:
        st.warning("Please enter Solute and Solvent")
        st.stop()

    # Auto molecule correction
    matched_solute = match_name(formula, solute_list)
    matched_solvent = match_name(solvent, solvent_list)

    st.info(
        f"Matched Solute: {matched_solute} | "
        f"Matched Solvent: {matched_solvent}"
    )

    query = f"""
    solute: {matched_solute},
    charge: {charge},
    solvent: {matched_solvent}
    """

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
- Use ONLY dataset context.
- Do NOT use external chemistry knowledge.
- Extract DeltaG exactly from dataset.
- Match solute and solvent carefully.
- Keep answer short.

Output format:

DeltaG: <value>
Explanation: <short dataset explanation>
"""
                },
                {
                    "role": "user",
                    "content": f"""
Dataset Context:
{context}

Solute: {matched_solute}
Charge: {charge}
Solvent: {matched_solvent}
"""
                }
            ]
        )

        st.success("Result")
        st.write(completion.choices[0].message.content)
