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


# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("🧪 MnSol ΔG Solvation Assistant")


# ------------------------------------------------
# DOWNLOAD FAISS INDEX
# ------------------------------------------------
FILE_ID = "1gUKTTKNjOqI2jP3I6bGVSmFtb3JD7jn2"
ZIP_FILE = "mnsol_faiss_index.zip"
INDEX_FOLDER = "mnsol_faiss_index"

if not os.path.exists(INDEX_FOLDER):

    with st.spinner("Downloading vector database..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")


# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------
df = pd.read_csv("new_dataset_with_predictions.csv")

solute_list = df["SoluteName"].astype(str).unique().tolist()
solvent_list = df["Solvent"].astype(str).unique().tolist()

# Formula → Solute mapping
formula_to_solute = dict(
    zip(
        df["Formula"].astype(str).str.lower(),
        df["SoluteName"]
    )
)


# ------------------------------------------------
# AUTO MATCH FUNCTIONS
# ------------------------------------------------
def resolve_solute(user_input):

    key = user_input.lower().strip()

    # Formula match
    if key in formula_to_solute:
        return formula_to_solute[key]

    # Fuzzy match
    match, score, _ = process.extractOne(
        user_input,
        solute_list,
        scorer=fuzz.WRatio
    )

    if score > 75:
        return match

    return user_input


def match_solvent(user_input):

    match, score, _ = process.extractOne(
        user_input,
        solvent_list,
        scorer=fuzz.WRatio
    )

    if score > 75:
        return match

    return user_input


# ------------------------------------------------
# EXACT DATASET LOOKUP ⭐
# ------------------------------------------------
def exact_lookup(solute, solvent, charge):

    result = df[
        (df["SoluteName"].str.lower() == solute.lower()) &
        (df["Solvent"].str.lower() == solvent.lower())
    ]

    if charge != "":
        result = result[
            result["Charge"].astype(str) == str(charge)
        ]

    if len(result) > 0:
        row = result.iloc[0]

        return row["Predicted_DeltaGsolv"]

    return None


# ------------------------------------------------
# LOAD VECTOR STORE
# ------------------------------------------------
@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    return FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()


# ------------------------------------------------
# HYBRID RETRIEVERS
# ------------------------------------------------
@st.cache_resource
def create_retrievers(vectorstore):

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = list(vectorstore.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    return vector_retriever, bm25


vector_retriever, bm25_retriever = create_retrievers(vectorstore)


def hybrid_search(query):

    semantic = vector_retriever.invoke(query)
    keyword = bm25_retriever.invoke(query)

    combined = {
        doc.page_content: doc
        for doc in semantic + keyword
    }

    return list(combined.values())[:5]


# ------------------------------------------------
# GROQ API
# ------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
st.subheader("Enter Molecule Details")

formula = st.text_input("Formula or Solute Name")
charge = st.text_input("Charge")
solvent = st.text_input("Solvent")

search = st.button("Get DeltaG")


# ------------------------------------------------
# EXECUTION
# ------------------------------------------------
if search:

    if not formula or not solvent:
        st.warning("Please enter required inputs")
        st.stop()

    matched_solute = resolve_solute(formula)
    matched_solvent = match_solvent(solvent)

    st.info(
        f"Matched Solute: {matched_solute} | "
        f"Matched Solvent: {matched_solvent}"
    )

    # ✅ EXACT LOOKUP FIRST
    deltag = exact_lookup(
        matched_solute,
        matched_solvent,
        charge
    )

    if deltag is not None:

        st.success("Exact Dataset Match Found ✅")

        st.write(f"""
### DeltaG: {deltag}

Explanation: Retrieved directly from MnSol dataset.
""")

        st.stop()

    # ------------------------------------------------
    # HYBRID RAG FALLBACK
    # ------------------------------------------------
    query = f"""
    SoluteName {matched_solute}
    Solvent {matched_solvent}
    Charge {charge}
    solvation free energy DeltaG
    """

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
Answer ONLY using dataset context.
Extract DeltaG value if present.
Keep answer short.
"""
            },
            {
                "role": "user",
                "content": f"""
Dataset Context:
{context}
"""
            }
        ]
    )

    st.write(completion.choices[0].message.content)
