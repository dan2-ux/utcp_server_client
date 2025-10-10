from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db1"
add_documents = not os.path.exists(db_location) 

# ---------------------------
# Load documents from CSV & JSON
# ---------------------------
def load_csv_json():
    documents = []
    ids = []

    # Load JSON (if needed, just stored as raw string)
#    with open("data.json") as f:
#        json_data = json.load(f)
#    documents.append(Document(
#        page_content=json.dumps(json_data),
#        metadata={"source": "data.json"}
#    ))
#    ids.append("json_0")

    # Load CSV
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    for i, row in df.iterrows():
        documents.append(Document(
            page_content=f"{row['prefer_name']} {row['type']} {row['path']}",
            metadata={"value_define": row["value_define"], "source": "data.csv"}
        ))
        ids.append(f"csv_{i}")

    return documents, ids

# ---------------------------
# Build vector store
# ---------------------------
vector_store = Chroma(
    collection_name="api_reader",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    csv_json_docs, csv_json_ids = load_csv_json()
    all_docs = csv_json_docs
    all_ids = csv_json_ids
    if all_docs:
        print(f"==== Adding {len(all_docs)} documents to Chroma ====")
        vector_store.add_documents(documents=all_docs, ids=all_ids)

# ---------------------------
# Create retriever
# ---------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
