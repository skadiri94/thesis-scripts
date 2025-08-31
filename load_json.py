import os
import json
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


def load_excel_as_documents(filepath: str) -> list[Document]:
    df = pd.read_excel(filepath)

    def row_to_text(row):
        return " | ".join(str(value) for value in row if pd.notna(value))

    texts = df.apply(row_to_text, axis=1).tolist()
    return [Document(page_content=text) for text in texts]


def load_json_as_documents(filepath: str) -> list[Document]:
    """Load and convert JSON content into LangChain Document objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    docs = []
    for item in data:
        text = json.dumps(item, indent=2)
        docs.append(Document(page_content=text))
    return docs


def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_and_save_vectorstore(docs: list[Document], embedding_model, path: str) -> None:
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    vectorstore.save_local(path)


def load_vectorstore(path: str, embedding_model) -> FAISS:
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


def search_vectorstore(vectorstore: FAISS, query: str, k=3) -> None:
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"Match {i+1}:\n{doc.page_content}\n")


def run_rag_query(vectorstore: FAISS, query: str) -> None:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    response = qa.invoke({"query": query})
    print("LLM Response:\n", response["result"])


def main():
    # --- Configuration ---
    json_path = "./data/enterprise-attack.json"
    faiss_path = "faiss_json_index"
    user_query = "What data is available about phishing attacks?"
    rag_query = "Summarize the information related to phishing in this dataset"

    # --- Workflow ---
    print("Loading and preparing documents...")
    documents = load_json_as_documents(json_path)
    docs = split_documents(documents)

    if not os.path.exists(faiss_path):
        print("Generating embeddings and saving vectorstore...")
        embedding_model = OpenAIEmbeddings()
        create_and_save_vectorstore(docs, embedding_model, faiss_path)
    else:
        print("Vectorstore already exists. Skipping creation.")

    print("Loading vectorstore and querying...")
    embedding_model = OpenAIEmbeddings()
    vectorstore = load_vectorstore(faiss_path, embedding_model)
    search_vectorstore(vectorstore, user_query)

    print("Running LLM-based RAG query...")
    run_rag_query(vectorstore, rag_query)


if __name__ == "__main__":
    main()
