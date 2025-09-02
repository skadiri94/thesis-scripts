import os
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


def load_excel_as_documents(filepath: str) -> list[Document]:
    """Load and convert Excel rows into LangChain Document objects."""
    df = pd.read_excel(filepath)

    def row_to_text(row):
        return " | ".join(str(value) for value in row if pd.notna(value))

    texts = df.apply(row_to_text, axis=1).tolist()
    return [Document(page_content=text) for text in texts]


def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    """Split documents into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_and_save_vectorstore(docs: list[Document], embedding_model, path: str) -> None:
    """Embed and store documents in a FAISS vector store."""
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    vectorstore.save_local(path)


def load_vectorstore(path: str, embedding_model) -> FAISS:
    """Load an existing FAISS vector store."""
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


def search_vectorstore(vectorstore: FAISS, query: str, k=3) -> None:
    """Search vector store and print results."""
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"Match {i+1}:\n{doc.page_content}\n")


def run_rag_query(vectorstore: FAISS, query: str) -> None:
    """Run a Retrieval-Augmented Generation (RAG) query using AI."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    # Use invoke instead of run
    response = qa.invoke({"query": query})
    print("LLM Response:\n", response["result"])


def main():
    # --- Configuration ---
    excel_path = "./data/enterprise-attack-datasources.xlsx"
    faiss_path = "./faiss_excel_index"
    user_query = "What does the data say about Certificate: Certificate Registration?"
    rag_query = "give description of the data source: Certificate: Certificate Registration"

    # --- Workflow ---
    print("Loading and preparing documents...")
    documents = load_excel_as_documents(excel_path)
    docs = split_documents(documents)

    # Check if the vector store already exists
    if not os.path.exists(faiss_path):
        print("Generating embeddings and saving vectorstore...")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")
        create_and_save_vectorstore(docs, embedding_model, faiss_path)
    else:
        print("Vectorstore already exists. Skipping creation.")

    print("Loading vectorstore and querying...")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vectorstore = load_vectorstore(faiss_path, embedding_model)
    search_vectorstore(vectorstore, user_query)

    print("Running LLM-based RAG query...")
    # run_rag_query(vectorstore, rag_query)


if __name__ == "__main__":
    main()
