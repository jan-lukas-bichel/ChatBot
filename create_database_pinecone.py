from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from os.path import join, dirname
import os

DATA_PATH = "data"

def main():
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_pinecone(chunks)

def load_documents():
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(DATA_PATH, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def save_to_pinecone(chunks: list[Document]):

    vectorstore_from_docs = PineconeVectorStore.from_documents(
        chunks,
        index_name=os.getenv('PINECONE_INDEX_NAME'),
        embedding=OpenAIEmbeddings()
    )

    print(f"Saved {len(chunks)} chunks to pinecone.")


if __name__ == "__main__":
    main()