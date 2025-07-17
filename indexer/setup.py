from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from portkey_ai import Portkey
from langchain_core.documents import Document
import chromadb
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from uuid import uuid4

llm = Portkey(virtual_key="openrouter-2cc325", api_key="68bnSBajtFs8UYdsPZa/rXloDR9V")

system_prompt = """
You are a helpful assistant with Expertise in Django Framewor that can help me understand the code in the file.
A file can have multiple classes, functions, variables, etc.
You need to understand the file and provide a concise description of the file.
Incase of file with multiple classes, functions, variables, etc. You need to provide the description of each of them.
Also note the import statements in the file.Also use that to understand which classes, functions, variables are used in the file.

"""


def get_file_content(file_path: str) -> str:
    """Read and return file content as string"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def get_file_description(content: str, file_path: str) -> str:
    """Generate description for file content using LLM"""
    prompt = f"""
    Please provide a concise description of the following code file.
    Focus on its main purpose and functionality.
    The file would be a python file with multiple classes, functions, variables, etc.
    They could also include imports , django models , classmethods. Also take into account the files it is importing to undersatand the worflow.
    
    Content:
    {content}...
    """
    response = llm.chat.completions.create(
        model="anthropic/claude-sonnet-4",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def index_repository(vector_store, repo_path: str):
    """Index repository files with their LLM-generated descriptions"""

    documents: List[Document] = []

    # Walk through repository
    for root, _, files in os.walk(repo_path):
        for file in files:
            # Skip common non-code files
            if file.startswith(".") or file in ["LICENSE", "README.md"]:
                continue

            file_path = os.path.join(root, file)

            # Read file content
            content = get_file_content(file_path)
            if not content:
                continue

            # Generate description using LLM
            description = get_file_description(content, file_path)

            # Create document with metadata
            doc = Document(
                page_content=description,
                metadata={
                    "source": file_path,
                    "file_name": file,
                },
                id=file_path,
            )
            documents.append(doc)

    # Create embeddings and store in Chroma

    # Convert Document objects to dictionary format for Chroma
    uids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uids)

    return vector_store


def main():
    vector_store = Chroma(
        collection_name="reminderapp_frontent",
        persist_directory="./chroma_langchain_db_frontend",  # Where to save data locally, remove if not necessary
    )  # path defaults to .chroma
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("reminderapp_frontend")
    vector_store_from_client = Chroma(client=persistent_client, collection_name="reminderapp_frontend")
    # we can do the creation and getting of a collection in one line
    # collection = client.get_or_create_collection(name="my_programming_collection")
    index_repository(
        vector_store_from_client,
        repo_path="/Users/mohitmathur/home/privateprojects/reminderapp/LetsDO/frontend/src/pages",
    )


if __name__ == "__main__":
    main()
