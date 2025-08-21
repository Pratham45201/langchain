"""

RAGs is a method where we combine LLMs with a retrieval system.

This retrieval system can search through vast sources of external information like documents,
databases, or knowledge bases whenever the LLM needs additional knowledge to give you a better answer.

The Retrieval System follows this steps:

1. First partitions the document into chunks
2. Creates embeddings of each chunk
3. Stores the embeddings in Vector DB

Now whenever the user asks a question the retrieval system first finds relevant chunks (using embeddings)
from the Vector DB for the given question then passes these chunks to the LLM along with the question and then the LLM returns
according answer.

"""

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))

# Document for RAG
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


"""
---------------------------- Vector db ----------------------------
"""

# Path for Vector DB
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the chroma vector store already exists, create otherwise
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the document for RAG exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path"
        )

    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    # Each chunk will have about 1000 characters with 50 characters overlap between chunks
    # Overlap helps in maintaining context between chunks, usually ranges from 20 - 100
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")


"""
---------------------------- Retriever ----------------------------
"""

# Load the existing or newly created vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# User question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
# Retrieve top k most relevant chunks
# Chunks should have at least 0.5 relevance
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)

# Get relevant chunks
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get("source", "Unknown")}\n")
