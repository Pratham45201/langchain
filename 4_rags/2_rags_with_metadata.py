import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path"
        )

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []

    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path=file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

# Load the vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the user's question
query = "What is Dracula feared off?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
relevant_docs = retriever.invoke(query)

# Prepare prompt
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents: \n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rought answer based on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("system", "{combined_input}"),
    ]
)

chain = prompt_template | model | StrOutputParser()
response = chain.invoke({"combined_input": combined_input})
print("\n--- Answer ---\n")
print(response)
