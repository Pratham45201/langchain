from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

# This could be a username or a unique ID for each user
SESSION_ID = "user_session_new"
PROJECT_ID = "langchain-7996b"
COLLECTION_NAME = "chat_history"

client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory(
    client=client, collection=COLLECTION_NAME, session_id=SESSION_ID
)

print("Chat history initialized")
print("Current chat history: ", chat_history.messages)
