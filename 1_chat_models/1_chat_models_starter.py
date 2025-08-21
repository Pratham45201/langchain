from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

result = llm.invoke("What is the square root of 49")

print(result.content)
