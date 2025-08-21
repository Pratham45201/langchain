from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

chat_history = []

system_message = SystemMessage("You are a helpful AI assistant")
chat_history.append(system_message)

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(query))

    result = llm.invoke(chat_history)
    response = result.content

    chat_history.append(AIMessage(response))

    print("AI: ", response)
