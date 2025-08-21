"""

There are three 3 types of messages in langchain

1. SystemMessage: Defines the AI's role and the sets the context for the conversation
    - Example : "You are a marketing expert"

2. HumanMessage: Represents user input or question directted to the AI
    - Example : "What's a good marketing strategy?"

3. AIMessage: Contains the AI's responses based on previous messages
    - Example: "Focus on social media engagement"
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on Instagram"),

    # AIMessage(),      we can than create a chain of messages for the LLM like this
    # HumanMessage(),
    # AIMessage(),
    # ...
]

result = llm.invoke(messages)

print(result.content)