"""

Types of Chaining:

1. Extended or Sequential Chaining:
    - Chaining tasks one by one in a straight/sequential line

2. Parallel Chaining:
    - Lets you run tasks parallely or simultaneously without being dependent on each other

3. Conditional Chaining:
    - Let you run a particular branch of chain on a condition

"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

# prompts
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes."),
    ]
)


# Create the combined chain using LangChain Expression Language (LCEL)
# This is example of sequential chain
chain = prompt_template | llm

result = chain.invoke({"topic": "cat", "joke_count": 2})
print(result.content, "\n")


# Using StrOutputParser() to automatically extract results
chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"topic": "cat", "joke_count": 2})
print("\n", result, "\n")
