"""
At high level chains are basically piped runnables which are executed sequentially
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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


# Create individual runnables
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create chain from runnables
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "cat", "joke_count": 2})
print(response)
