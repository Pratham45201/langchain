from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda
import pyperclip

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

""" Chaining translators """

# 1. Translation runnable, which will be passed to translation prompt template
prepare_for_translation = RunnableLambda(
    lambda output: {"output": output, "language": "french"}
)

# 2. Translation prompt template
translation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an efficient translator of english to {language}"),
        ("human", "Translate this {output}"),
    ]
)

# Chain
chain = (
    prompt_template
    | llm
    | StrOutputParser()
    | prepare_for_translation
    | translation_prompt
    | llm
    | StrOutputParser()
)

# Invoke chain
response = chain.invoke({"topic": "cat", "joke_count": 2})
pyperclip.copy(response)
print(response)
