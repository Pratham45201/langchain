'''
Classify the reivew into positive, negative, neutral or escalated.
Generate according notes for the user.
'''

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda, RunnableBranch

load_dotenv()

# model
model = ChatGroq(
    model="llama-3.1-8b-instant",
)

# Sentiment classification
sentiment_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an accurate sentiment analyzer"),
        (
            "human",
            "Classify the sentiment of this {review} into one of postive, negative or neutral",
        ),
    ]
)

sentiment_chain = sentiment_template | model | StrOutputParser()

# Positive response
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer support manager"),
        ("human", "Generate a thank you note for this positive review: {review}"),
    ]
)

# Negative response
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer support manager"),
        ("human", "Generate a apologetic note for this negative review: {review}"),
    ]
)

# Neutral response
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer support manager"),
        ("human", "Generate a according note for this neutral review: {review}"),
    ]
)

# Escalate feedback
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer support manager"),
        (
            "human",
            "Generate a message to escalate this review to human agent: {review}",
        ),
    ]
)


# Branches
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "positive" in x,
        neutral_feedback_template | model | StrOutputParser(),
    ),
    escalate_feedback_template | model | StrOutputParser(),
)

# Chain
chain = sentiment_chain | branches
response = chain.invoke({"review": "neutral product"})

print(response)
