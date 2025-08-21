from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

# model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

# Base prompt
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic"),
        ("human", "Provide a brief summary of the move {movie_name}"),
    ]
)


# Plot analysis step
def create_plot_chain(plot):
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            (
                "human",
                "Analyze the plot: {plot}. What are its strength and weaknesses?",
            ),
        ]
    )


plot_branch_chain = create_plot_chain | llm | StrOutputParser()


# Character analysis step
def create_character_chain(characters):
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            (
                "human",
                "Analyze the characters: {characters}. What are its strength and weaknesses?",
            ),
        ]
    )


characters_branch_chain = create_character_chain | llm | StrOutputParser()


# Combine analysis into final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return (
        f"Plot analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"
    )


# Chain
chain = (
    summary_template
    | llm
    | StrOutputParser()
    | RunnableParallel(
        plot=plot_branch_chain,
        character=characters_branch_chain,
    )
    | RunnableLambda(lambda x: combine_verdicts(x["plot"], x["character"]))
)

result = chain.invoke({"movie_name": "Inception"})
print(result)
