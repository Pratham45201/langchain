"""

AI Agent terminologies

1. Agent: AI decision-makers that can pick the right tool for the job without being told what to use

2. Tools: Tools are specific functions that Agents can use to complete tasks

3. reACT: One of the patterns in AI to build agents. It stands for Reasoning + Acting.

- Example

When asked "What is the temperature in paris + 5 ? "

The agent first thinks that 'I need to get temperature in paris' and for that it uses the `tool` which is weather API. After getting the temperature it thinks 'Now I can add 5 to this' and for that it uses the `tool` calculator. Finally it returns the answer which is 25.

"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import tool, create_react_agent, AgentExecutor
import datetime

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
)

query = "What is the current time in america? (You are in india)."

# reACT prompt template from langchain-hub
prompt_template = hub.pull("hwchase17/react")


# List of tools which can be used by the agent
# For this purpose we provide get current time function
# We also provide a context of the tool so that llm can decide when to use it
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [get_system_time]

# Agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Run agent
result = agent_executor.invoke({"input": query})
