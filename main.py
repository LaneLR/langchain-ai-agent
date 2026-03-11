from typing import List
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama

from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(description="List of sources used by the agent to answer the query")

### Creating custom tool for tavily search

# from tavily import TavilyClient

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the internet for the given query and returns the results.
#     Args: 
#         query: The search query.
#     Returns: 
#         The search result.
#     """
#     # print(f"Searching for query")
#     print(f"Results for: {query}")
#     return tavily.search(query=query)


    
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project="add_project_id", 
    location="us-central1",
    temperature=0,
    )

tools = [TavilySearch()]
# llm_with_structure = llm.with_structured_output(AgentResponse)
agent = create_react_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for a junior software engineer using React in Dallas, TX on LinkedIn and on Indeed posted within the last 3 days")})
    print(result)
    
if __name__ == "__main__":
    main()