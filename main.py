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


    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for a junior software engineer using React in Dallas, TX on LinkedIn and posted within the last 30 days")})
    print(result)
    
if __name__ == "__main__":
    main()