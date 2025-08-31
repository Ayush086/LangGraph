from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_cohere import ChatCohere
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# tools integratin
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()

llm = ChatCohere(model='command-r',
                temperature=0.5,
                k=0, streaming=True)

# tools definition
search_tool = DuckDuckGoSearchRun(region='us-en')

#custom tools
@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """perform arithmetic operation on two numbers.
    valid operations: add, sub, mul, div"""
    try: 
        if operation == 'add':
            result = first_number + second_number
        elif operation == 'sub':
            result = first_number - second_number
        elif operation == 'mul':
            result = first_number * second_number
        elif operation == 'div':
            if second_number == 0:
                return {'error': 'division by zero can not be done'}
            result = first_number / second_number
        else:
            return {'error': f"Unsupported Operation {operation}"}
        
        return {'first_num': first_number, 'second_num': second_number, 'operation': operation, 'result': result}
    except Exception as e:
        return {'error': f'calculation failed. {str(e)}'}
            
@tool
def stock_price_retriever_tool(symbol: str) -> dict:
    """Fetch latest stock price of given symbol (eg. 'TSLA') using alpha vantage with API key in URL"""
    
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9BO8HP247123YIA'
    r = requests.get(url)
    
    return r.json()

tools = [search_tool, stock_price_retriever_tool, calculator]
llm_with_tools = llm.bind_tools(tools)

# graph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# nodes
def chat_node(state: ChatState) -> ChatState:
    """ LLM node may answer or request a tool call"""
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools
                     )
# db connection
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False) # if set True - then it'll work for single thread only
# checkpointer
checkpointer = SqliteSaver(conn=connection)

# graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)



## ---- utility functions ---- ##
def retrieve_unique_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list(all_threads)