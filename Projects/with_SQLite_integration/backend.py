from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_cohere import ChatCohere
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatCohere(model='command-r-plus-08-2024',
                max_tokens=300,
                temperature=0.4,
                k=0, streaming=True)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# nodes
def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# db connection
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False) # if set True - then it'll work for single thread only

# checkpointer
checkpointer = SqliteSaver(conn=connection)

# graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)



## ---- utility functions ---- ##
def retrieve_unique_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list(all_threads)
