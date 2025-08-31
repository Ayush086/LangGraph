import streamlit as st
from backend import chatbot, retrieve_unique_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

#### =============== utility functions =============== ####

def generate_new_thread_id():
    id = uuid.uuid4()
    return id

def generate_thread_name(messages):
    if messages and len(messages) > 0:
        first_msg = messages[0].content[:25] + "..." if len(messages[0].content) > 25 else messages[0].content
        return first_msg
    return "New Chat"

def load_new_conversation():
    # generate new thread
    new_thread_id = generate_new_thread_id()
    st.session_state['thread_id'] = new_thread_id
    # add new thread id to session state
    add_chat_thread(new_thread_id)
    # reset chat history
    st.session_state['message_history'] = []

def add_chat_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_previous_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if state.values and 'messages' in state.values:
            return state.values['messages']
        else:
            return []  # Return empty list if no messages found
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []

    

#### ========== Session Setup ========== ####
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_new_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_unique_threads()

add_chat_thread(st.session_state['thread_id'])

### ============== Sidebar =============== ###
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    load_new_conversation()

st.sidebar.header("Conversations")
# display thread id in sidebar
for i, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    messages = load_previous_conversation(thread_id=thread_id)
    thread_name = generate_thread_name(messages) if messages else f"chat {i+1}"
    
    if st.sidebar.button(thread_name, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        
        # refactoring the fetched response
        temp_msgs = []
        if messages: 
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else: role = 'assistant'
                temp_msgs.append({'role': role, 'content': msg.content})
        
        # resetting the message history content
        st.session_state['message_history'] = temp_msgs



# load conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Ask here")

if user_input:
    # add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        
    # enabling to see chats thread wise for each query langsmith won't create new trace, new trace will only created when new thread is formed
    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']}, # it's for backend to retrieve previous chat history based on thread id (if exists)
        'metadata': {
            'thread_id': st.session_state['thread_id'] # it's for langsmith to store thread related trace under thread's tree
        },
        'run_name': 'chat_turn',
        }
    with st.chat_message('assistant'):
        status_holder = {'box': None}
        # streaming
        def stream_generator():
            for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG, stream_mode='messages'
            ):
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, 'name', 'tool')
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(f"Using {tool_name}-", expanded=True)
                    else:
                        status_holder['box'].update(
                            label=f"Using {tool_name} -", state='running', expanded=True
                        )
                if (isinstance(chunk, AIMessage) and hasattr(chunk, 'content') and chunk.content):
                    yield chunk.content
        
        assistant_message = st.write_stream(stream_generator())
        
    # store llm resposne to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': assistant_message})
