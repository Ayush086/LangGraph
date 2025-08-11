import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 't-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# load conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Ask here")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        
    with st.chat_message('assistant'):
        # streaming
        assistant_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG, stream_mode='messages'
            ) if getattr(message_chunk, 'content', "") # to handle when response is not generated
        )
        
    st.session_state['message_history'].append({'role': 'assistant', 'content': assistant_message})


