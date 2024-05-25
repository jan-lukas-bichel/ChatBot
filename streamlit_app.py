import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.retrievers import SimpleRetriever

st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets
client = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7, openai_api_key=st.secrets["OPENAI_API_KEY"], streaming=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
if st.session_state.messages != []:
    with st.chat_message("assistant"):
        stream = client.stream(
            input=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})