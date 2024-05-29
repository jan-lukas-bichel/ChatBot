import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

vectorstore = PineconeVectorStore(index_name=os.getenv('PINECONE_INDEX_NAME'), embedding=OpenAIEmbeddings())
AUGMENTED_PROMPT = ""
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question about the applicant based on the above context: {question}

If the question is off topic an there are no matching results, just ask for another
question about thew applicant and tell how awesome he is. 
"""

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
if chat_input := st.chat_input("What is up?"):

    # Similarity search with chat input
    results = vectorstore.similarity_search_with_relevance_scores(chat_input, k=5)
    if len(results) == 0 or results[0][1] < 0.7:
        results = [(Document(page_content="There is no relevant information regarding the applicant in the database") , 1)]

    # Combine results with chat input
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    AUGMENTED_PROMPT = prompt_template.format(context=context_text, question=chat_input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": chat_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(chat_input)

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