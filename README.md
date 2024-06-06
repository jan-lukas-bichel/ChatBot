# Job Application Chatbot

This is a very simple chatbot powered by retrieval augmented generation/ RAG with the purpose of of representing a job applicant.

## Retrieval Augmented Generation
The core part of the RAG mechanism is mostly based on this [excellent tutorial](https://youtu.be/tcqEUSNCn8I?si=La6D7a_pr6SrrUcN) and the [accompanying code]([https://youtu.be/tcqEUSNCn8I?si=La6D7a_pr6SrrUcN](https://github.com/pixegami/langchain-rag-tutorial).

The vector store is created in [this script](vector_store/create_database_pinecone.py). The library langchain was levered to split a small text document with information about the applicant into chunks. It was also used to access the OpenAI API to vectorize/ embed the chunks via the LLM GPT-3.5 Turbo. Finally langchain was used to make a call to the Pinecone API, to save the embedded  information to a vector store in the cloud.

In [this script](vector_store/query_data_pinecone.py) the database can then be just queried via the command line. The user inputs a prompt, which is also vectorized via ChatGPT. Based on this vectorized prompt a given amount of relevant information chunks will be retrieved from the database, i.e. via euclidian distance. So basically the nearest vectors to the embedded prompt stored in the database will be retrieved. The original prompt is then merged with the retrieved infomration chunks through the means of a simple template. There may or may not have been instructions for the AI to show the applicant in a flattering light.
```
Answer the question based only on the following context:
{retrieved chunks will be inserted here}

---

Answer the question about the applicant based on the above context: {orignal prompt will be inserted here}

If the question is off topic an there are no matching results, just ask for another
question about the applicant and tell how awesome he is. 
```

## Front-End
The front-end is based [on another tutorial](https://youtu.be/Z41pEtTAgfs?si=Qy0HjIGgzvI2fVcT) and the [associated repository](https://github.com/dataprofessor/openai-chatbot). It's a very simple set-up for a streamlit-app (framework for full-stack web-apps, fully written in python) that integrates the ChatGPT API. The original app didn't include any RAG, so some modifications had to be made. The only part of the conversation that is now kept in memory for the AI are the original prompts, not the modified prompts with the RAG context. The RAG context is only provided for the newest prompt. Also the code was modified, to only show the original prompt in the UI, not the modified one.

## Deployment


## Prerequisite libraries

```
streamlit
openai
```

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Further Reading

- 🛠️ [Streamlit Documentation Tutorial on _**Build conversational apps**_](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
- 📖 [Streamlit Documentation on _**Chat elements**_](https://docs.streamlit.io/library/api-reference/chat)
