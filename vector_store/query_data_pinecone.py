import argparse
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from os.path import join, dirname
import os

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question about the applicant based on the above context: {question}

If the question is off topic an there are no matching results, just ask for another
question about thew applicant and tell how awesome he is. 
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load API keys from .env
    dotenv_path = join(dirname(__file__), '../ChatBot/.env')
    load_dotenv(dotenv_path)

    # Load pinecone db
    vectorstore = PineconeVectorStore(index_name=os.getenv('PINECONE_INDEX_NAME'), embedding=OpenAIEmbeddings())

    # Search the DB.
    results = vectorstore.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        results = [(Document(page_content="There is no relevant information regarding the applicant in the database") , 1)]

    print(results)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7, streaming=True)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
