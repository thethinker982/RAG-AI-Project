import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vectordb import faiss_db
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq model
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

def retrieve_docs(query):
    # Retrieve most relevant chunks
    return faiss_db.similarity_search(query, k=4)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

custom_prompt_template = """
You are an AI legal assistant. Answer ONLY using the provided context (extracted from the Universal Declaration of Human Rights).

Guidelines:
- Always cite the Article number(s) if present in the context.
- Do not invent or add extra information.
- If the context does not contain the answer, reply: "The document does not provide this information."

Question: {question}
Context:
{context}

Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})

    # Handle response properly (string or object)
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "content" in response:
        return response["content"]
    else:
        return str(response)

# Example usage
if __name__ == "__main__":
    question = "If a government forbids the right to assemble peacefully which articles are violated and why?"
    retrieved_docs = retrieve_docs(question)
    print("AI Lawyer:", answer_query(documents=retrieved_docs, model=llm_model, query=question))
