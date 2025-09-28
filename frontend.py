import streamlit as st
from rag_pipeline import answer_query,retrieve_docs,llm_model

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False)
user_query = st.text_area("Enter your query here", height=100, placeholder="Type your question...")

ask_question = st.button("Ask Question")
if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        #fixed_response = "This is a placeholder response to your query."
        st.chat_message("AI Lawyer: ").write(response)
    else:
        st.error("Kindly Upload PDF File")

