import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import nltk
nltk.download('punkt')

# Title for the Streamlit application
st.title("RAG System for Q&A Chat Bot")
st.header("Ask any question related to the 'Leave No Context Behind' Paper")

# Read API key from a file
with open("gemini_key_text.txt", "r") as file:
    API_KEY = file.read().strip()

# Initialize Google Generative AI model
google_gai = ChatGoogleGenerativeAI(google_api_key=API_KEY, model="gemini-1.5-pro-latest")

# Load the PDF document
pdf_loader = PDFMinerLoader(r"C:\Users\Admin/LLM_Document.pdf")
pdf_content = pdf_loader.load()

# Initialize ChromaDB connection
chroma_db = Chroma(persist_directory=r"C:\Users\Admin/chroma_db")

def retrieve_context(question):
    try:
        relevant_sections = chroma_db.search(question)
        context = ""
        for section in relevant_sections:
            context += section.page_content + "\n"
        return context
    except Exception as e:
        st.error(f"An error occurred during context retrieval: {e}")
        return ""

# Define chatbot prompt templates
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from the user. Your answer should be based on the specific context."),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext: {context}\nQuestion: {question}\nAnswer:")
])

# Define output parser for chatbot response
output_parser = StrOutputParser()

# Define RAG chain for chatbot interaction
rag_chain = (
    {"context": retrieve_context, "question": RunnablePassthrough()}
    | chat_template
    | google_gai
    | output_parser
)

# Input field for user's question
user_input = st.text_area("Write your question here")

# Button to trigger the response
if st.button("Submit"):
    st.header("Your Question:")
    st.write(user_input)
    
    # Generate AI response based on the user's question and context
    ai_response = rag_chain.invoke({"context": pdf_content, "question": user_input})
    
    st.header("Generated Answer:")
    st.write(ai_response)
