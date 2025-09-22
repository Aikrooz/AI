from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_ollama import OllamaLLM
import os
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"


prompts = ChatPromptTemplate.from_messages([
    ("system", "You are a help full life coach assistant"),
    ("user", "Question {question}")
])

llm=OllamaLLM(model="gemma:2b",base_url="http://localhost:11434")
output_parser=StrOutputParser()
chain=prompts | llm | output_parser
st.title("Life Coach Chatbot ")
text_input = st.text_input("Tell me your life problems:")
st.write("By Aaronjames")
if text_input:
    st.write(chain.invoke({"question":text_input}))