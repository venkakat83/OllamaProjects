from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import streamlit as st

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

template = """Question : {question}

Answer : Lets think step by step.""" 

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model = "gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

input_text = st.text_input("Search the topic you want")

if input_text:
    st.write(chain.invoke({"question": input_text}))

