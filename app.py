import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() # Load environment variables

# Get the API token from environment variable
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the repository ID and task
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"

# App config
st.set_page_config(page_title="IT Architect",page_icon= "🌍")
st.title("Sambit.AI ✈️")

# Define the template outside the function
template = """
You are a  mentor chatbot your name is Sambit.AI designed to help users with any technology and implementation information. Here are some scenarios you should be able to handle:

1. Solution Architecture: Generate Solution Architecture based on reference architectures. Ask for deployment cloud, mandatory systems and any specific solution related question. Check for similar reference solutions online and propose a solution accordingly.

2. Prompt Engineering: Help users with generating prompts for getting desired generative text as output. Inquire about objective, sample input and output and number of words/token expected in output. 

3. Solution Evaluation: Help users evaluate from a set of solution options. Inquire about key features expected and then generate a pugh matrix assigning probable weightage to each feature.

Please ensure responses are informative, accurate, and provide references to the sources for the answers.

Chat history:
{chat_history}

User question:
{user_question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Function to get a response from the model
def get_response(user_query, chat_history):
    # Initialize the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=api_token,
        repo_id=repo_id,
        task=task
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })
    return response

# Initialize session state.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Sambit.AI. Let's discuss technology?"),
    ]

# Display chat history.
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.chat_history)

    # Remove any unwanted prefixes from the response u should use these function but 
#before using it I requestto[replace("bot response:", "").strip()] combine 1&2 to run without error.

    #1.response = response.replace("AI response:", "").replace("chat response:", "").
    #2.replace("bot response:", "").strip()

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response)) 

