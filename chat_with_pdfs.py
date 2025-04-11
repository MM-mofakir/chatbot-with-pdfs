

from langchain_community.embeddings import SentenceTransformerEmbeddings
import sqlite3
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit  as st
import os
import time
import chromadb

# Create directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('db'):
    os.mkdir('db')


# Initialize template as a session state 
if 'template' not in st.session_state:

    # Set value of template key
    st.session_state.template = """
    
    You are a knowledgeable chatbot, here to help with questions of the user. 
    Your tone should be polite, professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:

    """

# Initialize prompt as a session state
if 'prompt' not in st.session_state:
    
    # Set value of prompt key to PromptTemplate from langchain.prompts 
    st.session_state.prompt = PromptTemplate(
        
        # Set input variables 
        input_variables=["history", "context", "question"],
        
        # Set template to the session state, template 
        template=st.session_state.template,
    )

# Initialize memory as a session state
if 'memory' not in st.session_state:
   
    # Set value of memory key to ConversationBufferMemory from langchain.memory
    st.session_state.memory = ConversationBufferMemory(

        # Set params from input variables list
        memory_key="history",
        return_messages=True,
        input_key="question")
    

# Initialize vectorstore
if 'vectorstore' not in st.session_state:
   
    # Set value of vectorstore key to Chroma 
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore =   Chroma(client=chroma_client, collection_name="chatbot_knowledge", embedding_function=embedding_function)
    

if 'llm' not in st.session_state:

    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize session state
if 'chat_history' not in st.session_state:
    
    st.session_state.chat_history = []

st.title("Chat with your PDFs")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

for message in st.session_state.chat_history:
    
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()
             # loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")  
            loader = PyMuPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
                )

            all_splits = text_splitter.split_documents(data)   
            # all_splits = text_splitter.split_text(data) 
            # Create and persist the vector store
           
            try:
                st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3"),persist_directory="db")
            except Exception as e: print('error can not load chroma7   ',e) 

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
    if 'qa_chain' not in st.session_state:

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)

            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)


else:
    st.write("Please upload a PDF file.")

