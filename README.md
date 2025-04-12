

## Description

This application creates an AI chatbot that allows users to upload PDF documents and interact with them using natural language queries. It utilizes LangChain components, Chroma for vector storage, Ollama for running a local LLM (like LLaMA), and Streamlit for the user interface.

## Main Functionalities

Upload PDF files and extract text

Split text into manageable chunks

Store document embeddings in a Chroma vector database

Use LangChain's RetrievalQA system to answer questions based on uploaded PDF content

Maintain chat history and conversational memory

Generate responses using a locally hosted Ollama model

## Key Libraries Used

langchain, langchain_community: Core LLM and vector store components

streamlit: UI framework for the chatbot

chromadb: Persistent vector storage for document embeddings

PyMuPDF: Efficient PDF loader

## Workflow

Initialize App State: Create directories and set up session state for templates, memory, prompts, vector store, and model.

File Upload: User uploads a PDF file, which is saved locally.

Text Extraction and Splitting: The PDF content is split into chunks using RecursiveCharacterTextSplitter.

Embedding and Storage: Each text chunk is embedded using OllamaEmbeddings and stored in a Chroma vector database.

Retriever and QA Chain: A retriever is created from the vector store, and a RetrievalQA chain is initialized with the retriever and LLM.

## Chat Interaction:

User inputs a question.

The system retrieves relevant document chunks.

The LLM generates a response based on retrieved chunks and previous conversation history.

The conversation is displayed in the chat window.

Usage Instructions

Run the Streamlit app.

Upload a PDF file.

Ask questions about the uploaded document.

View responses generated from the LLM using contextual document knowledge.

## Features

Persistent memory and context tracking

Document-specific QA using embeddings

Local LLM integration

Interactive and dynamic UI via Streamlit

