from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from langchain import HuggingFaceHub
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from config import HUGGINGFACEHUB_API_TOKEN
import re

app = FastAPI()

def chatbot(file_path: str, question: str, k: int):
    loader = Docx2txtLoader(file_path=file_path)
    documentos = loader.load()
     # Se dividen los documentos en chunks
     # Se escogen un tamaño de 1000 para el "chunk_size" ya que 
     # entre más grande sea "chunk_size" va a preservar más contexto.
     # Lo que implica que usará más información para realizar la predicción
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Número de caracteres para cada chunk 
        chunk_overlap=150  # Número de solapamientos entre chunks
    )
    
    docs = text_splitter.split_documents(documentos)
    # Define embedding
    nombre_modelo = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_hf = HuggingFaceEmbeddings(
        model_name=nombre_modelo,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embedding_hf)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity", 
                                search_kwargs={"k": k})
    token_hf = HUGGINGFACEHUB_API_TOKEN
    repo_id = "google/gemma-7b"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=token_hf,
        model_kwargs={"temperature": 0.01},
    )

    # Create PromptTemplate
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an 
    answer. Use three sentences maximum. Keep the answer as consice as possible. 
    Respond in the language of the question. At the beggining of the 
    answer put the user_name in next way: "user_name, ".
    Always say "gracias por preguntar!" at the end of the answer.
    
    {context}
    Pregunta: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", 
                                                      "question"], 
                                                      template=template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain({"query": question})
    result = result['result']
    # Use a regular expression to extract all text between "Answer" and the next occurrence
    matches = re.findall(r"Answer:\n(.*?)(?=Helpful Answer:|$)", 
                         result, re.DOTALL)

    # Join the results if there are multiple matches or return as a single string
    result = " ".join(match.strip() for match in matches)
    return result

@app.post("/chatbot/")
async def chatbot_endpoint(file: UploadFile = File(...), 
                           question: str = Form(...), 
                           k: int = Form(5)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Call the chatbot function
        response = chatbot(temp_file_path, question, k)

        # Clean up temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
