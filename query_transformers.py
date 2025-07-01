"""
rag system
"""

# import typing
# import faiss
# import numpy as np
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Cargar variables de entorno
load_dotenv()

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Sistema RAG - Realizar consultas sobre documentos')
parser.add_argument('question', help='Pregunta a realizar al sistema RAG')
args = parser.parse_args()

# Configurar logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Cargar configuración desde variables de entorno
faiss_db_path = os.getenv('FAISS_DB_PATH', '/home/alejandrocr/llm/faiss_db')
embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME', 'paraphrase-MiniLM-L6-v2')
llm_model_name = os.getenv('LLM_MODEL_NAME', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
retrieval_k = int(os.getenv('RETRIEVAL_K', '4'))
max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', '150'))
temperature = float(os.getenv('TEMPERATURE', '0.7'))
do_sample = os.getenv('DO_SAMPLE', 'true').lower() == 'true'

# Cargar el índice FAISS desde disco si existe
if not os.path.exists(faiss_db_path):
    logger.error('No se encontró el directorio de índice FAISS en: %s', faiss_db_path)
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

faiss_db = FAISS.load_local(
    faiss_db_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# Crear el retriever a partir del VectorStore
retriever = faiss_db.as_retriever(search_kwargs={'k': retrieval_k})

# Cargar el modelo de lenguaje usando transformers
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Crear un pipeline de HuggingFace
transformers_pipeline = pipeline(
    'text-generation',
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    pad_token_id=tokenizer.eos_token_id
)

# Envolver el pipeline para usarlo con LangChain
huggingface_pipeline = HuggingFacePipeline(pipeline=transformers_pipeline)

# Crear la cadena RAG con RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=huggingface_pipeline,
    retriever=retriever,
    chain_type='stuff',
    return_source_documents=True
)

# Ejecutar una consulta
question = args.question
logger.info('Pregunta: "%s"', question)

answer = qa.invoke({'query': question})
result = answer['result']
print(result)
