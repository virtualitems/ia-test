"""
rag system
"""
import logging
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

# Configurar logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Cargar configuración desde variables de entorno
pdf_file_path = os.getenv('PDF_FILE_PATH', '4x.pdf')
faiss_db_path = os.getenv('FAISS_DB_PATH', '/home/alejandrocr/llm/faiss_db')
embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME', 'paraphrase-MiniLM-L6-v2')
chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '100'))

# Cargar el documento PDF para obtener los textos
logger.info('Cargando el documento "%s"', pdf_file_path)

loader = PyPDFLoader(pdf_file_path)
raw_documents = loader.load()

logger.info('Número de páginas cargadas: %d', len(raw_documents))

# Dividir el texto en fragmentos para mejorar la recuperación
logger.info('Dividiendo el texto en fragmentos')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)

documents = text_splitter.split_documents(raw_documents)

logger.info('Número de fragmentos creados: %d', len(documents))

# Cargar el modelo de embeddings
logger.info('Cargando el modelo de embeddings')

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

logger.info('Modelo de embeddings cargado: %s', embeddings.model_name)

# Crear y poblar el índice FAISS
logger.info('Creando el índice FAISS')

faiss_db = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
    # index_kwargs={"metric": faiss.METRIC_L2}
)

faiss_db.save_local(faiss_db_path)

logger.info('Índice FAISS creado y guardado en "%s"', faiss_db_path)
