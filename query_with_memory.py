import sys
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer

MEMORY_FILE = 'chat_memory.json'

def load_memory():
    if Path(MEMORY_FILE).exists():
        return ChatMemoryBuffer.from_string(Path(MEMORY_FILE).read_text(encoding='utf-8'))
    return ChatMemoryBuffer.from_defaults(token_limit=4096)

def save_memory(memory):
    Path(MEMORY_FILE).write_text(memory.to_string(), encoding='utf-8')

def build_chat_engine(folder, memory):
    docs = SimpleDirectoryReader(folder).load_data()
    Settings.embed_model = OllamaEmbedding(
        model_name='mxbai-embed-large:latest'
    )
    Settings.llm = Ollama(
        model='deepseek-r1:1.5b',                # 1 .5 B LLM
        request_timeout=120.0,
    )
    index = VectorStoreIndex.from_documents(docs)
    return index.as_chat_engine(
        chat_mode='context',
        memory=memory,
        similarity_top_k=4,
    )

def main():
    if len(sys.argv) != 2:
        print('Uso: python query_with_memory.py <carpeta_documentos>')
        sys.exit(1)

    folder = sys.argv[1]
    if not Path(folder).exists():
        print(f'Error: El directorio "{folder}" no existe.')
        print('Ejemplo: python query_with_memory.py docs/')
        sys.exit(1)

    memory = load_memory()
    chat_engine = build_chat_engine(folder, memory)

    try:
        while True:
            q = input('\nPregunta ("exit" para salir): ')
            if q.strip().lower() in {'exit', 'quit'}:
                break
            r = chat_engine.chat(q)
            print('\nRespuesta:\n', r)
    finally:
        save_memory(memory)

if __name__ == '__main__':
    main()
