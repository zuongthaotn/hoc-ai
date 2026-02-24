import json
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


def load_local_json_file(file="data/items.json"):
    """
    Load a local JSON file containing items.
    Args:
        file (str): Path to the JSON file. Defaults to "data/items.json".
    Returns:
        list: A list of items loaded from the JSON file.
    """
    items = []
    with open(file) as f:
        items = json.load(f)
    return items


def convert_data_to_documents(items):
    docs = []
    for item in items:
        content = f"item {item['id']} by {item['customer']}, product: {item['item']}, quantity: {item['total']}, status: {item['status']}"
        docs.append(Document(page_content=content))
    return docs

def get_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )

def get_vector_store(embedding, collection_name, persist_directory):
    """
    Create a Chroma vector store.
    Args:
        embedding: Embedding function to use.
        collection_name (str): Name of the collection. Defaults to "example_collection".
        persist_directory (str): Directory to persist the vector store. Defaults to "./chromaDB/shop01".
    Returns:
        Chroma: A Chroma vector store instance.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory
    )

# Load LLM

def load_llm(model_file):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_file,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
    )
    return llm
