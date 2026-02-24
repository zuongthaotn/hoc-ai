import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from utils import get_embedding, get_vector_store
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp


def load_llm_cpu(model_file="models/llm/phi-2.Q4_K_S.gguf"):
    llm = LlamaCpp(
        model_path=model_file,
        n_ctx=2048,
        temperature=0.7,
        n_batch=64,
        verbose=False
    )
    return llm

question = "How many orders are delivered?"
# Prepare the DB.
embedding = get_embedding()
db = get_vector_store(
    embedding=embedding,
    collection_name="example_collection",
    persist_directory="./chromaDB/shop01"
)
llm = load_llm_cpu()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

response = qa_chain.invoke({"query": "How many orders are delivered?"})
print(response)