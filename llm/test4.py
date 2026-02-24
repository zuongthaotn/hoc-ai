import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from utils import get_embedding, get_vector_store
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain



prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question:

{context}

Question: {question}
Answer:
"""
)

prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the following context to answer the question:"),
        ("system", "\nContext: \n {context}"),
        ("user", "\n Question:\n {question}")
    ]
)


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
results = db.similarity_search_with_relevance_scores(question, k=999)
context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
llm = load_llm_cpu()


llm_chain = LLMChain(llm=llm, prompt=prompt2)
# output_parser = StrOutputParser()
# llm_chain = llm | prompt2 | output_parser
response = llm_chain.invoke({"context": context, "question": question})
print(response)