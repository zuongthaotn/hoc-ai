import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from utils import get_embedding, get_vector_store
from agents.llama_agent import get_response
import time


def get_answer(question):
    query_text = question

    # Prepare the DB.
    embedding = get_embedding()
    step_1 = time.time()
    db = get_vector_store(
        embedding=embedding,
        collection_name="example_collection",
        persist_directory="./chromaDB/shop01"
    )
    step_2 = time.time()

    # # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=999)
    step_3 = time.time()
    if len(results) == 0:
        return {'content': 'Unable to find matching results.'}
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    step_4 = time.time()
    # context = """
    #     Order ORD001 by Alice, product: Macbook Pro, total: 2500 USD, status: Delivered.
    #     Order ORD002 by Bob, product: iPhone 14, total: 1200 USD, status: Pending.
    #     Order ORD003 by Alice, product: iPad, total: 800 USD, status: Delivered.
    #     Order ORD004 by Magnus, product: AirPods Pro, total: 250 USD, status: Cancelled.
    # """
    response = get_response(question, db, context)
    step_5 = time.time()
    print(f"Step1 took {step_1 - start:.2f} seconds")
    print(f"Step2 took {step_2 - step_1:.2f} seconds")
    print(f"Step3 took {step_3 - step_2:.2f} seconds")
    print(f"Step4 took {step_4 - step_3:.2f} seconds")
    print(f"Step5 took {step_5 - step_4:.2f} seconds")

    return {'content': response['result'] if 'result' in response else "I'm sorry, I don't have the answer"}

start = time.time()
get_answer("How many orders are cancelled?")