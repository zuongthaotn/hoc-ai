import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from utils import get_embedding, get_vector_store
from agents.openai_agent import get_response


question = "How many orders are delivered?"
# Prepare the DB.
embedding = get_embedding()
db = get_vector_store(
    embedding=embedding,
    collection_name="example_collection",
    persist_directory="./chromaDB/shop01"
)

# Search the DB.
results = db.similarity_search_with_relevance_scores(question, k=999)
if len(results) == 0:
    print('Unable to find matching results.')
    exit(0)
context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

response = get_response(question, db, context)
print(response)