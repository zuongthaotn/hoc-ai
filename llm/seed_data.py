import json
from uuid import uuid4
import warnings
warnings.filterwarnings("ignore")
from utils import convert_data_to_documents, get_embedding, get_vector_store

# 1. Load sample orders
with open("data/orders.json") as f:
    orders = json.load(f)

# 2. Convert orders to Documents
docs = convert_data_to_documents(orders)

# 3. Embeddings + Chroma

embedding = get_embedding()
vector_store = get_vector_store(
    embedding=embedding,
    collection_name="example_collection",
    persist_directory="./chromaDB/shop01"
)
uuids = [str(uuid4()) for _ in range(len(docs))]

# 4. Add documents to the vector store
vector_store.add_documents(documents=docs, ids=uuids)