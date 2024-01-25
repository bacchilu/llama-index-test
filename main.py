import logging
import os
import requests

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
)


logging.basicConfig(filename="./debug.log", level=logging.DEBUG)


PERSIST_DIR = "./storage"


def do_indexing():
    """Load data and build an index"""
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()

        data = requests.get("https://www.life365.eu/api/products/20406").json()
        documents = [*documents, *download_loader("JsonDataReader")().load_data(data)]

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index


def query(index: VectorStoreIndex, prompt: str):
    """Query your data"""
    query_engine = index.as_query_engine()
    return query_engine.query(prompt)


def print_qa(index: VectorStoreIndex, prompt: str):
    print(f"=> {prompt}")
    print(f"<= {query(index, prompt)}")
    print()


if __name__ == "__main__":
    index = do_indexing()
    print_qa(index, "Mi consigli un buon monitor?")
    print_qa(index, "Chi consiglia di mettere il sale nel tè?")
    print_qa(index, "What did the author do growing up?")
    print_qa(index, "Cos'è successo Mercoledì in Sicilia?")
    print_qa(index, "Where is Forlì?")
