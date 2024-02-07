import logging
import os
import requests
import itertools
import random

import openai
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
)
from dotenv import load_dotenv

from utils import print_qa


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")


openai.api_key = OPENAI_KEY


logging.basicConfig(filename="./debug.log", level=logging.DEBUG)


PERSIST_DIR = "./storage"


class DirectoryLoader:
    def __init__(self, folder: str):
        self.folder = folder

    def __call__(self):
        return SimpleDirectoryReader(self.folder).load_data()


class ApiLoader:
    def __init__(self, url: str):
        self.url = url

    def __call__(self):
        data = requests.get(self.url).json()
        documents = download_loader("JsonDataReader")().load_data(data)
        for d in documents:
            d.metadata = {
                "url": self.url,
                "id": data["id"],
                "barcode": data["barcode"],
                "code_simple": data["code_simple"],
                "brand": data["brand"],
            }
        return documents


def do_indexing(*loaders):
    """Load data and build an index"""
    if not os.path.exists(PERSIST_DIR):
        documents = list(itertools.chain(*[loader() for loader in loaders]))
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index


if __name__ == "__main__":
    index = do_indexing(
        DirectoryLoader("data"), ApiLoader("https://www.life365.eu/api/products/20406")
    )
    query_engine = index.as_query_engine()
    q = random.choice(
        (
            "Mi consigli un buon monitor?",
            "Chi consiglia di mettere il sale nel tè?",
            "What did the author do growing up?",
            "Cos'è successo Mercoledì in Sicilia?",
        )
    )
    print_qa(query_engine.query, q)
