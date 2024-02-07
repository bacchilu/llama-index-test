import os
import logging
import random

import openai
from pinecone import Pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from dotenv import load_dotenv

from utils import print_qa


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")


logging.basicConfig(filename="./debug.log", level=logging.DEBUG)


openai.api_key = OPENAI_KEY


INDEX_NAME = "life365"
EMBEDDING_MODEL = "text-embedding-3-small"


def text_to_vector(text):
    embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    return embed_model.get_text_embedding(text)


if __name__ == "__main__":
    p_client = Pinecone(api_key=PINECONE_KEY)
    index = p_client.Index(INDEX_NAME)

    # query_vector = text_to_vector(
    #     "Cosa mi serve in un impianto fotovoltaico con batterie?"
    # )
    # db_res = index.query(
    #     vector=query_vector,
    #     top_k=5,
    #     namespace="products-descriptions",
    #     include_metadata=True,
    # )
    # print(db_res)

    vector_store = PineconeVectorStore(
        pinecone_index=index, namespace="products-descriptions"
    )
    loaded_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=ServiceContext.from_defaults(
            embed_model=OpenAIEmbedding(model=EMBEDDING_MODEL)
        ),
    )
    query_engine = loaded_index.as_query_engine()
    q = random.choice(
        (
            "Mi serve un buon toner per una stampante HP",
            "Puoi suggerirmi un buon cavo di alimentazione? Anzi, due!",
            "Mi consigli un buon monitor?",
        )
    )
    print_qa(query_engine.query, q)
