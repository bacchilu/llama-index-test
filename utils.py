from pprint import pprint
from typing import Callable


def print_qa(query_fn: Callable[[str], str], prompt: str):
    print(f"=> {prompt}")
    response = query_fn(prompt)
    print(f"<= {response}")
    print()
    print("METADATA:")
    pprint(response.metadata)
