from typing import Callable


def print_qa(query_fn: Callable[[str], str], prompt: str):
    print(f"=> {prompt}")
    print(f"<= {query_fn(prompt)}")
    print()
