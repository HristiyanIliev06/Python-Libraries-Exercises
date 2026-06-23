from typing import Literal

def select_device(preferred: str) -> Literal["cpu", "cuda", "mps"]:
    # if preferred in ['cpu', 'cuda', 'mps']:
    #     return preferred
    # else:
    #     return "cpu"  # Default to CPU if the preferred option is not valid
    
    # return preferred if preferred in ['cpu', 'cuda', 'mps'] else "cpu"
    
    # match preferred:
    #     case "cpu" | "cuda" | "mps":
    #         return preferred
    #     case _:
    #         return "cpu"  # Default to CPU if the preferred option is not valid
    
    if preferred == "cpu":
        return "cpu"
    elif preferred == "cuda":
        return "cuda"
    elif preferred == "mps":
        return "mps"
    return "cpu"
    
def run_on(device: Literal["cpu", "cuda", "mps"]) -> str:
    return f"Running on {device}"

print(run_on(select_device("cuda")))  # Valid input
print(run_on(select_device("tpu")))   # Invalid input, defaults to CPU
    
