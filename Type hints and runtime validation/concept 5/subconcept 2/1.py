from typing import Literal
from pydantic import BaseModel, ValidationError

class DatasetConfig(BaseModel):
    name: str
    split: Literal["train", "val", "test"]
    max_samples: int
    shuffle: bool = False
    
obj1 = DatasetConfig(name="my_dataset", split="train", max_samples=1000, shuffle=True)
obj2 = DatasetConfig(name="my_dataset", split="train", max_samples="1000", shuffle=True)
obj3 = DatasetConfig(name="my_dataset", split="train", max_samples=1000, shuffle=1)

print(*list(map(lambda obj: '; '.join([f"{item} : {type(item[1])}" for item in obj]), [obj1, obj2, obj3])))

try:
    obj4 = DatasetConfig(name="my_dataset", split="validation", max_samples=1000, shuffle=True)
except ValidationError as e:
    print(f"obj4 is not created due to {e}.")