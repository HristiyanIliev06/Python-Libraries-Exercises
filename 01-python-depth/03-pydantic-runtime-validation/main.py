from typing import Iterable, Iterator, Literal, TypeVar
from pydantic import BaseModel, Field, ValidationError
T = TypeVar("T")

class RawRecord(BaseModel):
    text: str
    label: Literal["positive", "negative"]
    score: float = Field(ge = 0.0, le = 1.0)

def load_records(raw_dicts: list[dict],
errors: list[tuple[dict, ValidationError]])->Iterator[RawRecord]:
    for d in raw_dicts:
        try:
            yield RawRecord.model_validate(d)
        except ValidationError as e:
           errors.append((d, e))

def transform(records: Iterable[RawRecord])->Iterator[RawRecord]:
    for r in records:
        r.text = r.text.strip().lower()
        yield r

def filter_valid(records: Iterable[RawRecord],
filtered_errors: list[tuple[RawRecord, ValueError]])->Iterator[RawRecord]:
    for r in records:
        if len(r.text) > 0:
            yield r
        else:
            filtered_errors.append((r, ValueError("Text must contain at least one character")))

def batch(records: Iterable[T], size: int)->Iterator[list[T]]:
    buf: list[T] = []
    for r in records:
        buf.append(r)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
 
raw = [
    {"text": "  Hello World  ", "label": "positive", "score": 0.9},
    {"text": "     ", "label": "negative", "score": 0.1},
    {"id": 123},  # malformed — missing required fields
]

errors: list[tuple[dict, ValidationError]] = []
filtered_errors: list[tuple[RawRecord, ValueError]] = []
for b in batch(filter_valid(transform(load_records(raw, errors)), filtered_errors), size=2):
    print(b)

if errors:
    print(f"{len(errors)} record(s) rejected after validation:")
    for record, err in errors:
        print(record, "→", err)

if filtered_errors:
    print(f"{len(filtered_errors)} record(s) rejected after filtration:")
    for record, err in filtered_errors:
        print(record, "→", err)