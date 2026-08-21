import dataclasses as dc


@dc.dataclass(frozen=True)
class BatchMetadata:
    source_file: str
    batch_idx: int
    token_count: int
    processing_time_ms: float = dc.field(
        compare=False,
        repr=False,
    )
    is_truncated: bool = False
    tags: list[str] = dc.field(default_factory=list)


bmd1 = BatchMetadata(
    source_file="The 5 projects.txt",
    batch_idx=0,
    token_count=100,
    is_truncated=False,
    processing_time_ms=123.45,
    tags=["project", "example"],
)

print(bmd1)

bmd2 = dc.replace(bmd1, batch_idx=1)
print(bmd2)

print(bmd1 == bmd2)
print(bmd1.is_truncated == bmd2.is_truncated)

try:
    bmd1.batch_idx = 2
except dc.FrozenInstanceError as e:
    print(f"Mutation blocked: {e}")
