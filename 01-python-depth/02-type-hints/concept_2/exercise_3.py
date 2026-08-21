from dataclasses import dataclass, field

@dataclass
class EvalResult:
    model_name: str
    scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    average: float = 0.0
    _cache: dict = field(default_factory=dict, repr=False)

    def update(self, score: float) -> None:
        self.scores.append(score)
        self.average = sum(self.scores) / len(self.scores)

result1 = EvalResult(model_name="gpt")
result2 = EvalResult(model_name="gpt")
result1.update(0.9)

print(result1.average)
print(result1.scores is result2.scores)