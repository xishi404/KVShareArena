# Add a method or report a result

Implement `BaseRepairMethod`, then register your class with `@register("my_method")`.

```python
from kvsharearena import AnswerResult, BaseRepairMethod, register

@register("my_method")
class MyMethod(BaseRepairMethod):
    track = {"re"}

    def setup(self, model_id, config):
        raise NotImplementedError("Load your receiver and method here")

    def build_cache(self, source_text):
        raise NotImplementedError("Encode one source without the question")

    def answer(self, question, caches, config):
        raise NotImplementedError("Return AnswerResult with text and measured costs")
```

See `kvsharearena/drivers/naive_pos.py` for a working implementation. Import a new
driver before calling the harness so its registration runs.

Use the frozen inputs without selecting samples after seeing results. Include a method
card with the model, backend, versions, training data, implementation source, and cost
measurement procedure. Never substitute zeros for unavailable cost measurements.

```bash
kva validate runs/my_method.json --data_dir data/queryset
kva score runs/my_method.json --data_dir data/queryset
kva submit runs/my_method.json --data_dir data/queryset --dry-run
```

The last command prepares files and PR instructions without publishing. Remove
`--dry-run` only when you intend to open a pull request and have authenticated the
GitHub CLI. No `/evaluate` bot or automatic leaderboard admission is enabled in this
release. Maintainers review submissions before updating the measured-data snapshot.

Review requires matching model and prompt settings, the frozen sample IDs and gold
answers, independently recomputed quality, and documented costs. Do not trust an
uploaded score file as evidence of verification. Never include access tokens, private
paths, cluster logs, personal data, or model weights in submissions.
