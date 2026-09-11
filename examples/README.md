# A real scoring example

`position_alignment_hotpotqa.json` contains 100 previously measured Qwen3-8B outputs
from the position-alignment baseline on the frozen HotpotQA inputs.

```bash
kva score examples/position_alignment_hotpotqa.json --track re --subset hotpotqa
```

The record keeps its original precision. Its F1 is recomputed from predictions and
gold answers. This historical run is not the same-backend reference run used by every
later comparison; a nonzero paired difference from packaged references is possible.

The file predates `kva.result.v1`. It lacks complete cost telemetry and a method card,
so `kva validate` should reject it as a new submission. No synthetic costs were added.
New runs created with `kva run` use the current schema.
