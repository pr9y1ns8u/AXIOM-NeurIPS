# Experiment log

Append-only. One entry per run, written *before* the run starts. Never edit an
existing entry — if a run was wrong, add a new entry saying so. Failures
(OOMs, silent NaNs, budget-sum drift) get logged too.

Format:

```
## <date>-<letter>
Who:
What:
Model:
Git:
Hypothesis:
Result: [fill after]
```
