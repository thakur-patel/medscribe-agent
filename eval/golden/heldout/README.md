# Held-out cases

This folder is the held-out test set. Rules:

- Held-out cases are written by people on the team, never by an agent or a model.
- They are never read by prompts, tools, few-shot examples or any development loop. Only the eval runner reads them, and only for a final score.
- Same JSONL format as `../dev/`: one case per line, `split: "heldout"`.
- Synthetic data only. Never use real patient data.

Run them with a config whose `split` is `heldout`. Do not tune prompts on the results.
