# API Distillation Setup

Create local ignored config and env files at the project root:

```powershell
Copy-Item configs\distill_providers.example.json configs\distill_providers.json
Set-Content -Path .env -Value 'DOPA_DISTILL_API_KEY=replace-with-your-key'
```

Edit `configs\distill_providers.json` for your private OpenAI-compatible endpoint and teacher model.
Do not commit `.env`, provider keys, provider-specific local config, seed prompts, or generated outputs.

Dry-run:

```powershell
.\.venv\Scripts\python.exe scripts\distill_dialogue.py --provider openai_compatible_teacher --dry-run
```

First real run after selecting a private seed file:

```powershell
.\.venv\Scripts\python.exe scripts\distill_dialogue.py --provider openai_compatible_teacher --seed-file PATH\TO\seed_prompts.jsonl --limit 10 --sleep-seconds 1
```
