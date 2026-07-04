# Dialogue Source Notes

Updated: 2026-07-04

## Active continuation sources

- `Anthropic/hh-rlhf`
  - Status: enabled in `watch_dialogue_corpus.ps1`.
  - Reason: large human-feedback preference corpus with MIT license; useful for safety and preference-style dialogue after strict filtering.
  - Risk: contains unsafe prompts and rejected answers; keep `unsafe_security`, PII, and category filters enabled.

- `databricks/databricks-dolly-15k`
  - Status: enabled in `watch_dialogue_corpus.ps1`.
  - Reason: human-generated instruction/response records, not generative-AI-authored; small but high-signal.
  - Risk: English only and CC-BY-SA-3.0; downstream dataset manifests must preserve attribution/share-alike notes.

## Candidate sources not enabled by default

- `lmsys/lmsys-chat-1m`
  - Reason to consider: real-world user/chat traffic at large scale.
  - Reason not enabled: dataset license agreement has transfer and deletion obligations; keep it behind explicit user approval.

- `HuggingFaceH4/ultrafeedback_binarized`
  - Reason to consider: MIT-licensed preference/SFT format with chosen/rejected responses.
  - Reason not enabled: model-generated preference data; useful for later alignment experiments, not the current high-quality human dialogue target.

- `m-a-p/COIG-CQIA`
  - Reason to consider: Chinese, human-verified fields, multiple Chinese task subsets.
  - Reason not enabled: license/copyright metadata is not clean enough for automatic inclusion; needs manual license review before training use.
