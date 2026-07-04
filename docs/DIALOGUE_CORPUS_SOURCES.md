# Dialogue Corpus Source Policy

This project keeps the 100GB dialogue-cleaning target strict by default. Source breadth is useful only when license, provenance, and quality gates remain explicit.

## Default Sources

| Key | Dataset | License | Why it is enabled |
| --- | --- | --- | --- |
| `oasst1` | `OpenAssistant/oasst1` | Apache-2.0 | Human assistant conversation trees with review metadata. |
| `oasst2` | `OpenAssistant/oasst2` | Apache-2.0 | Larger follow-up OpenAssistant release; same tree/path converter. |
| `wildchat` | `allenai/WildChat-1M` | ODC-BY | Real user-chat data with redaction metadata; strict PII gates stay on. |
| `aya_dataset` | `CohereLabs/aya_dataset` | Apache-2.0 | Human-annotated multilingual prompt/completion pairs; useful for Chinese/English balance. |
| `aya_collection_translated_dolly` / `aya_collection_flan_cot` / `aya_collection_flan_qa` | `CohereLabs/aya_collection` | Apache-2.0 | Large translated Aya subsets; cleaner accepts only Chinese/English rows and still enforces strict quality gates. |
| `helpsteer2` | `nvidia/HelpSteer2` | CC-BY-4.0 | Human preference annotations; final DOPA scoring still filters by coding/tool/security relevance. |
| `helpsteer3_preference` | `nvidia/HelpSteer3`, `preference` subset | CC-BY-4.0 | Multilingual/code preference rows; keeps only the human-preferred response. |

## Optional Sources

| Key | Dataset | License | Default | Reason |
| --- | --- | --- | --- | --- |
| `hh_rlhf` | `Anthropic/hh-rlhf` | MIT | off | Useful safety/preference material, but contains many harmful or adversarial prompts; not language-flow default. |
| `ultrachat_200k` | `HuggingFaceH4/ultrachat_200k` | MIT | off | Synthetic chat source; useful for later ablations, not for the strict real-dialogue pass. |

## Restricted Sources

| Key | Dataset | License | Required flag | Reason |
| --- | --- | --- | --- | --- |
| `lmsys_chat_1m` | `lmsys/lmsys-chat-1m` | LMSYS-Chat-1M Dataset License Agreement | `--allow-restricted-license` | Real and large, but the agreement restricts transfer, redistribution, sublicensing, and may require deletion on request. It must never be enabled silently. |

## Fast Filtering

The Python cleaner remains the authoritative quality gate. The Rust binary `rust/dopa_dialogue_filter` is only a hard-reject prefilter for cheap checks:

- license allowlist
- language allowlist
- turn count
- character length
- PII/secret-like content
- mojibake and control characters
- bad code artifacts
- URL-only rows
- unsafe passcode brute-force patterns
- financial market advice
- extreme repetition

Use `--fast-filter auto` to use the Rust binary when `target/release/dopa_dialogue_filter.exe` exists. Use `--fast-filter on` to require it.
