# Model Availability Report

## Llama 3 8B

Status: C

Local copy: none

Format: none

Gating: gated-manual

User has access: no configured local token; current unauthenticated HEAD requests are gated

Disk required if download needed: about 16.1 GB for either `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-8B-Instruct` safetensors weights, plus tokenizer/config overhead

Action required from user: log into Hugging Face locally after accepting/requesting access in a browser:

- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Notes:

- No matching full Hugging Face snapshot was found in `~/.cache/huggingface/hub/`.
- No matching local sibling directory was found under `/Users/larry/Development`.
- No matching GGUF/AWQ/quantized copy was found in LM Studio, Ollama, or the searched development directories.
- Hugging Face metadata reports both Llama 3 8B repos as `gated: manual`. Without a configured token, both `config.json` and the first safetensors shard return gated 401 responses.

## Gemma 2 9B

Status: C

Local copy: none

Format: none

Gating: gated-manual

User has access: no configured local token; current unauthenticated HEAD requests are gated

Disk required if download needed:

- `google/gemma-2-9b-it`: about 18.5 GB safetensors weights, plus tokenizer/config overhead
- `google/gemma-2-9b`: about 37.0 GB safetensors weights, plus tokenizer/config overhead

Action required from user: log into Hugging Face locally after accepting/requesting access in a browser:

- [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

Notes:

- No matching full Hugging Face snapshot was found in `~/.cache/huggingface/hub/`.
- No matching local sibling directory was found under `/Users/larry/Development`.
- No matching GGUF/AWQ/quantized copy was found in LM Studio, Ollama, or the searched development directories.
- Hugging Face metadata reports both Gemma 2 9B repos as `gated: manual`. Without a configured token, both `config.json` and the first safetensors shard return gated 401 responses.

## Authentication

HF CLI logged in: no

Token has Llama access: unknown; no local token is configured

Token has Gemma access: unknown; no local token is configured

Details:

- `hf auth whoami` reports `Not logged in`.
- No `HF_HOME`, `HF_TOKEN`, `HUGGINGFACE_*`, or `TRANSFORMERS_CACHE` overrides were present in the active environment.
- No Hugging Face cache overrides were found in `~/.zshrc`, `~/.zshenv`, `~/.zprofile`, `~/.zlogin`, or project-local `.env` files.

## Disk space

Available: about 883 GB free on `/System/Volumes/Data`, covering both `/Users/larry/Development` and `~/.cache`.

Sufficient for both: yes

Even the larger pairing, Llama 3 8B plus base Gemma 2 9B, should require roughly 53 GB for safetensors weights before cache overhead, which is comfortably below the available free space.

## Recommendation

Neither Llama 3 8B nor Gemma 2 9B is currently available locally in a full Hugging Face format, and there is no configured HF token on this machine. Both model families are gated on Hugging Face, with current API metadata reporting `gated: manual`; I would not assume auto-approval from the local metadata alone. The path of least resistance is to choose the instruct variant you want for the cross-family experiment, complete the browser-side access/license step on Hugging Face, then run `hf auth login` locally and verify access with a metadata/HEAD check before downloading. If minimizing disk is a priority, `meta-llama/Meta-Llama-3-8B-Instruct` (~16.1 GB) or `google/gemma-2-9b-it` (~18.5 GB) are the practical targets; base `google/gemma-2-9b` is substantially larger (~37.0 GB).
