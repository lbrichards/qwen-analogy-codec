# Model Availability Report

## Llama 3 8B

Status: C

Local copy: none

Format: none

Gating: gated-manual

User has access: no; after registering the existing local Hugging Face credential for account `andol`, HEAD requests to both Llama 3 8B variants still return gated 403 responses

Disk required if download needed: about 16.1 GB for either `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-8B-Instruct` safetensors weights, plus tokenizer/config overhead

Action required from user: accept/request access in a browser for the desired Llama repository, then re-run a HEAD/access check:

- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Notes:

- No matching full Hugging Face snapshot was found in `~/.cache/huggingface/hub/`.
- No matching local sibling directory was found under `/Users/larry/Development`.
- No matching GGUF/AWQ/quantized copy was found in LM Studio, Ollama, or the searched development directories.
- Hugging Face metadata reports both Llama 3 8B repos as `gated: manual`. With the current `andol` token, both `config.json` and the first safetensors shard return gated 403 responses, so this account does not currently have access.

## Gemma 2 9B

Status: B

Local copy: none

Format: none

Gating: gated-manual, but access is already granted for the current `andol` token

User has access: yes

Disk required if download needed:

- `google/gemma-2-9b-it`: about 18.5 GB safetensors weights, plus tokenizer/config overhead
- `google/gemma-2-9b`: about 37.0 GB safetensors weights, plus tokenizer/config overhead

Action required from user: none for access; Gemma can be downloaded now using the configured HF token. Choose `google/gemma-2-9b-it` if you want the smaller instruct variant.

- [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

Notes:

- No matching full Hugging Face snapshot was found in `~/.cache/huggingface/hub/`.
- No matching local sibling directory was found under `/Users/larry/Development`.
- No matching GGUF/AWQ/quantized copy was found in LM Studio, Ollama, or the searched development directories.
- Hugging Face metadata reports both Gemma 2 9B repos as `gated: manual`, but the current `andol` token has access. HEAD checks succeeded for both `config.json` and the first safetensors shard for `google/gemma-2-9b` and `google/gemma-2-9b-it`.

## Authentication

HF CLI logged in: yes, as `andol`

Token has Llama access: no

Token has Gemma access: yes

Details:

- `hf auth whoami` reports user `andol`.
- The token was already present in the local git/macOS credential store for `huggingface.co`; it has now been registered with the Hugging Face token store so HF CLI and `huggingface_hub` can use it.
- No `HF_HOME`, `HF_TOKEN`, `HUGGINGFACE_*`, or `TRANSFORMERS_CACHE` overrides were present in the active environment.
- No Hugging Face cache overrides were found in `~/.zshrc`, `~/.zshenv`, `~/.zprofile`, `~/.zlogin`, or project-local `.env` files.

## Disk space

Available: about 883 GB free on `/System/Volumes/Data`, covering both `/Users/larry/Development` and `~/.cache`.

Sufficient for both: yes

Even the larger pairing, Llama 3 8B plus base Gemma 2 9B, should require roughly 53 GB for safetensors weights before cache overhead, which is comfortably below the available free space.

## Recommendation

Neither Llama 3 8B nor Gemma 2 9B is currently available locally in a full Hugging Face format, but the machine now has a working HF token configured for account `andol`. The path of least resistance is Gemma, specifically `google/gemma-2-9b-it`: access is already granted and the required safetensors weights are about 18.5 GB. Llama 3 8B still requires browser-side access approval for this account before it can be downloaded. If the cross-family experiment can proceed with Gemma as the second model, no additional browser action is needed; if Llama is required, accept/request access on Hugging Face first and re-run the access check.
