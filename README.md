# Kalki Intelligence

**Kalki** is a local AI-companion research project. Its first model, **Kalki Mini**, is a decoder-only transformer implemented in PyTorch and initialized from random weights. It follows the educational, step-by-step GPT approach in [Raschka's LLMs-from-scratch repository](https://github.com/rasbt/LLMs-from-scratch), with an experimental gated prefix-summary route called the **Compass Path**.

## Important scope

Kalki Mini is a small model-building project, not a ready-made all-knowing assistant. Random initialization means it starts with **no learned language knowledge**. It needs a large, legally usable text corpus and substantial training before it can produce coherent answers; a small laptop-sized run is a learning prototype. The 36M-parameter configuration is not expected to answer arbitrary questions well.

This project does **not** load another model, pretrained weights, a hosted LLM, or model API. The GPT-2 byte-pair tokenizer is used only to turn text into token IDs; all Kalki neural-network weights start random. The Compass Path is an original implementation experiment here, not a claim of proven or globally unprecedented research novelty.

## Laptop target

The development machine detected for this project has an NVIDIA RTX 5060 Laptop GPU with 8 GiB VRAM, about 16 GiB system RAM, and 24 logical CPU threads. Defaults are deliberately modest: 36M parameters, 256-token context, batch size 2, and mixed precision on CUDA. These settings are a starting point, not a measured performance guarantee. Longer training runs and larger data are possible, but training a competitive general-purpose foundation model is far beyond this laptop's practical compute and data budget.

## Setup

Use Python 3.10+ and install a PyTorch build suitable for the installed GPU from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Data and checkpoints are ignored by Git. To prepare a small human-authored conversational corpus, run `python prepare_oasst.py`. It fetches OASST1 and filters to English reviewed prompt/reply pairs marked non-synthetic, with examples flagged for privacy or harmful content excluded. The source dataset is Apache-2.0; see `data/README.md`. You can also provide your own licensed `.txt` files under `data/`.

## Train from scratch

```powershell
python prepare_oasst.py
python train.py --corpus data --steps 5000
```

Training writes `checkpoints/kalki-mini.pt`. With a small corpus, it will quickly memorize text and will not become a broad-knowledge assistant. More diverse, high-quality, appropriately licensed text and much more compute are needed for broader capability.

## Chat

```powershell
python chat.py
```

The model speaks as Kalki, but its answers are only as capable as its training. It may produce incorrect or incoherent text, particularly before meaningful training.

## Online continual learning

Add a new `.txt` file or a folder of new `.txt` files, with examples separated by blank lines, then run:

```powershell
python continue_learning.py --new-data data/updates --steps 400
```

In chat, `/teach question => verified answer` appends a local example for review under `data/updates/`; approved feedback is not trained automatically. Run the command below on approved text to perform an update. The updater trains the **same Kalki weights** on the new text and samples from a local reservoir replay buffer to reduce catastrophic forgetting. It reports before/after language-model loss on new and replay text, saves the prior checkpoint beside the active one as `kalki-mini.previous.pt`, and updates the active checkpoint only after training completes. This is online-style incremental fine-tuning with replay, not automatic truth verification or a guarantee against forgetting. Review and curate every update corpus; bad, private, or poisoned text can degrade Kalki. Replay files and checkpoints stay local and are excluded from commits.

## Model outline

- Decoder-only causal transformer, 8 layers, width 384, 6 attention heads.
- Rotary position embeddings, tied token/output embeddings, and a 4× GELU feed-forward block.
- **Compass Path:** a learned gate mixes each token's representation with a cumulative prefix summary, so it can draw on compressed prior context while preserving causal ordering.
- Approx. 35.8M trainable parameters with the default settings; 256-token context.
- GPT-2 tokenizer vocabulary only; no GPT-2 model parameters or other model is loaded.

## Reference

This project uses the public book-code repository as an educational reference for the GPT implementation sequence and training workflow. The Kalki source is independently written for this project. See the upstream repository and its license before reusing upstream code: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).
