# Step-by-step: Run Trans-EnV with Gemma on a cluster

These steps assume you have SSH access to a Linux cluster with NVIDIA GPUs and CUDA.

---

## Part 1: One-time setup on the cluster

### Step 1: Log in and get a GPU node

```bash
ssh your_username@cluster.address.edu
# If your cluster uses a scheduler (SLURM, etc.), request an interactive GPU node, e.g.:
# srun --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash
```

### Step 2: Clone the repo (or copy it from your Mac)

```bash
cd ~  # or your preferred directory
git clone <your-repo-url> ESLTransformations
cd ESLTransformations
```

If you already have the repo on your Mac, you can sync it:

```bash
# From your Mac:
rsync -avz --exclude .venv --exclude __pycache__ /Users/serenaliu/ESLTransformations/ your_username@cluster.address.edu:~/ESLTransformations/
```

### Step 3: Create a virtual environment

```bash
cd ~/ESLTransformations
python3 -m venv .venv
source .venv/bin/activate
```

Use Python 3.9, 3.10, or 3.11 (check with `python3 --version`).

### Step 4: Install PyTorch with CUDA

Check your cluster’s CUDA version:

```bash
nvidia-smi
```

Then install the matching PyTorch (example for **CUDA 12.1**):

```bash
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

- For **CUDA 11.8**: use `cu118` instead of `cu121` in the URL.
- For **CUDA 12.4**: try `cu124` if available, or use the latest compatible wheel from [PyTorch](https://pytorch.org/get-started/locally/).

### Step 5: Install project dependencies

```bash
pip install -r requirements.txt
```

If vLLM fails to build, retry after Step 4 and ensure the PyTorch version matches what vLLM expects.

### Step 6: (Optional) Set environment variables

Create a `.env` in the project root if you use Gemini or OpenAI elsewhere:

```bash
# Optional
export DATA_DIR=~/.cache/huggingface
export MODEL_DIR=~/.cache/huggingface
```

For Gemma via vLLM you do **not** need `GOOGLE_API_KEY` or `OPENAI_API_KEY`.

---

## Part 2: Run Trans-EnV with Gemma

You need two processes: (1) vLLM server with Gemma, (2) the main script. Run them on the **same node** so the script can use `localhost:6001`.

### Step 7: Start the vLLM server (first terminal/session)

Keep this running for the whole experiment.

```bash
cd ~/ESLTransformations
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model google/gemma-2-27b-it --port 6001
```

Wait until you see something like “Application startup complete” and the server is listening on port 6001. Leave this terminal open.

If the cluster has a different hostname per node, note the current hostname (`hostname`) for Step 9.

### Step 8: Open a second terminal on the same GPU node

SSH again (or open another tab) and, if using a scheduler, get a shell on the **same node** where vLLM is running. For SLURM you might do:

```bash
srun --gres=gpu:1 --nodelist=<same-node> --mem=16G --time=2:00:00 --pty bash
cd ~/ESLTransformations && source .venv/bin/activate
```

### Step 9: Run the Trans-EnV pipeline

**Alpacafarm sample (open-ended, CEFR A + L1, Gemma):**

```bash
cd ~/ESLTransformations
source .venv/bin/activate
python src/run/main.py \
  --dataset_name alpacafarm \
  --model_name google/gemma-2-27b-it \
  --tokenizer google/gemma-2-27b-it \
  --sampling \
  --task_name openended_esl \
  --cefr_level A \
  --l1 Arabic \
  --save_path ./outputs/alpacafarm/esl \
  --file_name A_arabic_sample \
  --batch_size 15 \
  --data_path . \
  --port_num 6001
```

If the script runs on a **different node** than the vLLM server, replace `localhost` by the vLLM node’s hostname and ensure the cluster allows traffic on port 6001 between nodes. The code may need to be told the base URL (e.g. `http://vllm-node:6001`); check `src/utils/model_utils.py` for how the client is built.

**Paper-style run (MMLU, L1, CEFR A):**

You need the MMLU CEFR asset at `assets/cefr/mmlu/A.csv`. If you have it:

```bash
python src/run/main.py \
  --batch_size 15 \
  --save_path ./outputs/mmlu/l1 \
  --file_name A_arabic \
  --l1 Arabic \
  --task_name L1 \
  --cefr_level A \
  --port_num 6001 \
  --dataset_name mmlu \
  --model_name google/gemma-2-27b-it \
  --tokenizer google/gemma-2-27b-it \
  --data_path .
```

### Step 10: Check outputs

Results will be under the paths you passed to `--save_path` and `--file_name`, e.g.:

- `./outputs/alpacafarm/esl/A_arabic_sample.jsonl`
- `./outputs/alpacafarm/esl/A_arabic_sample.pk`

To make the JSONL readable (e.g. on your Mac after copying the file):

```bash
python -c "
import json, sys
for i, line in enumerate(open(sys.argv[1])):
    if line.strip():
        print('---', i+1, '---')
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
        print()
" ./outputs/alpacafarm/esl/A_arabic_sample.jsonl
```

---

## Part 3: (Optional) Run the script from your Mac

If you prefer to run `main.py` on your Mac and only use the cluster for vLLM:

1. **On the cluster:** Start vLLM as in Step 7.
2. **On your Mac:** Forward port 6001:
   ```bash
   ssh -L 6001:localhost:6001 your_username@cluster.address.edu
   ```
   (In another terminal, or in the background.)
3. **On your Mac:** From the repo directory, with a venv that has the project deps (no need for vLLM):
   ```bash
   python src/run/main.py --dataset_name alpacafarm --model_name google/gemma-2-27b-it --tokenizer google/gemma-2-27b-it --sampling --task_name openended_esl --cefr_level A --l1 Arabic --save_path ./outputs/alpacafarm/esl --file_name A_arabic_sample --batch_size 15 --data_path . --port_num 6001
   ```
   The script will use `localhost:6001`, which SSH forwards to the cluster’s vLLM.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ModuleNotFoundError: No module named 'vllm'` | Activate the same venv where you ran `pip install -r requirements.txt`. |
| `Connection refused` to port 6001 | vLLM and main.py must reach the same host:port. Use same node or correct hostname and open firewall. |
| Out of GPU memory | Use a smaller batch size, e.g. `--batch_size 5`. |
| CUDA version mismatch | Run `nvidia-smi` and install the matching PyTorch wheel (cu118, cu121, etc.). |
| No `assets/cefr/mmlu/` | MMLU L1 run needs that asset; use alpacafarm + `openended_esl` for a run that doesn’t need it. |
