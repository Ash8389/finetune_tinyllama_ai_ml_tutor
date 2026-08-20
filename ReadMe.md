# 🦙 TinyLlama AI/ML Tutor

### Teaching TinyLlama to explain AI/ML like a tutor — using QLoRA

> A QLoRA fine-tuned version of TinyLlama-1.1B-Chat-v1.0 specialized for AI/ML education — structured explanations, coding questions, debugging, mathematics, and interview preparation.

<p align="center">

[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/ash270/tinyllama-ai-ml-tutor-qlora)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Ash8389/finetune_tinyllama_ai_ml_tutor)
[![QLoRA](https://img.shields.io/badge/Training-QLoRA-blue)](https://arxiv.org/abs/2305.14314)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

</p>

---

## 🚀 What if a Small LLM Could Be Specialized?

Large general-purpose models know a lot.

But knowing something and **teaching it consistently** are different problems.

This project takes **TinyLlama-1.1B-Chat-v1.0** and specializes it into **TinyLlama AI/ML Tutor** using **QLoRA + Supervised Fine-Tuning**.

The goal wasn't to make the model bigger. The goal was to make the model **better at one specific job**.

---

## 🎯 What Changed After Fine-Tuning?

| Capability | Base TinyLlama | Fine-Tuned Tutor |
|---|---|---|
| General conversation | ✅ | ✅ |
| AI/ML knowledge | General | 🎯 Specialized |
| Teaching-oriented answers | Inconsistent | ✅ Consistent |
| Structured explanations | Inconsistent | ✅ |
| Definition → Intuition → Example flow | ❌ Not guaranteed | ✅ |
| AI/ML comparisons | General | 🎯 Specialized |
| Debugging questions | General | 🎯 Specialized |
| Interview-style questions | General | 🎯 Specialized |
| Mathematical explanations | General | 🎯 AI/ML focused |
| Tutor-style response format | ❌ | ✅ |

### The target behavior

The fine-tuned model was trained to organize explanations as:

```text
Definition
   ↓
Why It Matters
   ↓
Intuition
   ↓
Example
   ↓
Key Points
   ↓
Summary
```

This is the primary behavioral specialization of the model.

---

## 🔬 Base Model vs. Fine-Tuned Model — Real Examples

The repository contains **200 side-by-side evaluations** (`evaluation/result.jsonl`) using held-out test questions. Both models receive the exact same question — the difference is entirely in *how they respond*. Below are four real, unedited examples pulled directly from that file.

### 1. "What is QLoRA, and what problem does it solve?"

<table>
<tr><th>Base TinyLlama</th><th>Fine-Tuned Tutor</th></tr>
<tr>
<td valign="top">

QLoRA (Quantized Logistic Regression) is a technique that uses a combination of quantization and logistic regression to solve the problem of contextual variation in machine learning... The combination of the two techniques of quantization and logistic regression has been shown to result in better generalization performance than standard logistic regression algorithms.

</td>
<td valign="top">

**Definition**
Quantization represents model weights using fewer bits.

**Why It Matters**
Lower-bit storage reduces memory requirements.

**Intuition**
A 4-bit representation uses far fewer bits per weight than FP16.

**Key Points**
- Quantization represents model weights using fewer bits.
- Lower-bit storage reduces memory requirements.
- The technique has trade-offs and should be evaluated for the intended task.

**Summary**
Quantization represents model weights using fewer bits. Lower-bit storage reduces memory requirements.

</td>
</tr>
</table>

> ⚠️ Notably, the base model doesn't just answer poorly here — it confidently misdefines QLoRA as "**Q**uantized **Lo**gistic **R**egression," which is factually wrong (QLoRA = **Q**uantized **Lo**w-**R**ank **A**daptation). The fine-tuned model correctly grounds its answer in quantization concepts instead.

### 2. "Why is Gradient Descent useful? (focus: the learning rate controls the size of an optimization step)"

<table>
<tr><th>Base TinyLlama</th><th>Fine-Tuned Tutor</th></tr>
<tr>
<td valign="top">

Gradient Descent is useful in modern machine learning because it is a commonly used method for optimizing parameter estimates... The optimization process involves several iterations, each time with a larger learning rate. Gradient Descent is a stochastic method, where each iteration selects a random step along the direction of the gradient vector...

</td>
<td valign="top">

**Definition**
The learning rate controls the size of an optimization step.

**Why It Matters**
A very large rate can cause instability, while a very small rate can make learning slow.

**Intuition**
If the gradient is 2 and the rate is 0.1, the parameter update is -0.2.

**Key Points**
- The learning rate controls the size of an optimization step.
- A very large rate can cause instability, while a very small rate can make learning slow.

**Summary**
The learning rate controls the size of an optimization step. A very large rate can cause instability, while a very small rate can make learning slow.

</td>
</tr>
</table>

> The base model rambles about stochasticity and convergence without ever isolating what was actually asked (the learning rate's role). The fine-tuned model goes straight to the point and backs it with a concrete number.

### 3. "How does Self-Attention work? (focus: scores divided by √d_k before softmax)"

<table>
<tr><th>Base TinyLlama</th><th>Fine-Tuned Tutor</th></tr>
<tr>
<td valign="top">

Self-attention works at a high level by utilizing a combination of context variation and attention scores to calculate the attention scores. Context variation: The context variation is a key component of Self-attention. It uses the attention scores to transform the query and key tokens...

</td>
<td valign="top">

**Definition**
The attention scores are divided by sqrt(d_k) before softmax.

**Why It Matters**
Scaling prevents large dot products from making softmax unnecessarily sharp as the key dimension grows.

**Intuition**
If d_k=64, scores are divided by 8 before softmax.

**Summary**
The attention scores are divided by sqrt(d_k) before softmax. Scaling prevents large dot products from making softmax unnecessarily sharp as the key dimension grows.

</td>
</tr>
</table>

> The base answer is circular ("uses attention scores to calculate the attention scores"). The fine-tuned answer gives the actual mechanism, a numeric example (d_k=64 → divide by 8), and the reason it matters.

### 4. "What does `detach()` do in PyTorch Autograd?"

<table>
<tr><th>Base TinyLlama</th><th>Fine-Tuned Tutor</th></tr>
<tr>
<td valign="top">

PyTorch Autograd, also known as PyTorch Propagation-based Graph-based backpropagation (PyTorch-Propagation), is a method used in deep learning algorithms... The main equation of PyTorch Autograd is the Adam optimizer's gradient update rule...

</td>
<td valign="top">

**Definition**
`detach()` returns a tensor detached from the current autograd graph.

**Why It Matters**
It is useful when a value should not propagate gradients into earlier computation.

**Intuition**
`y = x.detach()` prevents gradients through `y` from flowing back through that graph.

**Example**
Detached tensors can still share storage with the original tensor.

**Summary**
`detach()` returns a tensor detached from the current autograd graph. It is useful when a value should not propagate gradients into earlier computation.

</td>
</tr>
</table>

> The base model doesn't even address `detach()` — it invents an unrelated explanation about the Adam optimizer. The fine-tuned model answers the actual question asked.

**What changed, consistently across all 200 evaluations:**
- ❌ Base: generic, sometimes factually wrong, occasionally answers a different question than the one asked
- ✅ Fine-tuned: follows the teaching structure every time
- ✅ Fine-tuned: gives grounded intuition with concrete numbers/code
- ✅ Fine-tuned: stays on-topic and summarizes key points

See all 200 comparisons yourself in [`evaluation/result.jsonl`](./evaluation/result.jsonl).

---

## 📊 Evaluation

The project evaluates the base and fine-tuned models on 200 held-out test questions. Each question is sent to both models and responses are stored side-by-side in `evaluation/result.jsonl`.

**Inference configuration:**
```
max_new_tokens = 300
temperature    = 0.7
sampling       = enabled
```

The current evaluation focuses primarily on **qualitative behavior** — response structure, AI/ML specialization, teaching style, consistency, explanatory flow, and ability to answer different question types — rather than a single automatic score.

> **Note:** This project does not currently claim a percentage improvement in accuracy or quality. Quantitative benchmarking is planned (see Roadmap).

---

## 🧠 Why QLoRA?

Fine-tuning an entire 1.1B parameter model is unnecessary for this experiment.

```
TinyLlama
   │
   ├── 4-bit NF4 Quantization
   │
   └── LoRA Adapters
           │
           ▼
      Fine-Tuned Tutor
```

QLoRA makes it possible to train the model with significantly lower memory requirements while keeping the base model quantized.

**Quantization config:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**LoRA config:**
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

The model is prepared for k-bit training (`prepare_model_for_kbit_training`) and then LoRA adapters are attached (`get_peft_model`).

---

## 📚 Dataset

The training dataset contains **2,000 synthetically generated AI/ML tutor examples**.

```
Groq
  ↓
llama-3.1-8b-instant
  ↓
Synthetic AI/ML Q&A
  ↓
Validation
  ↓
Training Dataset
```

**Topics covered:** Self Attention, Multi-Head Attention, Gradient Descent, PyTorch Autograd, LoRA, QLoRA (spanning the Transformers, Machine Learning, PyTorch, and Fine-Tuning domains).

**Question types cycled through (9):** `concept`, `why`, `how`, `comparison`, `code`, `debugging`, `mathematics`, `interview`, `real_world`.

Each answer is generated around the target teaching structure — **Definition → Why It Matters → Intuition → Example → Key Points → Summary** — and validated (exactly one user + one assistant turn, non-trivial answer length) before being kept.

### Dataset split

| Dataset | Examples |
|---|---|
| Total | 2,000 |
| Training | 1,600 |
| Validation | 200 |
| Test | 200 |

```
2,000 Examples
      │
      ├── 1,600 Train
      ├──   200 Validation
      └──   200 Test
```

80/10/10 split, seed = 40.

---

## 🏋️ Training

Training was performed using 🤗 Transformers, PEFT, TRL (`SFTTrainer`), and bitsandbytes.

| Parameter | Value |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Method | QLoRA |
| Quantization | 4-bit NF4 |
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 4 |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Optimizer | `paged_adamw_8bit` |
| Precision | FP16 |
| Warmup Ratio | 0.03 |
| Seed | 42 |

### Training results

Training completed for 3 epochs / 1,365 steps. The best checkpoint was selected using evaluation loss.

| Metric | Result |
|---|---|
| Evaluation Loss | 0.625 |
| Evaluation Mean Token Accuracy | 0.820 |
| Train Loss (final step) | ~0.60 |
| Train Mean Token Accuracy (final step) | ~0.82 |

- Best checkpoint: `outputs/checkpoint-1365/`
- Final adapter: `outputs/final_adapter/`

---

## 🏗️ Architecture

```
                   ┌─────────────────────┐
                   │  Synthetic Dataset  │
                   │     2,000 Q&A       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Data Cleaning &     │
                   │ Chat Formatting     │
                   └──────────┬──────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ TinyLlama 1.1B                │
              │ 4-bit NF4 Quantized           │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │      QLoRA       │
                    │ LoRA Adapters    │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Fine-Tuned AI/ML     │
                  │ Tutor Adapter        │
                  └──────────┬───────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Base vs Fine-Tuned     │
                 │ Evaluation             │
                 └────────────┬───────────┘
                              │
                              ▼
                         FastAPI API
```

---

## 📁 Project Structure

```
finetune_tinyllama_ai_ml_tutor/
│
├── data/
│   └── raw/
│       ├── generated.jsonl
│       ├── train.jsonl
│       ├── validate.jsonl
│       └── test.jsonl
│
├── notebooks/
│   └── run_pipeline.ipynb
│
├── evaluation/
│   └── result.jsonl
│
├── outputs/
│   ├── checkpoint-1200/
│   ├── checkpoint-1365/
│   └── final_adapter/
│
├── src/
│   ├── create_data/     # Synthetic dataset generation (Groq)
│   ├── data/             # Loading, cleaning, chat-formatting, splitting
│   ├── model/             # 4-bit quantization + LoRA config/attach
│   ├── tokenizer/         # Tokenizer loader
│   ├── train/             # SFTConfig, SFTTrainer, adapter saving
│   ├── inference/         # Base + fine-tuned inference pipeline
│   └── api/               # FastAPI service (base vs fine-tuned comparison)
│
├── requirements.txt
└── README.md
```

The repository contains the complete pipeline — from synthetic data generation through training, inference, evaluation, and API serving.

---

## ⚡ Quick Start

### 1. Clone
```bash
git clone https://github.com/Ash8389/finetune_tinyllama_ai_ml_tutor.git
cd finetune_tinyllama_ai_ml_tutor
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

Open `notebooks/run_pipeline.ipynb`. It walks through:

```
Dataset → Cleaning → Tokenizer → 4-bit Model → LoRA → Training → Inference → Evaluation
```

---

## 🤗 Use the Fine-Tuned Model

The trained LoRA adapter is available on Hugging Face:

👉 **[ash270/tinyllama-ai-ml-tutor-qlora](https://huggingface.co/ash270/tinyllama-ai-ml-tutor-qlora)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

model = PeftModel.from_pretrained(
    base_model,
    "ash270/tinyllama-ai-ml-tutor-qlora"
)
```

---

## 🌐 Compare Both Models Through API

The project includes a FastAPI service that loads both the original and fine-tuned models.

Start the API:
```bash
uvicorn src.api.main:app --reload
```

Then:
```
GET /chat?q=What is gradient descent?
```

Response:
```json
{
  "finetuned_answer": "...",
  "basemodel_answer": "..."
}
```

This makes the project more than a training experiment — you can directly observe the behavioral difference between the two models.

---

## 💡 What I Learned

This project was built to understand the complete LLM fine-tuning lifecycle:

```
Synthetic Data Generation
        ↓
Data Validation
        ↓
Dataset Preparation
        ↓
Tokenization
        ↓
4-bit Quantization
        ↓
LoRA
        ↓
Supervised Fine-Tuning
        ↓
Model Evaluation
        ↓
Inference
        ↓
API Deployment
```

The interesting part wasn't simply training a model. It was understanding how data + training objective + LoRA configuration + evaluation can change the behavior of a relatively small language model.

---

## ⚠️ Limitations

This is an experimental AI/ML tutoring model.

**Dataset limitations** — the model was trained on only 2,000 examples, covering a relatively narrow set of AI/ML topics.

**Synthetic data** — the training data was generated by another LLM (Groq → `llama-3.1-8b-instant`). As shown in the QLoRA example above, the base model itself is prone to confident factual errors, and since the *training data* comes from a similarly-sized LLM, the fine-tuned model can inherit inaccuracies, biases, or repetitive phrasing from the generated dataset.

**Not a general-purpose replacement** — the model is specifically optimized for AI/ML tutoring and should not be expected to outperform larger general-purpose models on unrelated tasks. Important technical claims should still be independently verified.

---

## 🗺️ What's Next?

The project is intentionally still evolving.

- [ ] Expand the AI/ML topic coverage
- [ ] Add automated evaluation metrics
- [ ] Add human evaluation
- [ ] Compare against additional small language models
- [ ] Improve synthetic-data diversity
- [ ] Investigate dataset quality vs. model behavior
- [ ] Merge LoRA weights into a standalone model
- [ ] Build a web-based demo
- [ ] Deploy the inference API

---

## 🤝 Why This Project?

This project is an experiment in specializing small language models.

Instead of asking *"How can I use a larger model?"*, it explores:

**"How much can we improve a small model by giving it better data and a specialized training objective?"**

That is the core idea behind this project.

---

## 📚 References

- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT](https://huggingface.co/docs/peft)
- [TRL](https://huggingface.co/docs/trl)
- [QLoRA — Dettmers et al.](https://arxiv.org/abs/2305.14314)
- [Groq](https://groq.com/)

---

## 👨‍💻 Author

**Ashish Kumar Jha**

- GitHub: [@Ash8389](https://github.com/Ash8389)
- Hugging Face: [@ash270](https://huggingface.co/ash270)

⭐ If you found the project interesting, consider giving it a star!