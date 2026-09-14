# Qwen Fine-Tuned Customer Support AI Agent

A locally deployed customer support AI agent built around a **fine-tuned Qwen2.5-3B-Instruct model**.

The main focus of this project is the complete **LLM fine-tuning and local deployment pipeline**:

**Dataset → QLoRA Fine-Tuning → LoRA Adapter → Model Merge → GGUF Q8_0 → Ollama → Customer Support Agent**

The project combines a fine-tuned language model with application-level business logic, tool execution, confirmation flows, and safety guardrails.

---

## Project Overview

Customer support systems require more than generating natural-language responses.

A useful support model needs to understand customer intent, extract relevant entities, produce structured outputs, and interact with business operations in a controlled way.

This project explores how a relatively small open-source model can be adapted for this task through **supervised fine-tuning with QLoRA**, then deployed locally as part of an agentic application.

The base model used for training is:

**Qwen/Qwen2.5-3B-Instruct**

The final model is converted to **GGUF Q8_0** and served locally through **Ollama**.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │ Customer Support     │
                    │ Dataset              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Qwen2.5-3B-Instruct  │
                    │ 4-bit Quantization   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ QLoRA Fine-Tuning    │
                    │ SFT + LoRA           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ LoRA Adapter         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Merge with Base      │
                    │ Model                │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ GGUF Q8_0            │
                    │ Conversion            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Ollama               │
                    │ Local Inference      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Customer Support     │
                    │ Agent                │
                    └──────────────────────┘
```

---

## Fine-Tuning Pipeline

The model is fine-tuned using **Supervised Fine-Tuning (SFT)** with **QLoRA**.

### 1. Base Model

```text
Qwen/Qwen2.5-3B-Instruct
```

The model is loaded using 4-bit quantization to reduce GPU memory requirements during training.

### 2. 4-bit Quantization

The training configuration uses:

* `load_in_4bit=True`
* Quantization type: `NF4`
* Compute dtype: `float16`
* Double quantization: enabled

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### 3. LoRA Configuration

Instead of updating all model parameters, the training process adds trainable LoRA adapters.

Current configuration:

| Parameter       |     Value |
| --------------- | --------: |
| LoRA rank (`r`) |        16 |
| LoRA alpha      |        32 |
| LoRA dropout    |      0.05 |
| Bias            |      none |
| Task type       | Causal LM |

LoRA is applied to the main attention and MLP projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

### Trainable Parameters

The training output reports:

```text
Trainable parameters: 29,933,568
Total parameters:     3,115,872,256
Trainable percentage: 0.9607%
```

This means the fine-tuning process updates less than 1% of the model's parameters while adapting the model to the customer-support task.

---

## Training Environment

The training notebook was executed on Google Colab using:

| Component       | Configuration                  |
| --------------- | ------------------------------ |
| GPU             | NVIDIA Tesla T4                |
| GPU Memory      | 15 GB                          |
| CUDA            | 13.0                           |
| Base Model      | Qwen2.5-3B-Instruct            |
| Training Method | QLoRA + SFT                    |
| Precision       | 4-bit model + FP16 computation |

The setup is designed around the constraints of the **Google Colab free environment**.

---

## Dataset

The training dataset contains customer-support conversations represented as structured examples.

The model is trained to identify customer intent and extract relevant entities while producing a structured JSON response.

The dataset covers **15 customer-support intents**, including:

* Payment failure
* Damaged product
* Product availability
* Payment methods
* Refund status
* Order cancellation
* Order tracking
* Refund request
* Order delay
* Product technical questions
* Warranty inquiries
* Refund policy
* Shipping cost
* General complaints
* Order modification

The model is therefore not trained simply to generate generic customer-service text.

Instead, the fine-tuning objective is to make the model behave as a structured **customer-support decision component**.

---

## Structured Output

The fine-tuned model is prompted to return structured JSON containing information such as:

```json
{
  "intent": "order_tracking",
  "entities": {
    "order_id": "98234"
  },
  "requires_tool": true,
  "tool": "get_order_status",
  "requires_confirmation": false,
  "needs_clarification": false
}
```

This structured output allows the application layer to decide what should happen next.

For example:

```text
User message
     │
     ▼
Fine-tuned Qwen
     │
     ▼
Intent + Entities + Required Action
     │
     ▼
Agent Router
     │
     ├── Answer directly
     ├── Ask for clarification
     ├── Execute a tool
     └── Request confirmation
```

---

## Training Process

The fine-tuning workflow is implemented in:

```text
notebooks/qlora_training.ipynb
```

The notebook covers the main stages of the training pipeline:

```text
Load Dataset
     ↓
Format Chat Prompts
     ↓
Load Quantized Qwen Model
     ↓
Configure LoRA
     ↓
Supervised Fine-Tuning
     ↓
Save LoRA Adapter
     ↓
Merge Adapter with Base Model
     ↓
Convert Merged Model to GGUF Q8_0
```

The LoRA adapter is saved after training and then merged back into the base model to produce a standalone fine-tuned model.

---

## Model Merge

After fine-tuning, the LoRA adapter is merged with the original model:

```text
Qwen2.5-3B-Instruct
        +
   LoRA Adapter
        │
        ▼
Merged Fine-Tuned Model
```

The merged model can then be converted into a format suitable for local inference.

---

## GGUF Conversion

The merged Hugging Face model is converted to **GGUF Q8_0** using `llama.cpp`.

The current pipeline contains a single GGUF conversion path:

```bash
python llama.cpp/convert_hf_to_gguf.py /merged_model \
  --outfile /customer-support-ai/models/qwen_support_q8.gguf \
  --outtype q8_0
```

The resulting model is:

```text
qwen_support_q8.gguf
```

This is the model used by the local Ollama deployment.

---

## Local Deployment with Ollama

The GGUF model is deployed locally through Ollama.

The project includes a `Modelfile.txt` defining the model configuration, including:

* Local GGUF model
* Qwen chat template
* Customer-support system instructions
* Structured JSON output requirements
* Stop tokens
* Low-temperature generation

The final inference pipeline is:

```text
User
  │
  ▼
CustomerSupportAgent
  │
  ▼
Ollama
  │
  ▼
Fine-Tuned Qwen
  │
  ▼
Structured JSON
  │
  ▼
Agent Logic
```

---

## Customer Support Agent

The fine-tuned model is integrated into an application-level agent.

The agent is responsible for:

### Intent Detection

Identifying what the customer wants, such as:

```text
order_tracking
order_cancellation
refund_request
payment_failed
product_availability
...
```

### Entity Extraction

Extracting information required for an operation, such as:

```text
order_id
product
payment information
```

### Tool Routing

Depending on the model's structured output, the application can route the request to an appropriate business operation.

Current application-level tools include:

| Tool                 | Purpose                               |
| -------------------- | ------------------------------------- |
| `get_order_status`   | Retrieve order status                 |
| `cancel_order`       | Cancel an order                       |
| `escalate_shipping`  | Escalate shipping issues              |
| `check_inventory`    | Check product inventory               |
| `rag_policy_search`  | Search the project's policy knowledge |
| `register_complaint` | Register a customer complaint         |

---

## Confirmation & Safety Flow

Certain operations should not be executed immediately.

For example, order cancellation requires explicit confirmation from the customer.

```text
Customer
   │
   ▼
"Cancel my order 55443"
   │
   ▼
Model detects cancellation intent
   │
   ▼
Agent asks for confirmation
   │
   ▼
Customer: "Yes"
   │
   ▼
cancel_order()
   │
   ▼
Operation result
```

If the customer rejects the operation:

```text
Customer: "No"
        ↓
Cancellation aborted
```

This separates **model decision-making** from **business-critical execution**.

The language model does not directly modify the underlying business state.

---

## Project Structure

```text
qwen-finetuned-customer-support-agent/
│
├── data/
│
├── notebooks/
│   └── qlora_training.ipynb
│
├── src/
│   └── agent/
│       ├── core.py
│       ├── tools.py
│       └── mock_data.py
│
├── Modelfile.txt
├── test_agent.py
├── README.md
└── .gitignore
```

---

## Example

A customer can ask:

```text
Where is my order 98234?
```

The model can produce a structured decision such as:

```json
{
  "intent": "order_tracking",
  "entities": {
    "order_id": "98234"
  },
  "requires_tool": true,
  "tool": "get_order_status"
}
```

The agent then routes the request to:

```python
get_order_status("98234")
```

For a sensitive operation such as cancellation, the agent introduces a confirmation step before executing the operation.

---

## Why Fine-Tuning?

A general-purpose LLM can already answer customer-support questions.

The purpose of fine-tuning here is different.

The goal is to adapt the model to a **specific behavioral pattern**:

```text
Natural Language
       ↓
Customer Intent
       ↓
Entity Extraction
       ↓
Structured Decision
       ↓
Application Action
```

Instead of relying entirely on increasingly complex prompts, fine-tuning teaches the model the structure and behavior expected by the application.

This project therefore focuses on the practical engineering lifecycle of taking an open-source LLM from:

```text
Base Model
    ↓
Domain Dataset
    ↓
Parameter-Efficient Fine-Tuning
    ↓
Merged Model
    ↓
Local Model Format
    ↓
Production-Oriented Application
```

---

## Current Limitations

This is a portfolio and engineering project rather than a production customer-support platform.

Current limitations include:

* Business data is mock data.
* The application tools are simulated.
* The policy search component is lightweight application-level retrieval rather than a full vector database RAG pipeline.
* Evaluation can be expanded with a larger automated benchmark.
* Production concerns such as authentication, observability, distributed inference, and persistent state are outside the current scope.
* The model's performance is primarily evaluated through the training/validation workflow and application-level testing rather than a large production dataset.

---

## Future Work

Possible next steps include:

* Build a larger Persian customer-support dataset.
* Add a dedicated evaluation benchmark for intent classification and structured tool selection.
* Improve tool coverage and align the training labels with the implemented application tools.
* Add persistent conversation memory.
* Replace mock business systems with real APIs.
* Add observability and structured logging.
* Benchmark the fine-tuned model against the original Qwen model.
* Compare different LoRA configurations.
* Evaluate quantization quality and inference latency.
* Deploy the model behind a production API.

---

## Tech Stack

```text
Python
PyTorch
Transformers
TRL
PEFT
BitsAndBytes
Hugging Face Datasets
Qwen2.5-3B-Instruct
QLoRA
LoRA
llama.cpp
GGUF
Ollama
```

---

## What This Project Demonstrates

This project demonstrates practical experience with the lifecycle of an open-source LLM:

* Preparing a domain-specific instruction dataset
* Formatting conversational training data
* 4-bit model loading
* QLoRA fine-tuning
* LoRA configuration and target-module selection
* Supervised Fine-Tuning with TRL
* Training with limited GPU resources
* Saving and merging LoRA adapters
* Converting a Hugging Face model to GGUF Q8_0
* Local LLM deployment with Ollama
* Structured LLM outputs
* Tool routing
* Application-level guardrails
* Human confirmation for sensitive operations
* Integrating a fine-tuned LLM into an agentic application

The central idea is simple:

> **Fine-tuning the model is only one part of the system. The real engineering challenge is connecting that model to a controlled application that can safely turn model decisions into actions.**
