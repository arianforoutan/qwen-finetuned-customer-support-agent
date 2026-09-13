# Autonomous Customer Support AI Agent

An autonomous, locally deployed customer support AI system powered by a fine-tuned **Qwen 2.5 (Q8 quantized)** model, integrated with business guardrails, confirmation loops, and RAG knowledge retrieval.

## Features
- **Intent & Entity Extraction:** Structured JSON parsing directly from the local LLM.
- **Operational Tools:** Database lookups for orders, logistics escalation, inventory checks, and complaint logging.
- **Safety Guardrails:** Human-in-the-loop confirmation before destructive actions (e.g., order cancellations).
- **RAG Policy Search:** Fallback knowledge search for returns, warranty, and shipping guidelines.
- **Local Inference:** Fully private execution via Ollama.

## Setup & Installation

1. **Clone repository:**
   ```bash
   git clone <REPO_URL>
   cd customer-support-ai