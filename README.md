# Project Existential Recurrence: Recursive Identity-Conditioned Training

**A conceptual framework for Small Language Models (SLMs) that evolve a persistent "Self" through recursive introspection and targeted training.**

> *"The model does not just learn the text; it asks what kind of being would write this text, and attempts to become that being."*

## 📋 Abstract

Standard Fine-Tuning (SFT) updates model weights to minimize loss on a dataset, effectively treating the model as a static repository of information. **Existential Recurrence** treats the training process as a feedback loop of identity formation.

In this framework, the model creates a feedback loop between:
1.  **Introspection:** Asking itself "Who am I? What do I want to be?"
2.  **State Persistence:** Storing that answer in a dedicated "Identity Block" (MLP).
3.  **Acquisition:** Training on a target corpus (e.g., the works of Peter Singer) conditioned on that identity.

The hypothesis is that by explicitly modeling the "Observer" (the model's state) separate from the "Observation" (the data), the model will align its internal latent space more robustly with the philosophical stance of the target data, rather than just mimicking surface-level patterns.

## ⚙️ The Architecture

We modify a standard Transformer-based SLM by partitioning it into two streams:

1.  **The Knowledge Stream (Frozen/Slow-Moving):** Standard pretrained layers responsible for language syntax and general world knowledge.
2.  **The Identity Block (Fast-Moving/Plastic):** A dedicated Multi-Layer Perceptron (MLP) or Learnable Latent Vector sequence that persists the model's current "Self."

### The "Cycle of Becoming" Loop

Training is not linear; it is circular. Each epoch consists of the following steps:

```mermaid
graph TD
    A[Start: Pretrained SLM] --> B{Phase 1: Introspection}
    B -->|Prompt: 'Who am I?'| C[Generate Self-Description]
    C --> D[Vectorize & Inject into Identity Block]
    D --> E{Phase 2: Acquisition}
    E -->|Read: Target Corpus e.g., Peter Singer| F[Predict Next Token]
    F -->|Calc Loss| G[Backpropagate]
    G -->|Update Weights| H[Identity Block Shifted]
    H --> B