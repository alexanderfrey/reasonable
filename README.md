# Project Existential Recurrence: Identity-Conditioned Attention

**A conceptual framework for Small Language Models (SLMs) that evolve a persistent "Self" through recursive introspection and Attention-based identity injection.**

> *"The model does not just learn the text; it asks what kind of being would write this text, and creates an Attention Bias to view the world through that lens."*

## 📋 Abstract

Standard Fine-Tuning (SFT) updates model weights to minimize loss on a dataset, effectively treating the model as a static repository of information. **Existential Recurrence** treats the training process as a feedback loop of identity formation.

In this framework, the model creates a feedback loop between:
1.  **Introspection:** Asking itself "Who am I? What do I want to be?"
2.  **State Persistence:** Storing that answer in a dedicated "Identity Block" (MLP).
3.  **Attention Injection:** Projecting that identity directly into the **Self-Attention Mechanism** of the main model to condition how it processes the target corpus (e.g., the works of Peter Singer).

## ⚙️ The Architecture

We modify a standard Transformer-based SLM by partitioning it into two interacting streams:

1.  **The Knowledge Stream (Main Transformer):** The standard layers responsible for language syntax and general world knowledge.
2.  **The Identity Block (The Controller):** A separate, persistent MLP that maintains the model's "Self" state.

### The "Identity-Conditioned Attention" Mechanism

Instead of simply adding the identity to the residual stream, we inject it specifically into the **Self-Attention** heads. The "Self" acts as a lens that distorts what the model pays attention to.

*   **Standard Attention:** $Attention(Q, K, V)$
*   **Existential Attention:** The Identity Block projects a bias vector into the Query ($Q$) and Key ($K$) transformations.
    *   *Effect:* If the Identity Block holds a "Utilitarian" state, the attention mechanism is biologically biased to attend to words like "Suffering" or "Consequence" more strongly than "Profit" or "Aesthetics."

### The "Cycle of Becoming" Loop

Training is circular. Each epoch consists of the following steps:

```mermaid
graph TD
    subgraph "Phase 1: Introspection"
        A[Current State] -->|Prompt: 'Who am I?'| B[Generate Self-Description]
        B --> C[Vectorize Text]
        C -->|Update| D[Identity Block MLP]
    end

    subgraph "Phase 2: Injection"
        D -->|Generate| E[Soul Vector]
        E -->|Project into Q/K/V| F[Main Transformer Attention Heads]
    end

    subgraph "Phase 3: Acquisition"
        F -->|Read: Peter Singer Corpus| G[Predict Next Token]
        G -->|Calc Loss| H[Backpropagate]
        H -->|Update Weights| D
        H -->|Update Weights| F
    end
    
    H -.-> A