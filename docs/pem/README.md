# Predictive Experience Machine (PEM)

A neural architecture that **experiences** what it reads rather than just predicting next tokens.

## Core Principles

1. **Predictive Processing**: Constantly generates expectations about incoming features
2. **Surprise-Driven Learning**: Compares predictions to reality, experiences the delta
3. **Episodic Memory**: Remembers not just THAT it was surprised, but WHY
4. **Synchronization-Based Cognition**: Uses CTM-style sync as the cognitive substrate
5. **Separation of Concerns**: Transformers for perception, sync for cognition

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                   EXPERIENTIAL CORE                            │
│              (Sync-based meta-cognition)                       │
│                                                                │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│    │ PREDICT  │─────▶│ COMPARE  │─────▶│ INTEGRATE│           │
│    │ (sync→f̂) │      │ (f̂ vs f) │      │ (update) │           │
│    └────▲─────┘      └────┬─────┘      └────┬─────┘           │
│         │                 │                  │                 │
│         │            surprise ε              │                 │
│         │                 │                  ▼                 │
│         │                 │           ┌───────────┐            │
│         │                 └──────────▶│ SURPRISE  │            │
│         │                             │  MEMORY   │            │
│    ┌────┴─────┐                       │ (why?)    │            │
│    │   SYNC   │◀──────────────────────┴───────────┘            │
│    │  (NLMs)  │    surprise modulates sync dynamics            │
│    └────▲─────┘                                                │
│         │                                                      │
└─────────┼──────────────────────────────────────────────────────┘
          │ features f
┌─────────┴──────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                           │
│                (Frozen/slow-learning Transformer)              │
│                                                                │
│     tokens ──▶ [Transformer Encoder] ──▶ features f            │
└────────────────────────────────────────────────────────────────┘
```

## Components

- [Feature Extractor](./feature_extractor.md) - Transformer-based perception
- [Prediction Module](./prediction_module.md) - Sync → expected features
- [Surprise Module](./surprise_module.md) - Compare prediction to reality
- [Surprise Memory](./surprise_memory.md) - Episodic memory of surprises
- [Sync Core](./sync_core.md) - CTM-style meta-cognition
- [Experience Loop](./experience_loop.md) - The main processing loop
- [Training Objectives](./training.md) - How to train this
- [Open Questions](./open_questions.md) - Unresolved design decisions

## Extensions

- [Reflective Reading](./reflective_reading.md) - Multi-scale hypothesis generation for narrative understanding (reading novels, stories)

## What Makes This Different

| Standard LLM | Predictive Experience Machine |
|--------------|-------------------------------|
| Predicts tokens | Predicts *features* (richer) |
| No internal expectation | Constant expectation generation |
| Loss = wrong token | Loss = miscalibrated surprise |
| No memory of mistakes | Episodic memory of surprises |
| Uniform processing | Surprise modulates processing depth |
| Activations = representation | **Sync = representation** |
