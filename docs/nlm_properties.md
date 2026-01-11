NLMs as Oscillators / Frequency Generators

  Each NLM independently processes its neuron's temporal history. With different learned weights, each NLM develops different temporal dynamics - essentially different "frequencies" or oscillation patterns:

  - Some neurons might respond quickly to recent changes (high frequency)
  - Some might integrate slowly over longer history (low frequency)
  - Some might oscillate, some might be stable
  - The heterogeneity of these dynamics is the source of richness

  Sync as Phase/Frequency Correlation

  The sync module measures which neurons are "in sync" - i.e., which neurons have correlated temporal patterns. When two neurons oscillating at similar frequencies/phases are measured, they show high correlation.

  NLM_1: ~~~∿∿∿~~~  (slow oscillation)
  NLM_2: ∿∿∿∿∿∿∿∿  (fast oscillation)
  NLM_3: ~~~∿∿∿~~~  (slow, in phase with NLM_1)

  Sync would capture: NLM_1 ↔ NLM_3 are synchronized, NLM_2 is different