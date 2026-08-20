# Literature map

Search refreshed 2026-08-19 using primary papers and official model/runtime sources. Terminology
searched included truth vectors, contextual truth geometry, latent transformation operators,
equivariance, continuous reasoning, thought reuse, causal representation identifiability, global
sections, fibrations/base change, causal tracing, and activation patching. This is a due-diligence map,
not a novelty claim.

## Truth and context in residual representations

- Marks and Tegmark (2023), [The Geometry of Truth: Emergent Linear Structure in Large Language
  Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824). Establishes linear
  truth-related structure across datasets; motivates truth-vector baselines but does not test controlled
  context transport/base lifting.
- Adarsh, Maistro, and Lioma (2026), [How Context Shapes Truth: Geometric Transformations of
  Statement-level Truth Representations in LLMs](https://arxiv.org/abs/2601.06599). The nearest direct
  context/truth-geometry work; measures direction and magnitude under context. GCT adds an independent
  compositional oracle, fitted held-out operators, path defects, behavior, and identifiability controls.
- Gurnee and Tegmark (2023), [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207).
  Relevant precedent for held-out ridge decoding of continuous coordinates and the warning that
  decodability does not show causal use.

## Latent and reusable reasoning

- Hao et al. (2024), [Training Large Language Models to Reason in a Continuous Latent Space
  (Coconut)](https://arxiv.org/abs/2412.06769). Replaces textual chain-of-thought steps with recurrent
  hidden states; it trains a reasoning method rather than auditing pretrained context transport.
- Yang et al. (2024), [Buffer of Thoughts](https://arxiv.org/abs/2406.04271). Reuses high-level thought
  templates; relevant to the project's motivation but not latent transformation-law estimation.
- Ahmed et al. (2025), [Retrieval-of-Thought](https://arxiv.org/abs/2509.21743). Reuses reasoning steps
  through retrieval/graphs rather than testing representation equivariance.
- Wu et al. (2026), [Reasoning Cache](https://arxiv.org/abs/2602.03773). A long-horizon cache method;
  again, reuse is an engineered mechanism rather than evidence of naturally encoded transport laws.

## Equivariant and transformation-based representation learning

- Bronstein et al. (2021), [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and
  Gauges](https://arxiv.org/abs/2104.13478). Standard framework for group actions, invariance, and
  equivariance.
- Qi, Zhang, and Wang (2019), [Learning Generalized Transformation Equivariant Representations via
  Autoencoding Transformations](https://arxiv.org/abs/1906.08628). Learns generic transformation-aware
  representations; close terminology for GCT's empirical operator tests.
- Winter et al. (2022), [Unsupervised Learning of Group Invariant and Equivariant
  Representations](https://arxiv.org/abs/2202.07559). Separates invariant content and group action.
- Song et al. (2024), [Unsupervised Representation Learning from Sparse Transformation
  Analysis](https://arxiv.org/abs/2410.05564). Learns sparse latent flow fields and approximate
  equivariance. It is close to continuous-generator language outside LLM truth contexts.

## Causal/invariant representations and identifiability

- Arjovsky et al. (2019), [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893). Canonical
  invariant-prediction framework across environments.
- Ahuja et al. (2022), [Interventional Causal Representation Learning](https://arxiv.org/abs/2209.11924).
  Gives latent-factor identifiability results from interventions and highlights geometric signatures.
- Varici et al. (2024), [Linear Causal Representation Learning from Unknown Multi-node
  Interventions](https://arxiv.org/abs/2406.05937). Establishes identifiability conditions far stronger
  than GCT's observational residual probe. GCT therefore uses “recoverable coordinate,” not causal
  ontology discovery.

## Sheaves, contextuality, fibrations, and base change

- Abramsky and Brandenburger (2011), [The Sheaf-Theoretic Structure of Non-Locality and
  Contextuality](https://arxiv.org/abs/1102.0264). Characterizes contextuality through global-section
  obstruction in a defined measurement setting.
- Abramsky, Mansfield, and Barbosa (2011), [The Cohomology of Non-Locality and
  Contextuality](https://arxiv.org/abs/1111.3620). Defines a Cech obstruction for an abelian presheaf;
  nonvanishing is sufficient, not necessary. GCT does not implement this coefficient structure.
- Sterling, Angiuli, and Gratzer (2022), [A Cubical Language for Bishop
  Sets](https://arxiv.org/abs/2003.01491). Includes explicit fibred/model semantics and
  Beck-Chevalley preservation conditions; useful vocabulary discipline, not an empirical LLM method.

## Activation intervention and tracing

- Vig et al. (2020), [Causal Mediation Analysis for Interpreting Neural NLP](https://arxiv.org/abs/2004.12265).
  Early causal mediation/intervention framework for neural language models.
- Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262).
  Uses causal tracing and interventions to localize factual associations.
- Zhang and Nanda (2023), [Towards Best Practices of Activation Patching in Language
  Models](https://arxiv.org/abs/2309.16042). Shows patching conclusions depend on corruption and metric
  choices. GCT keeps patching outside confirmatory v0.
- Kramar et al. (2024), [AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to
  Components](https://arxiv.org/abs/2403.00745). Scalable attribution-patching approximation and
  false-negative analysis.

## Instrumentation and hardware

- [Qwen3-4B official model card](https://huggingface.co/Qwen/Qwen3-4B) and exact resolved config are
  recorded per run.
- [Hugging Face model output documentation](https://huggingface.co/docs/transformers/main_classes/output)
  defines embedding-plus-layer hidden-state tuples.
- [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/) introduced official Blackwell/CUDA 12.8
  wheels; [PyTorch 2.12](https://pytorch.org/blog/pytorch-2-12-release-blog/) recommends CUDA 13.0+
  wheels for Blackwell. The tested lock uses 2.12.0+cu130.
