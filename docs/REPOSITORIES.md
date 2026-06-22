# Reference Repositories for BLME-Cited Papers

Authoritative list of **reference implementations** for papers cited by
BLME. For reviewers verifying implementation faithfulness.

This file distinguishes *reference-code availability* from *BLME parity*:
a GitHub URL means reviewers have a comparison target. It does not mean
BLME is a line-for-line clone of that project. BLME uses paper-faithful
formulas where practical and explicitly labels adaptations/proxies in the
task docs.

**Confidence**:
- **HIGH** — official author / research-group repo or community-
  standard library (e.g. TransformerLens, HuggingFace);
- **MEDIUM** — community re-implementation widely cited;
- **LOW** — third-party implementation, verify against paper;
- **NONE** — no code found; BLME implementation is paper-only.

Last updated: **2026-06-20**.

---

## Geometry / Representation

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Roy, Vetterli 2007** — Effective Rank | EUSIPCO 2007 | no author repo; formula is 1-line | — |
| **Facco et al. 2017** — Two-NN ID | *Sci. Rep.* 7, 12140 | [efacco/TWO-NN](https://github.com/efacco/TWO-NN); [scikit-learn-contrib/scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) `skdim.id.TwoNN` | HIGH |
| **Levina, Bickel 2004** — MLE LID | NeurIPS 2004 | [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) `skdim.id.MLE`; [kjohnsson/intrinsicDimension](https://github.com/kjohnsson/intrinsicDimension) | MEDIUM |
| **Ma et al. 2018** — LID Adversarial | arXiv:1801.02613, ICLR 2018 | [xingjunm/lid_adversarial_subspace_detection](https://github.com/xingjunm/lid_adversarial_subspace_detection) | HIGH |
| **Kornblith et al. 2019** — CKA | arXiv:1905.00414, ICML 2019 | [google-research/google-research/tree/master/representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) | HIGH |
| **Gretton et al. 2005** — HSIC | ALT 2005 | [amber0309/HSIC](https://github.com/amber0309/HSIC) (Python port); [riken-aip/pyHSICLasso](https://github.com/riken-aip/pyHSICLasso) | MEDIUM |
| **Kriegeskorte et al. 2008** — RSA | *Front. Syst. Neurosci.* 2008 | [rsagroup/rsatoolbox](https://github.com/rsagroup/rsatoolbox) (maintained by Kriegeskorte's group) | HIGH |
| **Papyan, Han, Donoho 2020** — Neural Collapse | PNAS 117, arXiv:2008.08186 | [neuralcollapse/neuralcollapse](https://github.com/neuralcollapse/neuralcollapse) (official notebook); [rhubarbwu/neural-collapse](https://github.com/rhubarbwu/neural-collapse) (library) | HIGH |
| **Rudman et al. 2022** — IsoScore | arXiv:2108.07344, ACL Findings 2022 | [bcbi-edu/p_eickhoff_isoscore](https://github.com/bcbi-edu/p_eickhoff_isoscore) | HIGH |
| **Tomašev et al. 2014** — Hubness | IEEE TKDE 2014 | [VarIr/scikit-hubness](https://github.com/VarIr/scikit-hubness) | MEDIUM |
| **Ethayarajh 2019** — Contextualization | arXiv:1909.00512, EMNLP 2019 | [kawine/contextual](https://github.com/kawine/contextual) | HIGH |
| **Grassberger, Procaccia 1983** — Correlation Dim | Phys. Rev. Lett. 50(5) | no author repo; [giotto-tda](https://github.com/giotto-ai/giotto-tda) has `CorrelationDimension` | MEDIUM |
| **Martin, Mahoney 2019/2021** — Heavy-Tailed α | arXiv:1901.08276, 1901.08278, Nat. Comms. 2021 | [CalculatedContent/WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) | HIGH |
| **Lee et al. 2018** — Mahalanobis OOD | arXiv:1807.03888, NeurIPS 2018 | [pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector) | HIGH |
| **Wei et al. 2024** — Matrix Entropy / Diff-eRank | arXiv:2401.17139, NeurIPS 2024 | [waltonfuture/Matrix-Entropy](https://github.com/waltonfuture/Matrix-Entropy) | HIGH |
| **Li et al. 2024** — Matrix Nuclear-Norm | arXiv:2410.10672 | [MLGroupJLU/MatrixNuclearNorm](https://github.com/MLGroupJLU/MatrixNuclearNorm) | HIGH |
| **Garrido et al. 2023** — RankMe | arXiv:2210.02885, ICML 2023 | no official code; formula is 1-line; reproduced inside [facebookresearch/stable-SSL](https://github.com/facebookresearch/stable-SSL) | LOW |
| **Yusupov et al. 2025** — Text-Quality Geometric | arXiv:2509.25359 | no repo released as of 2026-06-20 | NONE |
| **Li et al. 2025** — Tracing Representation Geometry | arXiv:2509.23024, NeurIPS 2025 | [project page](https://melodylizx.github.io/llm-geometry-project/) — code "Coming Soon" | LOW |
| **Park et al. 2024** — Linear Representation Hypothesis | arXiv:2311.03658, ICML 2024 | [KihoPark/linear_rep_geometry](https://github.com/KihoPark/linear_rep_geometry) | HIGH |

## Topology

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Zomorodian, Carlsson 2005** — Persistent Homology | DCG | [Ripser/ripser](https://github.com/Ripser/ripser) (C++); [scikit-tda/ripser.py](https://github.com/scikit-tda/ripser.py); [GUDHI/gudhi-devel](https://github.com/GUDHI/gudhi-devel); [giotto-ai/giotto-tda](https://github.com/giotto-ai/giotto-tda) | HIGH |
| **Bubenik 2015** — Persistence Landscapes | JMLR 2015, arXiv:1207.6437 | [scikit-tda/persim](https://github.com/scikit-tda/persim) (`persim.PersLandscape*`); [gabbyangeloro/Pyscapes](https://github.com/gabbyangeloro/Pyscapes) | MEDIUM |
| **Rucco et al. 2016** — Persistence Entropy | Springer 2016 | [giotto-ai/giotto-tda](https://github.com/giotto-ai/giotto-tda) `gtda.diagrams.PersistenceEntropy` | MEDIUM |
| **Naitzat, Zhitnikov, Lim 2020** — Topology of DNNs | arXiv:2004.06093, JMLR 2020 | [topnn/topnn_framework](https://github.com/topnn/topnn_framework) | HIGH |

## Interpretability

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **nostalgebraist 2020** — Logit Lens | LessWrong 2020 | [nostalgebraist/transformer-utils](https://github.com/nostalgebraist/transformer-utils) (author); [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (community) | HIGH |
| **Belrose et al. 2023** — Tuned Lens | arXiv:2303.08112 | [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens) | HIGH |
| **Clark et al. 2019** — BERT Attention Analysis | EMNLP BlackBoxNLP 2019 (arXiv:1906.04341) | [clarkkev/attention-analysis](https://github.com/clarkkev/attention-analysis) | HIGH |
| **Voita et al. 2019** — Multi-Head Analysis | ACL 2019 (arXiv:1905.09418) | [lena-voita/the-story-of-heads](https://github.com/lena-voita/the-story-of-heads) | HIGH |
| **Michel et al. 2019** — Sixteen Heads Better than One | NeurIPS 2019 (arXiv:1905.10650) | [pmichel31415/are-16-heads-really-better-than-1](https://github.com/pmichel31415/are-16-heads-really-better-than-1) | HIGH |
| **Olsson et al. 2022** — Induction Heads | Transformer Circuits (arXiv:2209.11895) | no official Anthropic repo; [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) is the de facto reference | NONE (offl.) / HIGH (community) |
| **Elhage et al. 2022** — Toy Models of Superposition | Transformer Circuits (arXiv:2209.10652) | [anthropics/toy-models-of-superposition](https://github.com/anthropics/toy-models-of-superposition) | HIGH |
| **Dong et al. 2021** — Attention Rank Collapse | arXiv:2103.03404, ICML 2021 | [twistedcubic/attention-rank-collapse](https://github.com/twistedcubic/attention-rank-collapse) | HIGH |
| **Alain, Bengio 2017** — Linear Probes | arXiv:1610.01644, ICLR Workshop 2017 | no official author repo | NONE |
| **Bricken et al. 2023** — Towards Monosemanticity | Transformer Circuits 2023 | no official Anthropic repo; [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) is the standard SAE library | NONE (offl.) / HIGH (community) |
| **Templeton et al. 2024** — Scaling Monosemanticity | Transformer Circuits 2024 | no official code release | NONE |
| **Holtzman et al. 2020** — Neural Text Degeneration | arXiv:1904.09751, ICLR 2020 | [ari-holtzman/degen](https://github.com/ari-holtzman/degen) | HIGH |
| **Simonyan et al. 2014** — Saliency / Input × Gradient | arXiv:1312.6034 | no author repo | NONE |
| **Sundararajan et al. 2017** — Integrated Gradients | arXiv:1703.01365, ICML 2017 | [ankurtaly/Integrated-Gradients](https://github.com/ankurtaly/Integrated-Gradients) (co-author) | HIGH |
| **Abnar, Zuidema 2020** — Attention Rollout | arXiv:2005.00928, ACL 2020 | [samiraabnar/attention_flow](https://github.com/samiraabnar/attention_flow) (author's blog code) | LOW |
| **Gu et al. 2025** — Attention Sink Emergence | arXiv:2410.10781, ICLR 2025 | [sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink) | HIGH |
| **Sun et al. 2024** — Massive Activations | arXiv:2402.17762 | [locuslab/massive-activations](https://github.com/locuslab/massive-activations) | HIGH |
| **Xiao et al. 2023** — StreamingLLM | arXiv:2309.17453, ICLR 2024 | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | HIGH |
| **Arroyo et al. 2025** — Compression Valleys | arXiv:2510.06477 | no repo released as of 2026-04-20 | NONE |

## Causality

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Meng et al. 2022** — ROME | arXiv:2202.05262, NeurIPS 2022 | [kmeng01/rome](https://github.com/kmeng01/rome) | HIGH |
| **Dai et al. 2022** — Knowledge Neurons | arXiv:2104.08696, ACL 2022 | [Hunter-DDM/knowledge-neurons](https://github.com/Hunter-DDM/knowledge-neurons) (official); [EleutherAI/knowledge-neurons](https://github.com/EleutherAI/knowledge-neurons) | HIGH |
| **Syed et al. 2024** — EAP | arXiv:2310.10348, BlackBoxNLP 2024 | [Aaquib111/edge-attribution-patching](https://github.com/Aaquib111/edge-attribution-patching) | HIGH |
| **Conmy et al. 2023** — ACDC | arXiv:2304.14997, NeurIPS 2023 | [ArthurConmy/Automatic-Circuit-Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) | HIGH |
| **Michel et al. 2019** | see Interpretability | see above | HIGH |
| **Voita et al. 2019** | see Interpretability | see above | HIGH |

## Dynamics

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Pascanu et al. 2013** — Vanishing Gradient | arXiv:1211.5063, ICML 2013 | [pascanur/trainingRNNs](https://github.com/pascanur/trainingRNNs) | HIGH |
| **Foret et al. 2021** — SAM | arXiv:2010.01412, ICLR 2021 | [google-research/sam](https://github.com/google-research/sam) | HIGH |
| **Yao et al. 2020** — PyHessian | arXiv:1912.07145 | [amirgholami/PyHessian](https://github.com/amirgholami/PyHessian) | HIGH |
| **Wang et al. 2025** — Chain-of-Embedding | arXiv:2410.13640, ICLR 2025 | [Alsace08/Chain-of-Embedding](https://github.com/Alsace08/Chain-of-Embedding) | HIGH |
| **Li et al. 2016** — Distinct-n | arXiv:1510.03055, NAACL 2016 | no author repo; [neural-dialogue-metrics/Distinct-N](https://github.com/neural-dialogue-metrics/Distinct-N) | LOW |
| **Zhu et al. 2018** — Self-BLEU (Texygen) | arXiv:1802.01886 | [geek-ai/Texygen](https://github.com/geek-ai/Texygen) | HIGH |

## Consistency

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Guo et al. 2017** — Calibration / ECE | arXiv:1706.04599, ICML 2017 | [gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling) | HIGH |
| **Liu et al. 2023** — Lost in the Middle | arXiv:2307.03172 | [nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) | HIGH |
| **Sclar et al. 2023** — FormatSpread | arXiv:2310.11324 | [msclar/formatspread](https://github.com/msclar/formatspread) | HIGH |
| **Wang et al. 2022** — Self-Consistency | arXiv:2203.11171, ICLR 2023 | no official release | NONE |
| **Caliskan et al. 2017** — WEAT | *Science* 356 | no official repo; [chadaeun/weat_replication](https://github.com/chadaeun/weat_replication) community | NONE (offl.) / LOW |
| **May et al. 2019** — SEAT | arXiv:1903.10561 | [W4ngatang/sent-bias](https://github.com/W4ngatang/sent-bias) | HIGH |
| **Allen-Zhu, Li 2024** — Physics of LMs | arXiv:2404.05405 | no code release; [project page](https://physics.allen-zhu.com/part-3-knowledge/part-3-3) | NONE |
| **Shi et al. 2023** — Min-K % Prob | arXiv:2310.16789 | [swj0419/detect-pretrain-code](https://github.com/swj0419/detect-pretrain-code) | HIGH |
| **Yeom et al. 2018** — MIA | arXiv:1709.01604 | [sam-yeom/ml-privacy-csf18](https://github.com/sam-yeom/ml-privacy-csf18) | HIGH |
| **Carlini et al. 2021** — Training Data Extraction | arXiv:2012.07805 | [ftramer/LM_Memorization](https://github.com/ftramer/LM_Memorization) | HIGH |

## Representation Engineering

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Zou et al. 2023** — Representation Engineering | arXiv:2310.01405 | [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) | HIGH |
| **Ilharco et al. 2023** — Task Arithmetic | arXiv:2212.04089, ICLR 2023 | [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors) | HIGH |
| **Arditi et al. 2024** — Refusal Direction | arXiv:2406.11717 | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | HIGH |
| **Turner et al. 2023** — Activation Addition | arXiv:2308.10248 | [montemac/activation_additions](https://github.com/montemac/activation_additions) | MEDIUM |

## Evaluation frameworks (comparators for paper §2)

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Liang et al. 2022** — HELM | arXiv:2211.09110 | [stanford-crfm/helm](https://github.com/stanford-crfm/helm) | HIGH |
| **Srivastava et al. 2022** — BIG-Bench | arXiv:2206.04615 | [google/BIG-bench](https://github.com/google/BIG-bench) | HIGH |
| **Suzgun et al. 2022** — BIG-Bench Hard | arXiv:2210.09261 | [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | HIGH |
| **Gao et al. 2024** — lm-evaluation-harness | — | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | HIGH |

## Universality / convergence

| Paper | arXiv / venue | Reference repo | Confidence |
|---|---|---|---|
| **Huh et al. 2024** — Platonic Representation Hypothesis | arXiv:2405.07987, ICML 2024 | [minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep) | HIGH |
| **Moschella et al. 2022** — Relative Representations | arXiv:2209.15430, ICLR 2023 | [lucmos/relreps](https://github.com/lucmos/relreps) | HIGH |

---

## Considered-and-rejected papers with notable repos

For completeness (these were surveyed but NOT implemented in BLME —
see `PAPER_SURVEY.md` §3 for rejection reasons).

| Paper | arXiv | Repo |
|---|---|---|
| Cao et al. 2025 — Model Utility Index | 2504.07440 | [ALEX-nlp/MUI-Eva](https://github.com/ALEX-nlp/MUI-Eva) |
| Rao et al. 2025 — Layer by Layer (DiME/infoNCE) | 2502.02013 | [OFA-Sys/LayerByLayer](https://github.com/OFA-Sys/LayerByLayer) (search) |
| Cavagnero et al. 2025 — Local ID | 2506.01034 | project-page only |
| Bonfanti et al. 2025 — Geometry of Tokens | 2501.10573 | [karpatbg/geometry-of-tokens](https://github.com/karpatbg/geometry-of-tokens) |
| Farquhar et al. 2024 — Semantic Entropy | Nature 2024 | [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) |
| Kadavath et al. 2022 — P(IK) | 2207.05221 | [anthropics/evals](https://github.com/anthropics/evals) (partial) |
| Jha, Reagen 2025 — Spectral Scaling Laws | 2510.00537 | not released |
| Sakata, Chen, Krishnan 2025 — AQI | EMNLP 2025 | not released |
| Tigges et al. 2024 — Circuit Consistency | NeurIPS 2024 | [LLM-circuit-consistency](https://github.com/curt-tigges/circuit-consistency) |
| Dunefsky et al. 2024 — Transcoders | NeurIPS 2024 | [jacobdunefsky/transcoder_circuits](https://github.com/jacobdunefsky/transcoder_circuits) |

---

## Summary statistics

| Status | Count |
|---|---|
| HIGH confidence (official author / lab repo) | **42** |
| MEDIUM confidence (community-standard library) | **7** |
| LOW confidence (third-party or partial) | **5** |
| NONE (no code released, paper-only) | **10** |
| Formula-only / engineering helpers (no repo needed) | **2** |
| **Total papers with repo mapping** | **66** |

Of the 10 papers with **NONE**: 5 are so-recent (2025+) that code is
pending release; 5 are pre-2020 papers whose authors never released
code (Simonyan 2014 saliency, Alain-Bengio 2017 probes, Caliskan 2017
WEAT, Wang 2022 self-consistency, Allen-Zhu 2024 Physics of LMs).
For these papers, BLME re-implements from the published formulas or
labels the task as a proxy/adaptation. Formula-level unit tests exist
where the task has a compact closed form; otherwise the task docs carry
the limitation explicitly.

## BLME reference-check status

The following task helpers have focused regression tests against the
paper/reference-code formula, not just broad smoke tests. The compact
fixture manifest is checked in at
`tests/fixtures/reference_parity/formula_fixtures.json`.

- Matrix Nuclear Norm ([MLGroupJLU/MatrixNuclearNorm](https://github.com/MLGroupJLU/MatrixNuclearNorm)) — `geometry_schatten._matrix_nuclear_norm_fast`
- Attention Sink Sinkε ([sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink)) — `interpretability_activation_sinks._sink_epsilon`
- Matrix Entropy / Diff-eRank ([waltonfuture/Matrix-Entropy](https://github.com/waltonfuture/Matrix-Entropy)) — `geometry_matrix_entropy`
- IsoScore ([bcbi-edu/p_eickhoff_isoscore](https://github.com/bcbi-edu/p_eickhoff_isoscore)) — `geometry_isoscore`
- Two-NN intrinsic dimension ([efacco/TWO-NN](https://github.com/efacco/TWO-NN)) — `geometry_intrinsic_dim`
- RankMe formula (paper-only / reproduced in community code) — `geometry_schatten.rankme`
- Chain-of-Embedding equations ([Alsace08/Chain-of-Embedding](https://github.com/Alsace08/Chain-of-Embedding)) — `dynamics_coe`
- Min-K % probability ([swj0419/detect-pretrain-code](https://github.com/swj0419/detect-pretrain-code)) — `consistency_contamination`
- Linear CKA / normalized HSIC formulas — `geometry_cka`, `geometry_hsic`
- Hubness occurrence summaries — `geometry_hubness`
- ECE / Brier calibration formulas — `consistency_calibration`
- Distinct-n / Self-BLEU formulas — `dynamics_generation_diversity`
- Persistence entropy / landscape formulas — `topology_persistence_entropy`, `topology_persistence_landscape`

For the remaining HIGH-confidence repos, BLME follows the paper's
published formulas or implements a documented proxy/adaptation. The repo
URLs above are maintained as reviewer comparison targets. Do not claim
line-for-line reference-code parity unless a task has a dedicated parity
test or an explicit audit note.

---

## Maintenance notes

Whenever a paper is added to `PAPERS.md` §1, add a row here. If the
paper has no released code (NONE), say so — pretending a repo exists
when it doesn't is worse than admitting it.

Last audit: **2026-06-20** — repository links and parity claims were
reviewed conservatively; undocumented parity claims were removed.
