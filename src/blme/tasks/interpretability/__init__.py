from .logit_lens import LogitLensTask
from .attribution import ComponentAttributionTask
from .attention import AttentionEntropyTask
from .induction import InductionHeadTask
from .prediction_entropy import PredictionEntropyTask
from .probing import LinearProbingTask
from .sparsity import ActivationSparsityTask
from .sae_features import SAEFeatureDimensionalityTask
from .attention_graph import AttentionGraphTopologyTask
from .weight_activation_alignment import WeightActivationAlignmentTask
from .attention_polysemanticity import AttentionEffectiveRankTask
from .superposition import SuperpositionIndexTask
from .attention_rank import AttentionRankCollapseTask
from .head_roles import HeadRolesTask
from .activation_sinks import ActivationSinksTask
# Campaign-2 addition (2026-06, scipy-parity-verified)
from .activation_kurtosis import ActivationKurtosisTask
# Campaign-2 Wave-2/3 addition (2026-07, Abnar rollout parity-verified)
from .attention_rollout import AttentionRolloutTask
