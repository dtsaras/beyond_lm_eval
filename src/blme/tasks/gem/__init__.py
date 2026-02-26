"""GEM-specific diagnostic tasks.

These tasks validate the Geometric Embedding Mixture (GEM) hypothesis —
that hidden states in LLMs can be interpreted as weighted mixtures of
token embeddings. They are not general-purpose LLM property measurements,
but rather hypothesis-validating experiments for the GEM framework.
"""
from .alignment import AlignmentResidualTask, SubstitutionConsistencyTask
from .editing import MixtureEditingTask
from .trajectories import MixtureTrajectoriesTask
