"""Preferential-attachment mobility-edge experiments."""

from .baseline import (
    MODEL_BA,
    MODEL_BA_REWIRED,
    MODEL_AFFINE_PA,
    MODEL_AFFINE_PA_REWIRED,
    MODEL_NONLINEAR_PA,
    MODEL_NONLINEAR_PA_REWIRED,
    attachment_offset_for_degree_ccdf_alpha,
    attachment_offset_for_ccdf_tau,
    generate_nonlinear_preferential_attachment_graph,
    generate_preferential_attachment_graph,
)

__all__ = [
    "MODEL_BA",
    "MODEL_BA_REWIRED",
    "MODEL_AFFINE_PA",
    "MODEL_AFFINE_PA_REWIRED",
    "MODEL_NONLINEAR_PA",
    "MODEL_NONLINEAR_PA_REWIRED",
    "attachment_offset_for_degree_ccdf_alpha",
    "attachment_offset_for_ccdf_tau",
    "generate_nonlinear_preferential_attachment_graph",
    "generate_preferential_attachment_graph",
]
