"""Feature engineering shared by the training scripts and the live service.

Importing the same module in both places is deliberate: it is the only way to
guarantee the vector the model was trained on is the vector it is served.
"""

from app.features.build import (
    FEATURE_NAMES,
    INDUSTRIES,
    REGIONS,
    SOURCES,
    SEQ_FEATURE_NAMES,
    SEQ_LEN,
    STAGES,
    build_feature_frame,
    build_feature_vector,
    build_sequence,
    feature_schema,
)

__all__ = [
    "FEATURE_NAMES",
    "INDUSTRIES",
    "REGIONS",
    "SOURCES",
    "SEQ_FEATURE_NAMES",
    "SEQ_LEN",
    "STAGES",
    "build_feature_frame",
    "build_feature_vector",
    "build_sequence",
    "feature_schema",
]
