"""NEURONG learning framework (WIP).

This package provides a small, stable API layer to present NEURONG as a
"simulation + learning backend" while keeping NEURON as the modeling frontend.

Design goals:
- Make tasks look like calling an API (compile -> train), not running a script.
- Encode export/row-index timing contracts in one place.
- Keep Python customizability mainly at the loss/objective layer.
"""

from .bundle import ExportBundle
from .frontend_neuron import CompileConfig, compile_from_neuron
from .runtime import Runtime, RuntimeConfig
from .task import LearningTask, TaskSpec, Trainable, Probe
from .trainer import Trainer, TrainerConfig
from .optim import AdamState
from .objective import Objective, ObjectiveSpec
from .method import GradientMethod, MethodSpec
from .backend import LearningBackend
from .ops import LearningOps
from .learner import OnlineVJPLearner, VJPForwardResult, VJPLossResult, VJPSegment, VJPStepResult
from .segment import MonitorHandleGroup, SegmentResult, SegmentRunner, SegmentSpec, SegmentWindow
from .signals import (
    CaptureSpec,
    CapturedPack,
    MappedSignal,
    capture_signals_cached,
    empty_signal,
    ReplayGrads,
    replay_grads_from_cached_signals,
    replay_dw_from_cached_signals,
)
from .cli import main as cli_main

__all__ = [
    "CompileConfig",
    "ExportBundle",
    "Runtime",
    "RuntimeConfig",
    "LearningTask",
    "TaskSpec",
    "Trainable",
    "Probe",
    "Trainer",
    "TrainerConfig",
    "AdamState",
    "Objective",
    "ObjectiveSpec",
    "GradientMethod",
    "MethodSpec",
    "LearningBackend",
    "LearningOps",
    "OnlineVJPLearner",
    "VJPSegment",
    "VJPForwardResult",
    "VJPLossResult",
    "VJPStepResult",
    "SegmentResult",
    "SegmentRunner",
    "SegmentSpec",
    "SegmentWindow",
    "MonitorHandleGroup",
    "MappedSignal",
    "CaptureSpec",
    "CapturedPack",
    "empty_signal",
    "capture_signals_cached",
    "ReplayGrads",
    "replay_grads_from_cached_signals",
    "replay_dw_from_cached_signals",
    "compile_from_neuron",
    "cli_main",
]
