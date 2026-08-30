from __future__ import annotations

from flashml.errors import DependencyUnavailableError
from flashml.state import AppState


def get_moge():
    if AppState.moge is None:
        raise DependencyUnavailableError("reconstruct service is not enabled")
    return AppState.moge


def get_simpleclick():
    if AppState.simpleclick is None:
        raise DependencyUnavailableError("interactive-segment service is not enabled")
    return AppState.simpleclick


def get_oneformer():
    if AppState.oneformer is None:
        raise DependencyUnavailableError("segment service is not enabled")
    return AppState.oneformer


def get_rorem():
    if AppState.rorem is None:
        raise DependencyUnavailableError("remove service is not enabled")
    return AppState.rorem
