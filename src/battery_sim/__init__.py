"""Battery thermal-runaway reduced-order reference solver."""

from .io import load_scenario
from .solver import SimulationResult, Simulator

__all__ = ["SimulationResult", "Simulator", "load_scenario"]
__version__ = "0.1.0"

