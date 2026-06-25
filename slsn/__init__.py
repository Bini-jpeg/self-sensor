"""Self-Localizing Sensor Network (SLSN).

Drop sensors in a GPS-denied space; recover their relative geometry
from inter-node signal timing alone.
"""
from .network import SensorNetwork, Node
from .graph import shortest_path_distances, connectivity_matrix, is_connected
from .localization import Localizer

__all__ = [
    "SensorNetwork", "Node",
    "Localizer",
    "shortest_path_distances", "connectivity_matrix", "is_connected",
]
__version__ = "0.1.0"
