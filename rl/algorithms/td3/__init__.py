from .td3 import TD3Algorithm, TD3Config
from .. import register_algorithm

register_algorithm("td3", TD3Algorithm)

__all__ = ["TD3Algorithm", "TD3Config"]

