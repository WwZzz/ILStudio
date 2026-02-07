"""
Deployment Module

This module provides deployment utilities for IL-Studio policies.

Note: The subpackage `deploy.remote` has been renamed to `deploy.comm`.
      Imports here are kept for backward compatibility.
"""

# Import from the new location (deploy.comm)
from .comm import (
    PolicyServer,
    PolicyClient,
    FastAPIPolicyServer,
    FastAPIPolicyClient,
    parse_server_address,
    is_server_address,
    is_http_address,
    create_server,
    create_client,
    BaseServer,
    BaseClient,
)

__all__ = [
    # TCP
    "PolicyServer",
    "PolicyClient",
    # FastAPI
    "FastAPIPolicyServer",
    "FastAPIPolicyClient",
    # Utilities
    "parse_server_address",
    "is_server_address",
    "is_http_address",
    # Factories
    "create_server",
    "create_client",
    # Base classes
    "BaseServer",
    "BaseClient",
]
