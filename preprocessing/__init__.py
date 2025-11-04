"""
Preprocessing Module for SRP Project

This module contains utilities for preprocessing satellite TLE data and
calculating SRP-related features.

Modules:
--------
- satellite_constants: Satellite parameters database
- data_selector: Select and prepare DataFrame columns (class-based)
- position_calculator: Compute satellite and Sun positions (function-based)
- shadow_calculator: Calculate eclipse/shadow conditions (function-based)
- srp_features: Compute SRP features (class-based)
- srp_acceleration: Calculate Cannonball SRP acceleration (function-based)

Author: Divyanshu Panday
Date: October 2025
"""

# Import satellite constants
from .satellite_constants import SATELLITE_PARAMETERS, get_params

# Import classes
from .data_selector import DataSelector
from .srp_features import SRPFeatureCalculator

# Import functions
from .position_calculator import add_skyfield_position_columns
from .shadow_calculator import (
    angle_between,
    compute_shadow_factor,
    add_shadow_factor_column
)
from .srp_acceleration import add_cannonball_srp_acceleration

__all__ = [
    # Constants
    'SATELLITE_PARAMETERS',
    'get_params',
    # Classes
    'DataSelector',
    'SRPFeatureCalculator',
    # Functions
    'add_skyfield_position_columns',
    'angle_between',
    'compute_shadow_factor',
    'add_shadow_factor_column',
    'add_cannonball_srp_acceleration',
]

__version__ = '1.0.0'
