"""
Data Acquisition Module for SRP Project

This module contains utilities for downloading and processing satellite
TLE data from Space-Track.org.

Author: Divyanshu Panday
"""

from .spacetrack_client import SpaceTrackClient

__all__ = ['SpaceTrackClient']
__version__ = '1.0.0'
