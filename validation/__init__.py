"""
Validation Module

Compare Physics-based (Cannonball) SRP vs ML-predicted SRP.

Workflow:
1. Collect TLE data for specific satellite + date range
2. Run preprocessing pipeline (positions, shadow, features)
3. Calculate Physics-based SRP acceleration
4. Load ML model and predict SRP acceleration
5. Compare results

Author: Divyanshu Panday
Date: October 2025
"""

__version__ = '1.0.0'
