"""
Satellite Parameters Database

Simple database to get satellite parameters by NORAD ID.

Author: Divyanshu Panday
Date: October 2025
"""

# Satellite parameter database (key = NORAD ID)
SATELLITE_PARAMETERS = {
    59386: { 'name': 'STARLINK-31688',
        'mass_kg': 730.0,
        'surface_area_m2': 35.2,
        'orbit_alt_km': 483.0,
        'cr': 1.3
    },

    60309: { 'name': 'STARLINK-11213',

        'mass_kg': 960.0,
        'surface_area_m2': 35.5,
        'orbit_alt_km': 365.0,
        'cr': 1.2
    },

    43476: { 'name': 'GRACE-FO-1',
        'mass_kg': 600.0,
        'surface_area_m2': 3.5,
        'orbit_alt_km': 490.0,
        'cr': 1.2
    },

    41240: { 'name': 'JASON-3',
        'mass_kg': 553.0,
        'surface_area_m2': 6.5,
        'orbit_alt_km': 1336.0,
        'cr': 1.3
    },
    41790: { 'name': 'SCATSAT 1',
        'mass_kg': 370.0,
        'surface_area_m2': 4,
        'orbit_alt_km': 728.0,
        'cr': 1.2
    },
    44804: { 'name': 'CARTOSAT-3',
        'mass_kg': 1625.0,
        'surface_area_m2': 15.0,
        'orbit_alt_km': 509.0,
        'cr': 1.3
    },
    44857: { 'name': 'RISAT-2BR1',
        'mass_kg': 628.0,
        'surface_area_m2': 11.3,
        'orbit_alt_km': 557.0,
        'cr': 1.3},
    37839: {'name': 'JUGNU',
        'mass_kg': 3,
        'surface_area_m2': 0.04,
        'orbit_alt_km': 850.0,
        'cr': 1.2
},
    40069: {'name': 'SCATSAT 1',
        'mass_kg': 370,
        'surface_area_m2': 4.0,
        'orbit_alt_km': 728.0,
        'cr': 1.2
},
    39634: {'name': 'Sentinel-1A',
        'mass_kg': 2157,
        'surface_area_m2': 27.0,
        'orbit_alt_km': 693.0,
        'cr': 1.2}
}


def get_params(norad_id: int) -> dict:
    """Get satellite parameters by NORAD ID."""
    if norad_id not in SATELLITE_PARAMETERS:
        raise ValueError(f"NORAD ID {norad_id} not found!")
    return SATELLITE_PARAMETERS[norad_id]



