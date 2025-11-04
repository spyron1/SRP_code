# Data Acquisition Module

**Author:** Divyanshu Panday  
**Purpose:** Download TLE data from Space-Track.org for SRP analysis

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Edit the `.env` file in the project root:

```env
SPACETRACK_USERNAME=your_username_here
SPACETRACK_PASSWORD=your_password_here
```

> **Note:** Register for free at [Space-Track.org](https://www.space-track.org/auth/createAccount)

## Usage

### Basic Usage

```python
from data_acquisition import SpaceTrackClient

# Initialize client (loads credentials from .env)
client = SpaceTrackClient()

# Download satellite data (saves to raw_data/ folder)
df = client.save_to_excel(
    norad_cat_id=43476,        # GRACE-FO satellite
    start_epoch='2023-05-01',  # Start date (YYYY-MM-DD)
    end_epoch='2025-09-11'     # End date (YYYY-MM-DD)
)

print(f"Downloaded {len(df)} records")
```

### Advanced Usage

```python
# Custom filename
df = client.save_to_excel(
    norad_cat_id=59386,
    start_epoch='2024-01-01',
    end_epoch='2025-09-11',
    custom_filename='STARLINK-31688_processed'
)

# Force re-download even if file exists
df = client.save_to_excel(
    norad_cat_id=43476,
    start_epoch='2023-05-01',
    end_epoch='2025-09-11',
    force_download=True  # Re-download
)

# Multiple satellites
satellites = {
    'GRACE-FO-1': 43476,
    'GRACE-FO-2': 43477,
    'JASON-3': 41240
}

for name, norad_id in satellites.items():
    print(f"\nDownloading {name}...")
    df = client.save_to_excel(
        norad_cat_id=norad_id,
        start_epoch='2023-01-01',
        end_epoch='2025-10-14'
    )
```

### Running the Example

```bash
python data_acquisition/spacetrack_client.py
```

## Output

Files are saved to the `raw_data/` directory (inside data_acquisition folder):

- **Filename format:** `NORAD_{id}_processed.xlsx`
- **Contains:** All TLE fields from Space-Track.org
- **Columns include:** EPOCH, TLE_LINE1, TLE_LINE2, INCLINATION, ECCENTRICITY, etc.
- **Smart caching:** Won't re-download if file already exists (unless `force_download=True`)

## Features

✅ Dynamic NORAD_CAT_ID input  
✅ Flexible epoch date ranges  
✅ Auto-generated or custom filenames  
✅ **Smart caching** - Avoids duplicate downloads  
✅ **Force re-download** option when needed  
✅ Secure credential management via .env  
✅ All JSON fields preserved in Excel  
✅ Automatic directory creation  
✅ Session-based authentication  

## Troubleshooting

**File already exists message**
- This is normal! The module checks if you've already downloaded this data
- To re-download anyway, use `force_download=True`

**Error: Credentials not found**
- Ensure `.env` file exists in project root
- Check that `SPACETRACK_USERNAME` and `SPACETRACK_PASSWORD` are set

**Error: No records returned**
- Verify NORAD_CAT_ID is correct
- Check date range is valid
- Ensure satellite has data for that period

**Authentication failed**
- Verify credentials at [Space-Track.org](https://www.space-track.org)
- Check internet connection
