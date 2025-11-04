"""
TLE Collector for Validation

Fetches TLE data for last 24 hours for satellites in satellite_constants.py.
Returns DataFrame (no Excel storage).
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_acquisition.spacetrack_client import SpaceTrackClient
from preprocessing.satellite_constants import SATELLITE_PARAMETERS


class ValidationTLECollector:
    """
    Collects TLE data for validation testing (last 24 hours)
    Only supports satellites defined in satellite_constants.py
    """
    
    def __init__(self, username, password):
        """
        Initialize with Space-Track credentials
        
        Parameters:
        -----------
        username : str
            Space-Track username
        password : str
            Space-Track password
        """
        self.client = SpaceTrackClient(username, password)
        self.available_norad_ids = list(SATELLITE_PARAMETERS.keys())
        
    def collect_tle_last_24hrs(self, norad_id):
        """
        Collect TLE data for last 24 hours
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID (must be in satellite_constants.py)
            
        Returns:
        --------
        DataFrame : Full TLE data (for preprocessing)
        """
        # Validate NORAD ID
        if norad_id not in self.available_norad_ids:
            raise ValueError(
                f"NORAD ID {norad_id} not found in satellite_constants.py!\n"
                f"Available IDs: {self.available_norad_ids}"
            )
        
        # Calculate last 24 hours (UTC) - using timezone-aware datetime
        from datetime import timezone
        end_date = datetime.now(timezone.utc).date() + timedelta(days=1)  # Include future buffer
        start_date = end_date - timedelta(days=2)  # Get 2 days to ensure we have recent data
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Collecting TLE for NORAD {norad_id} (latest data)")
        print(f"Date range: {start_date_str} to {end_date_str}")
        
        # Build custom query URL with DESCENDING order to get latest first
        query_url = (
            f"{self.client.BASE_URL}/basicspacedata/query/class/gp_history/"
            f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20DESC/"
            f"EPOCH/{start_date_str}--{end_date_str}/format/json"
        )
        
        # Login if needed
        if not self.client.session:
            self.client.login()
        
        # Download JSON data directly (no file saving)
        print(f"Downloading data from Space-Track...")
        response = self.client.session.get(query_url, timeout=120)
        response.raise_for_status()
        tle_records = response.json()
        
        if not tle_records:
            print(f"⚠️ No TLE data found for NORAD {norad_id} in last 24 hours")
            return None
        
        # Convert to DataFrame
        tle_data = pd.DataFrame(tle_records)
        
        print(f"✅ Fetched {len(tle_data)} TLE records")
        
        # Data already sorted by EPOCH DESC (latest first) from query
        # Display only LATEST TLE (TLE_LINE1 and TLE_LINE2)
        if 'TLE_LINE1' in tle_data.columns and 'TLE_LINE2' in tle_data.columns:
            print("\n📡 Latest TLE:")
            print("=" * 80)
            print(f"  {tle_data.iloc[0]['TLE_LINE1']}")  # First record is latest
            print(f"  {tle_data.iloc[0]['TLE_LINE2']}")
            if 'EPOCH' in tle_data.columns:
                print(f"  Epoch: {tle_data.iloc[0]['EPOCH']}")
            print()
        
        # Return full DataFrame (sorted latest first) for preprocessing
        return tle_data
    
    def collect_tle_last_2days(self, norad_id):
        """
        Collect TLE data for last 2 days
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID (must be in satellite_constants.py)
            
        Returns:
        --------
        DataFrame : Full TLE data (for preprocessing)
        """
        # Validate NORAD ID
        if norad_id not in self.available_norad_ids:
            raise ValueError(
                f"NORAD ID {norad_id} not found in satellite_constants.py!\n"
                f"Available IDs: {self.available_norad_ids}"
            )
        
        # Calculate last 2 days (UTC) - using timezone-aware datetime
        from datetime import timezone
        end_date = datetime.now(timezone.utc).date() + timedelta(days=1)  # Include future buffer
        start_date = end_date - timedelta(days=3)  # Get 3 days to ensure we have 2 full days
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Collecting TLE for NORAD {norad_id} (last 2 days)")
        print(f"Date range: {start_date_str} to {end_date_str}")
        
        # Build custom query URL with DESCENDING order to get latest first
        query_url = (
            f"{self.client.BASE_URL}/basicspacedata/query/class/gp_history/"
            f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20DESC/"
            f"EPOCH/{start_date_str}--{end_date_str}/format/json"
        )
        
        # Login if needed
        if not self.client.session:
            self.client.login()
        
        # Download JSON data directly (no file saving)
        print(f"Downloading data from Space-Track...")
        response = self.client.session.get(query_url, timeout=120)
        response.raise_for_status()
        tle_records = response.json()
        
        if not tle_records:
            print(f"⚠️ No TLE data found for NORAD {norad_id} in last 2 days")
            return None
        
        # Convert to DataFrame
        tle_data = pd.DataFrame(tle_records)
        
        print(f"✅ Fetched {len(tle_data)} TLE records")
        
        # Data already sorted by EPOCH DESC (latest first) from query
        # Display only LATEST TLE (TLE_LINE1 and TLE_LINE2)
        if 'TLE_LINE1' in tle_data.columns and 'TLE_LINE2' in tle_data.columns:
            print("\n📡 Latest TLE:")
            print("=" * 80)
            print(f"  {tle_data.iloc[0]['TLE_LINE1']}")  # First record is latest
            print(f"  {tle_data.iloc[0]['TLE_LINE2']}")
            if 'EPOCH' in tle_data.columns:
                print(f"  Epoch: {tle_data.iloc[0]['EPOCH']}")
            print()
        
        # Return full DataFrame (sorted latest first) for preprocessing
        return tle_data
    
    def collect_tle_last_7days(self, norad_id):
        """
        Collect TLE data for last 7 days
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID (must be in satellite_constants.py)
            
        Returns:
        --------
        DataFrame : Full TLE data (for preprocessing)
        """
        # Validate NORAD ID
        if norad_id not in self.available_norad_ids:
            raise ValueError(
                f"NORAD ID {norad_id} not found in satellite_constants.py!\n"
                f"Available IDs: {self.available_norad_ids}"
            )
        
        # Calculate last 7 days (UTC) - using timezone-aware datetime
        from datetime import timezone
        end_date = datetime.now(timezone.utc).date() + timedelta(days=1)  # Include future buffer
        start_date = end_date - timedelta(days=8)  # Get 8 days to ensure we have 7 full days
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Collecting TLE for NORAD {norad_id} (last 7 days)")
        print(f"Date range: {start_date_str} to {end_date_str}")
        
        # Build custom query URL with DESCENDING order to get latest first
        query_url = (
            f"{self.client.BASE_URL}/basicspacedata/query/class/gp_history/"
            f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20DESC/"
            f"EPOCH/{start_date_str}--{end_date_str}/format/json"
        )
        
        # Login if needed
        if not self.client.session:
            self.client.login()
        
        # Download JSON data directly (no file saving)
        print(f"Downloading data from Space-Track...")
        response = self.client.session.get(query_url, timeout=120)
        response.raise_for_status()
        tle_records = response.json()
        
        if not tle_records:
            print(f"⚠️ No TLE data found for NORAD {norad_id} in last 7 days")
            return None
        
        # Convert to DataFrame
        tle_data = pd.DataFrame(tle_records)
        
        print(f"✅ Fetched {len(tle_data)} TLE records")
        
        # Data already sorted by EPOCH DESC (latest first) from query
        # Display only LATEST TLE (TLE_LINE1 and TLE_LINE2)
        if 'TLE_LINE1' in tle_data.columns and 'TLE_LINE2' in tle_data.columns:
            print("\n📡 Latest TLE:")
            print("=" * 80)
            print(f"  {tle_data.iloc[0]['TLE_LINE1']}")  # First record is latest
            print(f"  {tle_data.iloc[0]['TLE_LINE2']}")
            if 'EPOCH' in tle_data.columns:
                print(f"  Epoch: {tle_data.iloc[0]['EPOCH']}")
            print()
        
        # Return full DataFrame (sorted latest first) for preprocessing
        return tle_data
    
    def collect_tle_custom_date(self, norad_id, start_date, end_date):
        """
        Collect TLE data for custom date range
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID (must be in satellite_constants.py)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        DataFrame : Full TLE data (for preprocessing)
        """
        # Validate NORAD ID
        if norad_id not in self.available_norad_ids:
            raise ValueError(
                f"NORAD ID {norad_id} not found in satellite_constants.py!\n"
                f"Available IDs: {self.available_norad_ids}"
            )
        
        print(f"Collecting TLE for NORAD {norad_id} (custom date range)")
        print(f"Date range: {start_date} to {end_date}")
        
        # Build custom query URL with DESCENDING order to get latest first
        query_url = (
            f"{self.client.BASE_URL}/basicspacedata/query/class/gp_history/"
            f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20DESC/"
            f"EPOCH/{start_date}--{end_date}/format/json"
        )
        
        # Login if needed
        if not self.client.session:
            self.client.login()
        
        # Download JSON data directly (no file saving)
        print(f"Downloading data from Space-Track...")
        response = self.client.session.get(query_url, timeout=120)
        response.raise_for_status()
        tle_records = response.json()
        
        if not tle_records:
            print(f"⚠️ No TLE data found for NORAD {norad_id} in date range")
            return None
        
        # Convert to DataFrame
        tle_data = pd.DataFrame(tle_records)
        
        print(f"✅ Fetched {len(tle_data)} TLE records")
        
        # Data already sorted by EPOCH DESC (latest first) from query
        # Display only LATEST TLE (TLE_LINE1 and TLE_LINE2)
        if 'TLE_LINE1' in tle_data.columns and 'TLE_LINE2' in tle_data.columns:
            print("\n📡 Latest TLE:")
            print("=" * 80)
            print(f"  {tle_data.iloc[0]['TLE_LINE1']}")  # First record is latest
            print(f"  {tle_data.iloc[0]['TLE_LINE2']}")
            if 'EPOCH' in tle_data.columns:
                print(f"  Epoch: {tle_data.iloc[0]['EPOCH']}")
            print()
        
        # Return full DataFrame (sorted latest first) for preprocessing
        return tle_data



def main():
    """
    Example usage - Collect TLE for last 24 hours
    """
    # Load credentials from .env
    load_dotenv()
    username = os.getenv("SPACETRACK_USERNAME")
    password = os.getenv("SPACETRACK_PASSWORD")
    
    # Initialize collector
    collector = ValidationTLECollector(username, password)
    
    # Show available satellites
    print("Available satellites in satellite_constants.py:")
    print(collector.available_norad_ids)
    print()
    
    # Example: Collect TLE for STARLINK-31688 (NORAD 59386)
    norad_id = 59386
    
    # Collect last 24 hours TLE data
    tle_df = collector.collect_tle_custom_date(norad_id,'2024-04-01','2024-04-03')
    # print("+_______")
    # print(tle_df)
    if tle_df is not None:
        print(f"\n✅ Returned DataFrame with {len(tle_df)} records")
        print(f"   Columns: {len(tle_df.columns)}")
        print(f"   Ready for preprocessing pipeline!")



if __name__ == '__main__':
    main()
