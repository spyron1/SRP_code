"""
Space-Track Data Acquisition Module

This module provides functionality to download TLE data from Space-Track.org
and save it to Excel format with all available fields preserved.

Author: Divyanshu Panday
Date: October 2025
"""

import os
from typing import Optional
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv


class SpaceTrackClient:
    """Client for downloading and processing Space-Track.org data."""
    
    BASE_URL = "https://www.space-track.org"
    LOGIN_ENDPOINT = "/ajaxauth/login"
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Space-Track client with credentials.
        
        Parameters
        ----------
        username : str, optional
            Space-Track username (will load from .env if not provided)
        password : str, optional
            Space-Track password (will load from .env if not provided)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.username = username or os.getenv("SPACETRACK_USERNAME")
        self.password = password or os.getenv("SPACETRACK_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError(
                "Space-Track credentials not found. Please set SPACETRACK_USERNAME "
                "and SPACETRACK_PASSWORD in .env file or pass them to the constructor."
            )
        
        self.session = None
    
    def login(self) -> requests.Session:
        """
        Authenticate with Space-Track.org.
        
        Returns
        -------
        requests.Session
            Authenticated session object
            
        Raises
        ------
        requests.HTTPError
            If authentication fails
        """
        self.session = requests.Session()
        response = self.session.post(
            f"{self.BASE_URL}{self.LOGIN_ENDPOINT}",
            data={"identity": self.username, "password": self.password},
            timeout=30
        )
        response.raise_for_status()
        print("✓ Successfully authenticated with Space-Track.org")
        return self.session
    
    def build_query_url(self, 
                       norad_cat_id: int,
                       start_epoch: str,
                       end_epoch: str,
                       data_class: str = "gp_history") -> str:
        """
        Build Space-Track query URL for TLE data.
        
        Parameters
        ----------
        norad_cat_id : int
            NORAD Catalog ID of the satellite
        start_epoch : str
            Start date in format 'YYYY-MM-DD'
        end_epoch : str
            End date in format 'YYYY-MM-DD'
        data_class : str
            Space-Track data class (default: 'gp_history')
            
        Returns
        -------
        str
            Complete query URL
            
        Examples
        --------
        >>> client = SpaceTrackClient()
        >>> url = client.build_query_url(43476, '2023-05-01', '2025-09-11')
        """
        query_url = (
            f"{self.BASE_URL}/basicspacedata/query/class/{data_class}/"
            f"NORAD_CAT_ID/{norad_cat_id}/orderby/TLE_LINE1%20ASC/"
            f"EPOCH/{start_epoch}--{end_epoch}/format/json"
        )
        return query_url
    
    def download_json(self, url: str) -> list:
        """
        Download JSON data from Space-Track query URL.
        
        Parameters
        ----------
        url : str
            Full Space-Track query URL
            
        Returns
        -------
        list
            List of records (dictionaries)
            
        Raises
        ------
        ValueError
            If no records returned or invalid JSON format
        """
        if not self.session:
            self.login()
        
        print(f"Downloading data from Space-Track...")
        response = self.session.get(url, timeout=120)
        response.raise_for_status()
        
        records = response.json()
        if not isinstance(records, list) or len(records) == 0:
            raise ValueError("No records returned or unexpected JSON format.")
        
        print(f"✓ Downloaded {len(records)} records")
        return records
    
    def save_to_excel(self, 
                     norad_cat_id: int,
                     start_epoch: str,
                     end_epoch: str,
                     output_dir: str = "raw_data",
                     custom_filename: Optional[str] = None) -> pd.DataFrame:
        """
        Download Space-Track data and save to Excel with all columns preserved.
        If file exists, it will be overwritten.
        
        Parameters
        ----------
        norad_cat_id : int
            NORAD Catalog ID of the satellite
        start_epoch : str
            Start date in format 'YYYY-MM-DD'
        end_epoch : str
            End date in format 'YYYY-MM-DD'
        output_dir : str
            Output directory path (default: 'raw_data')
        custom_filename : str, optional
            Custom filename (without extension). If None, auto-generates 
            based on NORAD_CAT_ID
            
        Returns
        -------
        pd.DataFrame
            Downloaded data as DataFrame
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if custom_filename:
            filename = f"{custom_filename}.xlsx"
        else:
            filename = f"NORAD_{norad_cat_id}.xlsx"
        
        output_path = os.path.join(output_dir, filename)
        
        # Build query URL
        query_url = self.build_query_url(norad_cat_id, start_epoch, end_epoch)
        
        # Download data
        records = self.download_json(query_url)
        
        # Build union of all keys preserving order
        all_keys = []
        for rec in records:
            for k in rec.keys():
                if k not in all_keys:
                    all_keys.append(k)
        
        # Create DataFrame preserving all fields
        df = pd.DataFrame.from_records(records)
        df = df.reindex(columns=all_keys)
        
        # Save to Excel
        df.to_excel(output_path, index=False)
        
        print(f"✓ Saved {len(df)} rows to: {output_path}")
        print(f"  Date range: {start_epoch} to {end_epoch}")
        print(f"  Total columns: {len(df.columns)}")
        
        return df


def main():
    """Example usage and testing of SpaceTrackClient."""
    
    # Initialize client (credentials loaded from .env)
    client = SpaceTrackClient()
    
    df_grace = client.save_to_excel(
        norad_cat_id=39634,
        start_epoch='2023-01-01',
        end_epoch='2025-09-30',
        output_dir='raw_data'  # Changed to raw_data
    )
    
    print("\n--- Data Preview ---")
    print(df_grace.head())
    print(f"\nShape: {df_grace.shape}")
    print(f"Columns: {list(df_grace.columns[:10])}...")  # Show first 10 columns
    
  
    
 

    



if __name__ == "__main__":
    main()
