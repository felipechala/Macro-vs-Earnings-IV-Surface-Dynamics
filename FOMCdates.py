import pandas as pd

def get_fomc_dates():
    """
    Returns known FOMC meeting dates.
    Update this list periodically from: 
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    
    Returns:
      pd.DataFrame with columns ['date', 'year']
    """
    known_dates = [
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025 (scheduled)
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    ]
    
    df = pd.DataFrame({
        'date': pd.to_datetime(known_dates),
    })
    df['year'] = df['date'].dt.year
    
    return df


def get_next_fomc_date(df):
    """
    Returns the next FOMC meeting date (as a date object),
    or None if not available.
    """
    if df is None or df.empty:
        return None
    
    today = pd.Timestamp.now()
    future_dates = df[df['date'] > today]
    
    if future_dates.empty:
        return None
    
    return future_dates.iloc[0]['date'].date()


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":
    print("Loading FOMC meeting dates...\n")
    
    # Get all FOMC dates
    fomc_df = get_fomc_dates()
    
    print("All FOMC dates (2023-2025):")
    print(fomc_df)
    print()
    
    # Get next FOMC meeting
    next_fomc = get_next_fomc_date(fomc_df)
    print(f"Next FOMC meeting date: {next_fomc}")