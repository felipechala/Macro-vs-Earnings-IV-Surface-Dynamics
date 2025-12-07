import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import glob
warnings.filterwarnings('ignore')

# Import from your other modules
from earningsDates import preprocess, get_next_earnings_date
from FOMCdates import get_fomc_dates, get_next_fomc_date


# ======================================================
# 1. Load option data from CSV/Excel
# ======================================================

def load_option_data_from_file(file_path, tickers=None):
    """
    Load option data from CSV/Excel file(s).
    
    Parameters:
      file_path: path to CSV/Excel file, OR list of file paths
      tickers: list of tickers to filter (optional)
    
    Returns:
      DataFrame with standardized column names
    """
    # Handle multiple files
    if isinstance(file_path, list):
        print(f"\nLoading option data from {len(file_path)} files...")
        all_dfs = []
        
        for i, path in enumerate(file_path, 1):
            print(f"\n--- Processing file {i}/{len(file_path)}: {path} ---")
            try:
                df = load_single_file(path, tickers)
                all_dfs.append(df)
                print(f"    Successfully loaded {len(df)} rows from {path}")
            except Exception as e:
                print(f"    Failed to load {path}: {e}")
                continue
        
        if not all_dfs:
            raise RuntimeError("No data loaded from any files")
        
        # Combine all dataframes
        print(f"\nCombining data from {len(all_dfs)} files...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates (same date, ticker, strike, cp_flag)
        if 'ticker' in combined_df.columns:
            dup_cols = ['date', 'ticker', 'impl_strike', 'cp_flag']
        else:
            dup_cols = ['date', 'impl_strike', 'cp_flag']
        
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dup_cols, keep='first')
        after_dedup = len(combined_df)
        
        if before_dedup > after_dedup:
            print(f"   Removed {before_dedup - after_dedup} duplicate rows")
        
        print(f"\nFinal combined dataset:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        if 'ticker' in combined_df.columns:
            print(f"  Tickers: {sorted(combined_df['ticker'].unique())}")
        
        return combined_df
    
    # Single file
    else:
        print(f"\nLoading option data from: {file_path}")
        return load_single_file(file_path, tickers)


def load_single_file(file_path, tickers=None):
    """
    Load option data from a single CSV or Excel file.
    
    Parameters:
      file_path: path to CSV or Excel file
      tickers: list of tickers to filter (optional)
    
    Returns:
      DataFrame with standardized column names
    """
    print(f"  Reading file...")
    
    # Determine file type and load
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Must be .csv, .xlsx, or .xls")
    
    print(f"  Raw data: {len(df)} rows, {len(df.columns)} columns")
    
    # Standardize column names
    # Try to map common variations
    column_mapping = {
        # Common variations of column names
        'Security ID': 'secid',
        'securityid': 'secid',
        'security_id': 'secid',
        'Date': 'date',
        'DATE': 'date',
        'Days to Expiration': 'days',
        'DaysToExpiration': 'days',
        'days_to_expiration': 'days',
        'dte': 'days',
        'Delta of the Option': 'delta',
        'Delta': 'delta',
        'DELTA': 'delta',
        'delta_option': 'delta',
        'Interpolated Implied Volatility of the Option': 'impl_volatility',
        'Implied Volatility': 'impl_volatility',
        'ImpliedVolatility': 'impl_volatility',
        'impl_volatility': 'impl_volatility',
        'iv': 'impl_volatility',
        'IV': 'impl_volatility',
        'The Strike Price Corresponding to this Delta': 'impl_strike',
        'Strike Price': 'impl_strike',
        'Strike': 'impl_strike',
        'strike': 'impl_strike',
        'impl_strike': 'impl_strike',
        'The Premium of a Theoretical Option with this Delta and Implied Volatility': 'impl_premium',
        'Premium': 'impl_premium',
        'premium': 'impl_premium',
        'impl_premium': 'impl_premium',
        'A Measure of the Accuracy of the Implied Volatility Calculation': 'dispersion',
        'Dispersion': 'dispersion',
        'dispersion': 'dispersion',
        'C=Call, P=Put': 'cp_flag',
        'CP Flag': 'cp_flag',
        'cp_flag': 'cp_flag',
        'cpflag': 'cp_flag',
        'call_put': 'cp_flag',
        'CUSIP Number': 'cusip',
        'CUSIP': 'cusip',
        'cusip': 'cusip',
        'Ticker Symbol': 'ticker',
        'Ticker': 'ticker',
        'ticker': 'ticker',
        'TICKER': 'ticker',
        'symbol': 'ticker',
        'SIC Code': 'sic',
        'SIC': 'sic',
        'sic': 'sic',
        'Index Flag': 'index_flag',
        'index_flag': 'index_flag',
        'Exchange Designator': 'exchange_d',
        'exchange_d': 'exchange_d',
        'Exchange': 'exchange_d',
        'Class Designator': 'class',
        'class': 'class',
        'The Type of Security': 'issue_type',
        'Issue Type': 'issue_type',
        'issue_type': 'issue_type'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError("No date column found in file")
    
    # Convert numeric columns
    numeric_cols = ['days', 'delta', 'impl_volatility', 'impl_strike', 'impl_premium', 'dispersion']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert delta to absolute decimal if needed
    # Check if delta is in percentage format (values > 1 or < -1)
    if 'delta' in df.columns:
        if df['delta'].abs().max() > 1:
            # Delta is in percentage format (e.g., -90, 85)
            df['delta'] = df['delta'].abs() / 100.0
        else:
            # Delta is already in decimal format, just take absolute value
            df['delta'] = df['delta'].abs()
    
    # Clean data
    df = df[~df['date'].isna()]
    
    if 'impl_volatility' in df.columns:
        df = df[~df['impl_volatility'].isna()]
        df = df[df['impl_volatility'] > 0]
    else:
        raise ValueError("No impl_volatility column found in file")
    
    if 'days' in df.columns:
        df = df[df['days'] > 0]
    
    # Filter by tickers if provided
    if tickers is not None and 'ticker' in df.columns:
        df = df[df['ticker'].isin(tickers)]
        print(f"  Filtered to tickers: {tickers}")
    
    # Sort by date
    if 'ticker' in df.columns:
        df = df.sort_values(['ticker', 'date', 'days']).reset_index(drop=True)
    else:
        df = df.sort_values(['date', 'days']).reset_index(drop=True)
    
    print(f"  Cleaned data: {len(df)} rows")
    
    return df


# ======================================================
# 2. Collect earnings dates from earnings scraper
# ======================================================

def collect_earnings_dates(tickers, start_date='2023-01-01'):
    """
    Collect earnings dates for multiple tickers using earnings_scraper.
    
    Returns:
      DataFrame with columns ['date', 'ticker']
    """
    all_earnings = []
    
    for ticker in tickers:
        print(f"Fetching earnings for {ticker}...")
        try:
            _, _, earnings, _ = preprocess(ticker, n_exp=4)
            
            if earnings is not None and not earnings.empty:
                for date in earnings.index:
                    all_earnings.append({
                        'date': pd.Timestamp(date).tz_localize(None),
                        'ticker': ticker
                    })
        except Exception as e:
            print(f"[WARN] Could not process {ticker}: {e}")
            continue
    
    if not all_earnings:
        print("[WARN] No earnings dates collected")
        return pd.DataFrame(columns=['date', 'ticker'])
    
    df = pd.DataFrame(all_earnings)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter to start_date onwards
    start = pd.Timestamp(start_date)
    df = df[df['date'] >= start]
    
    print(f"\nCollected {len(df)} earnings events across {df['ticker'].nunique()} tickers")
    return df


# ======================================================
# 3. Tag event windows
# ======================================================

def tag_event_windows(df, event_dates, event_type='earnings', window_days=3):
    """
    Tag observations within +/- window_days of events.
    For earnings: match by ticker and date
    For FOMC: apply to all tickers
    
    Returns df with added columns:
      - event_type
      - days_to_event
      - is_event_period
    """
    df = df.copy()
    df['event_type'] = None
    df['days_to_event'] = None
    df['is_event_period'] = False
    
    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'])
    
    if event_type == 'earnings' and isinstance(event_dates, pd.DataFrame):
        # Earnings: match by ticker
        for _, row in event_dates.iterrows():
            event_date = pd.Timestamp(row['date'])
            ticker = row['ticker']
            
            window_start = event_date - pd.Timedelta(days=window_days)
            window_end = event_date + pd.Timedelta(days=window_days)
            
            if 'ticker' in df.columns:
                mask = (df['ticker'] == ticker) & \
                       (df['date'] >= window_start) & \
                       (df['date'] <= window_end)
            else:
                mask = (df['date'] >= window_start) & (df['date'] <= window_end)
            
            df.loc[mask, 'days_to_event'] = (df.loc[mask, 'date'] - event_date).dt.days
            df.loc[mask, 'event_type'] = event_type
            df.loc[mask, 'is_event_period'] = True
    else:
        # FOMC: apply to all tickers
        event_dates = pd.to_datetime(event_dates)
        
        for event_date in event_dates:
            window_start = event_date - pd.Timedelta(days=window_days)
            window_end = event_date + pd.Timedelta(days=window_days)
            
            mask = (df['date'] >= window_start) & (df['date'] <= window_end)
            
            df.loc[mask, 'days_to_event'] = (df.loc[mask, 'date'] - event_date).dt.days
            df.loc[mask, 'event_type'] = event_type
            df.loc[mask, 'is_event_period'] = True
    
    return df


# ======================================================
# 4. IV surface metrics
# ======================================================

def compute_iv_surface_metrics(df, groupby_cols=['date', 'ticker']):
    """
    Compute IV surface characteristics:
      - Level: ATM IV (delta closest to 0.5)
      - Skew: IV(0.25 delta put) - IV(0.75 delta call)
      - Curvature: average wing IV - ATM IV
    """
    metrics = []
    
    for name, group in df.groupby(groupby_cols):
        if len(group) < 5:
            continue
        
        # ATM: delta closest to 0.5
        group = group.copy()
        group['delta_diff'] = (group['delta'] - 0.5).abs()
        
        if len(group) == 0:
            continue
            
        atm_idx = group['delta_diff'].idxmin()
        atm_iv = group.loc[atm_idx, 'impl_volatility']
        
        # Skew: 25-delta put (0.25) vs 25-delta call (0.75)
        delta_25_put = group[(group['cp_flag'] == 'P') & (group['delta'].between(0.20, 0.30))]
        delta_25_call = group[(group['cp_flag'] == 'C') & (group['delta'].between(0.70, 0.80))]
        
        if len(delta_25_put) > 0 and len(delta_25_call) > 0:
            skew = delta_25_put['impl_volatility'].mean() - delta_25_call['impl_volatility'].mean()
        else:
            skew = np.nan
        
        # Curvature: wing options (10-delta) vs ATM
        delta_10_put = group[(group['cp_flag'] == 'P') & (group['delta'].between(0.05, 0.15))]
        delta_10_call = group[(group['cp_flag'] == 'C') & (group['delta'].between(0.85, 0.95))]
        
        if len(delta_10_put) > 0 and len(delta_10_call) > 0:
            wing_iv = (delta_10_put['impl_volatility'].mean() + 
                      delta_10_call['impl_volatility'].mean()) / 2
            curvature = wing_iv - atm_iv
        else:
            curvature = np.nan
        
        result = {
            'date': name[0] if isinstance(name, tuple) else name,
            'atm_iv': atm_iv,
            'skew': skew,
            'curvature': curvature,
            'n_options': len(group)
        }
        
        # Add ticker if in groupby
        if isinstance(name, tuple) and len(name) > 1:
            result['ticker'] = name[1]
        
        metrics.append(result)
    
    return pd.DataFrame(metrics)


# ======================================================
# 5. Mean reversion models
# ======================================================

def exponential_decay(t, a, tau):
    """Exponential decay: y(t) = a * exp(-t/tau)"""
    return a * np.exp(-t / tau)


def fit_mean_reversion(df, time_col='days_to_event', value_col='atm_iv'):
    """
    Fit AR(1) and exponential decay for mean reversion.
    """
    ts = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    
    if len(ts) < 10:
        return {
            'ar1_halflife': np.nan,
            'exp_halflife': np.nan,
            'ar1_coef': np.nan,
            'exp_tau': np.nan
        }
    
    # AR(1) model - use daily aggregates
    daily_avg = ts.groupby(time_col)[value_col].mean().reset_index()
    y = daily_avg[value_col].values
    
    if len(y) < 3:
        return {
            'ar1_halflife': np.nan,
            'exp_halflife': np.nan,
            'ar1_coef': np.nan,
            'exp_tau': np.nan
        }
    
    y_lag = np.roll(y, 1)[1:]
    y_curr = y[1:]
    
    if len(y_curr) > 0 and np.std(y_lag) > 0 and np.std(y_curr) > 0:
        coef = np.corrcoef(y_lag, y_curr)[0, 1]
        if 0 < coef < 1:
            ar1_halflife = -np.log(2) / np.log(coef)
        else:
            ar1_halflife = np.nan
    else:
        ar1_halflife = np.nan
        coef = np.nan
    
    # Exponential decay (post-event only)
    ts_post = daily_avg[daily_avg[time_col] >= 0]
    if len(ts_post) >= 3:
        try:
            baseline = daily_avg[daily_avg[time_col] < 0][value_col].mean()
            if np.isnan(baseline):
                baseline = ts_post[value_col].min()
            
            y_norm = ts_post[value_col].values - baseline
            t = ts_post[time_col].values
            
            if y_norm[0] > 0 and not np.all(y_norm <= 0):
                popt, _ = curve_fit(exponential_decay, t, y_norm, 
                                   p0=[y_norm[0], 1], maxfev=5000,
                                   bounds=([0, 0.1], [np.inf, 10]))
                tau = popt[1]
                exp_halflife = tau * np.log(2)
            else:
                exp_halflife = np.nan
                tau = np.nan
        except Exception as e:
            exp_halflife = np.nan
            tau = np.nan
    else:
        exp_halflife = np.nan
        tau = np.nan
    
    return {
        'ar1_halflife': ar1_halflife,
        'exp_halflife': exp_halflife,
        'ar1_coef': coef,
        'exp_tau': tau
    }


# ======================================================
# 6. Plotting
# ======================================================

def plot_iv_comparison(earnings_metrics, fomc_metrics, metric='atm_iv'):
    """
    Plot IV metric comparison for earnings vs FOMC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Earnings
    ax = axes[0]
    if 'days_to_event' in earnings_metrics.columns:
        earn_grouped = earnings_metrics.groupby('days_to_event')[metric].agg(['mean', 'std', 'count'])
        earn_grouped = earn_grouped[earn_grouped['count'] >= 5]
        
        if len(earn_grouped) > 0:
            ax.plot(earn_grouped.index, earn_grouped['mean'], 'o-', 
                    linewidth=2, markersize=6, label='Mean')
            ax.fill_between(earn_grouped.index,
                            earn_grouped['mean'] - earn_grouped['std'],
                            earn_grouped['mean'] + earn_grouped['std'],
                            alpha=0.3, label='±1 SD')
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Event')
    ax.set_xlabel('Days from Earnings', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Earnings Events', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FOMC
    ax = axes[1]
    if 'days_to_event' in fomc_metrics.columns:
        fomc_grouped = fomc_metrics.groupby('days_to_event')[metric].agg(['mean', 'std', 'count'])
        fomc_grouped = fomc_grouped[fomc_grouped['count'] >= 5]
        
        if len(fomc_grouped) > 0:
            ax.plot(fomc_grouped.index, fomc_grouped['mean'], 'o-',
                    linewidth=2, markersize=6, color='orange', label='Mean')
            ax.fill_between(fomc_grouped.index,
                            fomc_grouped['mean'] - fomc_grouped['std'],
                            fomc_grouped['mean'] + fomc_grouped['std'],
                            alpha=0.3, color='orange', label='±1 SD')
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Event')
    ax.set_xlabel('Days from FOMC', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('FOMC Events', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mean_reversion_detail(earn_metrics, fomc_metrics):
    """
    Detailed mean reversion plot with fitted curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Earnings
    ax = axes[0]
    if 'days_to_event' in earn_metrics.columns and 'atm_iv' in earn_metrics.columns:
        earn_post = earn_metrics[earn_metrics['days_to_event'] >= 0].copy()
        
        if len(earn_post) > 5:
            daily = earn_post.groupby('days_to_event')['atm_iv'].mean().reset_index()
            ax.scatter(earn_post['days_to_event'], earn_post['atm_iv'], 
                      alpha=0.2, s=10, label='Individual obs.')
            ax.plot(daily['days_to_event'], daily['atm_iv'], 
                   'o-', linewidth=2, markersize=8, label='Daily avg')
            
            try:
                baseline = earn_metrics[earn_metrics['days_to_event'] < 0]['atm_iv'].mean()
                if np.isnan(baseline):
                    baseline = daily['atm_iv'].min()
                    
                y_norm = daily['atm_iv'].values - baseline
                t = daily['days_to_event'].values
                
                if y_norm[0] > 0:
                    popt, _ = curve_fit(exponential_decay, t, y_norm, 
                                       p0=[y_norm[0], 1], maxfev=5000,
                                       bounds=([0, 0.1], [np.inf, 10]))
                    
                    t_fit = np.linspace(0, daily['days_to_event'].max(), 100)
                    y_fit = exponential_decay(t_fit, *popt) + baseline
                    ax.plot(t_fit, y_fit, 'r-', linewidth=2, 
                           label=f'Exp fit (τ={popt[1]:.2f}d, t½={popt[1]*np.log(2):.2f}d)')
            except Exception as e:
                print(f"[WARN] Earnings exp fit failed: {e}")
    
    ax.set_xlabel('Days After Earnings', fontsize=12)
    ax.set_ylabel('ATM IV', fontsize=12)
    ax.set_title('Earnings: Mean Reversion', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FOMC
    ax = axes[1]
    if 'days_to_event' in fomc_metrics.columns and 'atm_iv' in fomc_metrics.columns:
        fomc_post = fomc_metrics[fomc_metrics['days_to_event'] >= 0].copy()
        
        if len(fomc_post) > 5:
            daily = fomc_post.groupby('days_to_event')['atm_iv'].mean().reset_index()
            ax.scatter(fomc_post['days_to_event'], fomc_post['atm_iv'],
                      alpha=0.2, s=10, color='orange', label='Individual obs.')
            ax.plot(daily['days_to_event'], daily['atm_iv'],
                   'o-', linewidth=2, markersize=8, color='orange', label='Daily avg')
            
            try:
                baseline = fomc_metrics[fomc_metrics['days_to_event'] < 0]['atm_iv'].mean()
                if np.isnan(baseline):
                    baseline = daily['atm_iv'].min()
                    
                y_norm = daily['atm_iv'].values - baseline
                t = daily['days_to_event'].values
                
                if y_norm[0] > 0:
                    popt, _ = curve_fit(exponential_decay, t, y_norm,
                                       p0=[y_norm[0], 1], maxfev=5000,
                                       bounds=([0, 0.1], [np.inf, 10]))
                    
                    t_fit = np.linspace(0, daily['days_to_event'].max(), 100)
                    y_fit = exponential_decay(t_fit, *popt) + baseline
                    ax.plot(t_fit, y_fit, 'r-', linewidth=2,
                           label=f'Exp fit (τ={popt[1]:.2f}d, t½={popt[1]*np.log(2):.2f}d)')
            except Exception as e:
                print(f"[WARN] FOMC exp fit failed: {e}")
    
    ax.set_xlabel('Days After FOMC', fontsize=12)
    ax.set_ylabel('ATM IV', fontsize=12)
    ax.set_title('FOMC: Mean Reversion', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ======================================================
# 7. Main analysis
# ======================================================

def run_integrated_analysis(file_paths, tickers, output_prefix='event_analysis'):
    """
    Main pipeline:
      1. Load option data from CSV/Excel file(s)
      2. Collect earnings dates for tickers
      3. Load FOMC dates
      4. Tag event windows
      5. Compute IV metrics
      6. Fit mean reversion models
      7. Generate plots and summary
    
    Parameters:
      file_paths: path to CSV/Excel file, OR list of file paths
      tickers: list of tickers to analyze
      output_prefix: prefix for output files
    """
    print("="*70)
    print("INTEGRATED MACRO VS EARNINGS EVENT ANALYSIS")
    print("="*70)
    
    # Step 1: Load option data from file(s)
    print("\n1. Loading option data from file(s)...")
    option_data = load_option_data_from_file(file_paths, tickers=tickers)
    
    # Step 2: Collect earnings dates
    print("\n2. Collecting earnings dates...")
    earnings_df = collect_earnings_dates(tickers, start_date='2023-01-01')
    print(f"   Found {len(earnings_df)} earnings events")
    
    # Step 3: Load FOMC dates
    print("\n3. Loading FOMC dates...")
    fomc_df = get_fomc_dates()
    fomc_dates = fomc_df['date'].values
    print(f"   Found {len(fomc_dates)} FOMC dates")
    
    # Step 4: Tag event windows
    print("\n4. Tagging event windows...")
    df_earn = tag_event_windows(option_data.copy(), earnings_df, 'earnings', window_days=3)
    df_fomc = tag_event_windows(option_data.copy(), fomc_dates, 'fomc', window_days=3)
    
    n_earn_obs = df_earn['is_event_period'].sum()
    n_fomc_obs = df_fomc['is_event_period'].sum()
    print(f"   Earnings window: {n_earn_obs} observations")
    print(f"   FOMC window: {n_fomc_obs} observations")
    
    # Step 5: Compute IV metrics
    print("\n5. Computing IV surface metrics...")
    
    # Earnings events
    earn_events = df_earn[df_earn['is_event_period']].copy()
    groupby_cols = ['date', 'ticker'] if 'ticker' in earn_events.columns else ['date']
    earn_metrics = compute_iv_surface_metrics(earn_events, groupby_cols=groupby_cols)
    
    # Merge back days_to_event
    if 'ticker' in earn_events.columns:
        earn_window_info = earn_events.groupby(['date', 'ticker']).agg({
            'days_to_event': 'first',
            'event_type': 'first'
        }).reset_index()
        earn_metrics = earn_metrics.merge(earn_window_info, on=['date', 'ticker'], how='left')
    else:
        earn_window_info = earn_events.groupby('date').agg({
            'days_to_event': 'first',
            'event_type': 'first'
        }).reset_index()
        earn_metrics = earn_metrics.merge(earn_window_info, on='date', how='left')
    
    # FOMC events
    fomc_events = df_fomc[df_fomc['is_event_period']].copy()
    groupby_cols = ['date', 'ticker'] if 'ticker' in fomc_events.columns else ['date']
    fomc_metrics = compute_iv_surface_metrics(fomc_events, groupby_cols=groupby_cols)
    
    if 'ticker' in fomc_events.columns:
        fomc_window_info = fomc_events.groupby(['date', 'ticker']).agg({
            'days_to_event': 'first',
            'event_type': 'first'
        }).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_window_info, on=['date', 'ticker'], how='left')
    else:
        fomc_window_info = fomc_events.groupby('date').agg({
            'days_to_event': 'first',
            'event_type': 'first'
        }).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_window_info, on='date', how='left')
    
    print(f"   Earnings: {len(earn_metrics)} metric observations")
    print(f"   FOMC: {len(fomc_metrics)} metric observations")
    
    # Step 6: Fit mean reversion
    print("\n6. Fitting mean reversion models...")
    earn_reversion = fit_mean_reversion(earn_metrics, 'days_to_event', 'atm_iv')
    fomc_reversion = fit_mean_reversion(fomc_metrics, 'days_to_event', 'atm_iv')
    
    print("\n   Mean Reversion Half-Lives:")
    print(f"   Earnings - AR(1): {earn_reversion['ar1_halflife']:.2f} days")
    print(f"   Earnings - Exp:   {earn_reversion['exp_halflife']:.2f} days")
    print(f"   FOMC - AR(1):     {fomc_reversion['ar1_halflife']:.2f} days")
    print(f"   FOMC - Exp:       {fomc_reversion['exp_halflife']:.2f} days")
    
    # Step 7: Generate plots
    print("\n7. Generating plots...")
    
    # ATM IV
    fig1 = plot_iv_comparison(earn_metrics, fomc_metrics, 'atm_iv')
    fig1.savefig(f'{output_prefix}_atm_iv.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_atm_iv.png")
    
    # Skew
    fig2 = plot_iv_comparison(earn_metrics, fomc_metrics, 'skew')
    fig2.savefig(f'{output_prefix}_skew.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_skew.png")
    
    # Curvature
    fig3 = plot_iv_comparison(earn_metrics, fomc_metrics, 'curvature')
    fig3.savefig(f'{output_prefix}_curvature.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_curvature.png")
    
    # Mean reversion detail
    fig4 = plot_mean_reversion_detail(earn_metrics, fomc_metrics)
    fig4.savefig(f'{output_prefix}_mean_reversion.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_mean_reversion.png")
    
    # Summary table
    summary = pd.DataFrame({
        'Event Type': ['Earnings', 'FOMC'],
        'N Events': [len(earnings_df), len(fomc_dates)],
        'N Observations': [len(earn_metrics), len(fomc_metrics)],
        'AR(1) Half-Life (days)': [earn_reversion['ar1_halflife'], fomc_reversion['ar1_halflife']],
        'Exp Half-Life (days)': [earn_reversion['exp_halflife'], fomc_reversion['exp_halflife']],
        'AR(1) Coefficient': [earn_reversion['ar1_coef'], fomc_reversion['ar1_coef']],
        'Exp Time Constant (τ)': [earn_reversion['exp_tau'], fomc_reversion['exp_tau']]
    })
    
    summary.to_csv(f'{output_prefix}_summary.csv', index=False)
    print(f"   Saved: {output_prefix}_summary.csv")
    
    # Save detailed metrics
    earn_metrics.to_csv(f'{output_prefix}_earnings_metrics.csv', index=False)
    fomc_metrics.to_csv(f'{output_prefix}_fomc_metrics.csv', index=False)
    print(f"   Saved: {output_prefix}_earnings_metrics.csv")
    print(f"   Saved: {output_prefix}_fomc_metrics.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return {
        'earnings_metrics': earn_metrics,
        'fomc_metrics': fomc_metrics,
        'earnings_reversion': earn_reversion,
        'fomc_reversion': fomc_reversion,
        'summary': summary
    }

if __name__ == "__main__":
    
    data_file = glob.glob("Data/*.csv")  # All CSVs in 'Data' folder
    data_file = sorted(data_file)  # Sort by filename
    
    # Tickers to analyze
    tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet Class A
    "GOOG",  # Alphabet Class C
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "META",  # Meta Platforms (Facebook)
    "NFLX",  # Netflix
    "AMD",   # Advanced Micro Devices
    "INTC",  # Intel
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "PFE",   # Pfizer
    "BAC",   # Bank of America
    "WMT",   # Walmart
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "JNJ",   # Johnson & Johnson
    "DIS",   # Disney
    "CRM",   # Salesforce
    "T",     # AT&T
    "VZ",    # Verizon
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "PG",    # Procter & Gamble
    "TSM",   # Taiwan Semiconductor
    "ADBE",  # Adobe
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq-100 ETF
    "IWM",   # Russell 2000 ETF
    "EEM",   # Emerging Markets ETF
    "GDX",   # Gold Miners ETF
    "XLF",   # Financials Select Sector SPDR
    "GLD",   # SPDR Gold Shares
    "HYG",   # iShares High-Yield Corp Bond ETF
    "TLT",   # iShares 20+ Year Treasury Bond ETF
    "SLV",   # iShares Silver Trust
    "SOXL",  # Direxion Semiconductor Bull 3x ETF
    "TQQQ",  # ProShares UltraPro QQQ 3x ETF
    "SQQQ",  # ProShares UltraPro Short QQQ 3x ETF
    "SPXL",  # Direxion S&P 500 Bull 3x Shares ETF
    "XLE",   # Energy Select Sector SPDR ETF
    "XLK",   # Technology Select Sector SPDR ETF
    "XLY"    # Consumer Discretionary Select Sector SPDR ETF
]
    
    # Run analysis
    results = run_integrated_analysis(
        file_paths=data_file,
        tickers=tickers,
        output_prefix='macro_vs_earnings'
    )
    
    # Display summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(results['summary'].to_string(index=False))