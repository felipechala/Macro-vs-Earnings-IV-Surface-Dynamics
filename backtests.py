import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from earningsDates import preprocess
from FOMCdates import get_fomc_dates
from reversion import (
    load_option_data_from_file,
    collect_earnings_dates,
    tag_event_windows,
    compute_iv_surface_metrics
)


# ======================================================
# 1. Strategy P&L calculations
# ======================================================

def butterfly_pnl(atm_iv_entry, atm_iv_exit, curvature_entry, curvature_exit, 
                  days_held, notional=100):
    """
    Butterfly spread P&L.
    Long 2 ATM, Short 2 wings (OTM put + call).
    Profits from IV collapse.
    """
    # ATM IV collapse (negative vega position)
    atm_change = atm_iv_exit - atm_iv_entry
    vega_pnl = -2.0 * atm_change * notional
    
    # Wing IV change (positive vega from short position, but lower sensitivity)
    wing_change = curvature_exit - curvature_entry
    vega_pnl += 0.8 * wing_change * notional
    
    # Time decay benefit
    theta_pnl = days_held * notional * 0.015
    
    return vega_pnl + theta_pnl


def calendar_pnl(iv_entry, iv_exit, days_held, notional=100):
    """
    Calendar spread P&L.
    Short near-term, Long far-term.
    Profits from time decay differential.
    """
    # Near-term decay (beneficial - we're short)
    near_decay = days_held * notional * 0.025
    
    # Far-term decay (harmful - we're long)
    far_decay = -days_held * notional * 0.008
    
    # IV change (short near has negative vega, long far has positive)
    iv_change = iv_exit - iv_entry
    vega_pnl = -0.6 * iv_change * notional + 0.4 * iv_change * notional
    
    return near_decay + far_decay + vega_pnl


def dispersion_pnl(stock_iv_entry, stock_iv_exit, index_iv_entry, index_iv_exit, 
                   notional=100):
    """
    Dispersion trade P&L.
    Long single-stock vol, Short index vol.
    Profits from stock vol outperforming index vol.
    """
    stock_change = stock_iv_exit - stock_iv_entry
    index_change = index_iv_exit - index_iv_entry
    
    # Long stock vol
    stock_pnl = stock_change * notional
    
    # Short index vol
    index_pnl = -index_change * notional
    
    return stock_pnl + index_pnl


# ======================================================
# 2. Backtesting engine
# ======================================================

def backtest_strategy(metrics_df, strategy_type, entry_day, exit_day, notional=100):
    """
    Backtest a strategy across all events.
    
    Parameters:
      metrics_df: DataFrame with IV metrics and days_to_event
      strategy_type: 'butterfly', 'calendar', or 'dispersion'
      entry_day: days relative to event (e.g., -1 for T-1)
      exit_day: days relative to event (e.g., 1 for T+1)
      notional: position size
    
    Returns:
      DataFrame of trades with P&L
    """
    trades = []
    
    if 'days_to_event' not in metrics_df.columns:
        return pd.DataFrame()
    
    # Group by event
    if 'ticker' in metrics_df.columns:
        events = metrics_df.groupby(['date', 'ticker']).first().reset_index()
    else:
        events = metrics_df.groupby('date').first().reset_index()
    
    for _, event in events.iterrows():
        event_date = event['date']
        ticker = event.get('ticker', 'INDEX')
        
        # Get entry data
        if 'ticker' in metrics_df.columns:
            entry = metrics_df[
                (metrics_df['ticker'] == ticker) & 
                (metrics_df['days_to_event'] == entry_day)
            ]
            exit = metrics_df[
                (metrics_df['ticker'] == ticker) & 
                (metrics_df['days_to_event'] == exit_day)
            ]
        else:
            entry = metrics_df[metrics_df['days_to_event'] == entry_day]
            exit = metrics_df[metrics_df['days_to_event'] == exit_day]
        
        if len(entry) == 0 or len(exit) == 0:
            continue
        
        entry = entry.iloc[0]
        exit = exit.iloc[0]
        
        # Extract IVs
        atm_iv_entry = entry.get('atm_iv', np.nan)
        atm_iv_exit = exit.get('atm_iv', np.nan)
        curv_entry = entry.get('curvature', 0)
        curv_exit = exit.get('curvature', 0)
        
        if np.isnan(atm_iv_entry) or np.isnan(atm_iv_exit):
            continue
        
        # Calculate P&L based on strategy type
        days_held = abs(exit_day - entry_day)
        
        if strategy_type == 'butterfly':
            pnl = butterfly_pnl(atm_iv_entry, atm_iv_exit, curv_entry, curv_exit, 
                               days_held, notional)
        elif strategy_type == 'calendar':
            pnl = calendar_pnl(atm_iv_entry, atm_iv_exit, days_held, notional)
        else:  # dispersion handled separately
            continue
        
        trades.append({
            'date': event_date,
            'ticker': ticker,
            'entry_day': entry_day,
            'exit_day': exit_day,
            'pnl': pnl,
            'iv_entry': atm_iv_entry,
            'iv_exit': atm_iv_exit,
            'iv_change': atm_iv_exit - atm_iv_entry
        })
    
    return pd.DataFrame(trades)


def backtest_dispersion(single_stock_df, index_df, entry_day, exit_day, notional=100):
    """
    Backtest dispersion trades (single-stock vol vs index vol).
    """
    trades = []
    
    if 'days_to_event' not in single_stock_df.columns or 'days_to_event' not in index_df.columns:
        return pd.DataFrame()
    
    # Get common dates
    stock_dates = single_stock_df[single_stock_df['days_to_event'] == entry_day]['date'].unique()
    index_dates = index_df[index_df['days_to_event'] == entry_day]['date'].unique()
    common_dates = set(stock_dates).intersection(set(index_dates))
    
    for date in common_dates:
        # Stock entry/exit
        stock_entry = single_stock_df[
            (single_stock_df['date'] == date) & 
            (single_stock_df['days_to_event'] == entry_day)
        ]
        stock_exit = single_stock_df[
            (single_stock_df['date'] == date) & 
            (single_stock_df['days_to_event'] == exit_day)
        ]
        
        # Index entry/exit
        index_entry = index_df[
            (index_df['date'] == date) & 
            (index_df['days_to_event'] == entry_day)
        ]
        index_exit = index_df[
            (index_df['date'] == date) & 
            (index_df['days_to_event'] == exit_day)
        ]
        
        if (len(stock_entry) == 0 or len(stock_exit) == 0 or 
            len(index_entry) == 0 or len(index_exit) == 0):
            continue
        
        stock_iv_entry = stock_entry.iloc[0].get('atm_iv', np.nan)
        stock_iv_exit = stock_exit.iloc[0].get('atm_iv', np.nan)
        index_iv_entry = index_entry.iloc[0].get('atm_iv', np.nan)
        index_iv_exit = index_exit.iloc[0].get('atm_iv', np.nan)
        
        if any(np.isnan([stock_iv_entry, stock_iv_exit, index_iv_entry, index_iv_exit])):
            continue
        
        pnl = dispersion_pnl(stock_iv_entry, stock_iv_exit, 
                            index_iv_entry, index_iv_exit, notional)
        
        trades.append({
            'date': date,
            'entry_day': entry_day,
            'exit_day': exit_day,
            'pnl': pnl,
            'stock_iv_change': stock_iv_exit - stock_iv_entry,
            'index_iv_change': index_iv_exit - index_iv_entry,
            'dispersion': (stock_iv_exit - stock_iv_entry) - (index_iv_exit - index_iv_entry)
        })
    
    return pd.DataFrame(trades)


# ======================================================
# 3. Performance metrics
# ======================================================

def calculate_metrics(trades_df, rf_rate=0.02):
    """
    Calculate risk-adjusted performance metrics.
    """
    if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
        return {
            'n_trades': 0,
            'total_pnl': np.nan,
            'mean_pnl': np.nan,
            'std_pnl': np.nan,
            'sharpe': np.nan,
            'information_ratio': np.nan,
            'max_drawdown': np.nan,
            'win_rate': np.nan
        }
    
    pnl = trades_df['pnl'].values
    n = len(pnl)
    
    total = pnl.sum()
    mean = pnl.mean()
    std = pnl.std()
    
    # Sharpe ratio (annualized, assuming ~12 trades/year)
    if std > 0:
        sharpe = (mean - rf_rate/12) / std * np.sqrt(12)
        info_ratio = mean / std
    else:
        sharpe = np.nan
        info_ratio = np.nan
    
    # Max drawdown
    cumulative = pnl.cumsum()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    # Win rate
    wins = (pnl > 0).sum()
    win_rate = wins / n
    
    return {
        'n_trades': n,
        'total_pnl': total,
        'mean_pnl': mean,
        'std_pnl': std,
        'sharpe': sharpe,
        'information_ratio': info_ratio,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }


# ======================================================
# 4. Visualization
# ======================================================

def plot_strategy_comparison(results_df, title):
    """
    Plot performance metrics comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Filter valid strategies
    valid = results_df[results_df['n_trades'] > 0].copy()
    
    if len(valid) == 0:
        print("[WARN] No valid trades to plot")
        return fig
    
    strategies = valid['strategy'].values
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#C1292E', '#8B4789']
    
    # Total P&L
    ax = axes[0, 0]
    bars = ax.bar(range(len(strategies)), valid['total_pnl'], 
                   color=colors[:len(strategies)], alpha=0.8)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Total P&L ($)', fontsize=11)
    ax.set_title('Total P&L', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sharpe Ratio
    ax = axes[0, 1]
    sharpe = valid['sharpe'].fillna(0)
    bars = ax.bar(range(len(strategies)), sharpe, 
                   color=colors[:len(strategies)], alpha=0.8)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Win Rate
    ax = axes[1, 0]
    bars = ax.bar(range(len(strategies)), valid['win_rate'] * 100, 
                   color=colors[:len(strategies)], alpha=0.8)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win Rate', fontsize=12, fontweight='bold')
    ax.axhline(50, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Max Drawdown
    ax = axes[1, 1]
    bars = ax.bar(range(len(strategies)), valid['max_drawdown'], 
                   color=colors[:len(strategies)], alpha=0.8)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Max Drawdown ($)', fontsize=11)
    ax.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_cumulative_pnl(trades_list, strategy_names, title):
    """
    Plot cumulative P&L over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#C1292E', '#8B4789']
    
    for trades, name, color in zip(trades_list, strategy_names, colors):
        if len(trades) > 0 and 'pnl' in trades.columns:
            sorted_trades = trades.sort_values('date')
            cumulative = sorted_trades['pnl'].cumsum()
            ax.plot(range(len(cumulative)), cumulative, 
                   label=name, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ======================================================
# 5. Main backtesting pipeline
# ======================================================

def run_backtest(file_paths, tickers, output_prefix='backtest'):
    """
    Main backtesting pipeline.
    
    Tests:
      - Butterfly spreads
      - Calendar spreads
      - Dispersion trades
      
    For:
      - Earnings events
      - FOMC events
      
    Comparing:
      - Pre-positioning (T-1 → T+1)
      - Post-event reversion (T+1 → T+3)
    """
    print("="*70)
    print("OPTION STRATEGY BACKTESTING")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    option_data = load_option_data_from_file(file_paths, tickers=tickers)
    
    # Collect events
    print("\n2. Collecting events...")
    single_stocks = [t for t in tickers if t not in ['QQQ', 'IWM', 'SPY', 'SPX']]
    indices = [t for t in tickers if t in ['QQQ', 'IWM', 'SPY', 'SPX']]
    
    earnings_df = collect_earnings_dates(single_stocks if single_stocks else tickers, 
                                         start_date='2023-01-01')
    fomc_df = get_fomc_dates()
    fomc_dates = fomc_df['date'].values
    
    print(f"   Earnings: {len(earnings_df)} events")
    print(f"   FOMC: {len(fomc_dates)} events")
    
    # Tag events
    print("\n3. Tagging event windows...")
    df_earn = tag_event_windows(option_data.copy(), earnings_df, 'earnings', window_days=5)
    df_fomc = tag_event_windows(option_data.copy(), fomc_dates, 'fomc', window_days=5)
    
    # Compute IV metrics
    print("\n4. Computing IV metrics...")
    
    # Earnings
    earn_events = df_earn[df_earn['is_event_period']].copy()
    groupby_cols = ['date', 'ticker'] if 'ticker' in earn_events.columns else ['date']
    earn_metrics = compute_iv_surface_metrics(earn_events, groupby_cols=groupby_cols)
    
    if 'ticker' in earn_events.columns:
        earn_info = earn_events.groupby(['date', 'ticker']).agg({
            'days_to_event': 'first'
        }).reset_index()
        earn_metrics = earn_metrics.merge(earn_info, on=['date', 'ticker'], how='left')
    else:
        earn_info = earn_events.groupby('date').agg({
            'days_to_event': 'first'
        }).reset_index()
        earn_metrics = earn_metrics.merge(earn_info, on='date', how='left')
    
    # FOMC
    fomc_events = df_fomc[df_fomc['is_event_period']].copy()
    groupby_cols = ['date', 'ticker'] if 'ticker' in fomc_events.columns else ['date']
    fomc_metrics = compute_iv_surface_metrics(fomc_events, groupby_cols=groupby_cols)
    
    if 'ticker' in fomc_events.columns:
        fomc_info = fomc_events.groupby(['date', 'ticker']).agg({
            'days_to_event': 'first'
        }).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_info, on=['date', 'ticker'], how='left')
    else:
        fomc_info = fomc_events.groupby('date').agg({
            'days_to_event': 'first'
        }).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_info, on='date', how='left')
    
    print(f"   Earnings: {len(earn_metrics)} observations")
    print(f"   FOMC: {len(fomc_metrics)} observations")
    
    # Backtest strategies
    print("\n5. Backtesting strategies...")
    all_results = []
    all_trades = {}
    
    # Pre-positioning strategies (T-1 → T+1)
    print("\n   Pre-positioning (T-1 → T+1):")
    
    print("      - Earnings Butterfly...")
    earn_fly_pre = backtest_strategy(earn_metrics, 'butterfly', -1, 1)
    metrics = calculate_metrics(earn_fly_pre)
    metrics['strategy'] = 'Earnings Butterfly (Pre)'
    metrics['event_type'] = 'Earnings'
    metrics['timing'] = 'Pre'
    all_results.append(metrics)
    all_trades['earn_fly_pre'] = earn_fly_pre
    
    print("      - Earnings Calendar...")
    earn_cal_pre = backtest_strategy(earn_metrics, 'calendar', -1, 1)
    metrics = calculate_metrics(earn_cal_pre)
    metrics['strategy'] = 'Earnings Calendar (Pre)'
    metrics['event_type'] = 'Earnings'
    metrics['timing'] = 'Pre'
    all_results.append(metrics)
    all_trades['earn_cal_pre'] = earn_cal_pre
    
    print("      - FOMC Butterfly...")
    fomc_fly_pre = backtest_strategy(fomc_metrics, 'butterfly', -1, 1)
    metrics = calculate_metrics(fomc_fly_pre)
    metrics['strategy'] = 'FOMC Butterfly (Pre)'
    metrics['event_type'] = 'FOMC'
    metrics['timing'] = 'Pre'
    all_results.append(metrics)
    all_trades['fomc_fly_pre'] = fomc_fly_pre
    
    print("      - FOMC Calendar...")
    fomc_cal_pre = backtest_strategy(fomc_metrics, 'calendar', -1, 1)
    metrics = calculate_metrics(fomc_cal_pre)
    metrics['strategy'] = 'FOMC Calendar (Pre)'
    metrics['event_type'] = 'FOMC'
    metrics['timing'] = 'Pre'
    all_results.append(metrics)
    all_trades['fomc_cal_pre'] = fomc_cal_pre
    
    # Dispersion (if applicable)
    if single_stocks and indices:
        print("      - Dispersion...")
        stock_metrics = earn_metrics[earn_metrics['ticker'].isin(single_stocks)]
        index_metrics = fomc_metrics
        disp_pre = backtest_dispersion(stock_metrics, index_metrics, -1, 1)
        metrics = calculate_metrics(disp_pre)
        metrics['strategy'] = 'Dispersion (Pre)'
        metrics['event_type'] = 'Mixed'
        metrics['timing'] = 'Pre'
        all_results.append(metrics)
        all_trades['disp_pre'] = disp_pre
    
    # Post-event reversion (T+1 → T+3)
    print("\n   Post-event reversion (T+1 → T+3):")
    
    print("      - Earnings Butterfly...")
    earn_fly_post = backtest_strategy(earn_metrics, 'butterfly', 1, 3)
    metrics = calculate_metrics(earn_fly_post)
    metrics['strategy'] = 'Earnings Butterfly (Post)'
    metrics['event_type'] = 'Earnings'
    metrics['timing'] = 'Post'
    all_results.append(metrics)
    all_trades['earn_fly_post'] = earn_fly_post
    
    print("      - Earnings Calendar...")
    earn_cal_post = backtest_strategy(earn_metrics, 'calendar', 1, 3)
    metrics = calculate_metrics(earn_cal_post)
    metrics['strategy'] = 'Earnings Calendar (Post)'
    metrics['event_type'] = 'Earnings'
    metrics['timing'] = 'Post'
    all_results.append(metrics)
    all_trades['earn_cal_post'] = earn_cal_post
    
    print("      - FOMC Butterfly...")
    fomc_fly_post = backtest_strategy(fomc_metrics, 'butterfly', 1, 3)
    metrics = calculate_metrics(fomc_fly_post)
    metrics['strategy'] = 'FOMC Butterfly (Post)'
    metrics['event_type'] = 'FOMC'
    metrics['timing'] = 'Post'
    all_results.append(metrics)
    all_trades['fomc_fly_post'] = fomc_fly_post
    
    print("      - FOMC Calendar...")
    fomc_cal_post = backtest_strategy(fomc_metrics, 'calendar', 1, 3)
    metrics = calculate_metrics(fomc_cal_post)
    metrics['strategy'] = 'FOMC Calendar (Post)'
    metrics['event_type'] = 'FOMC'
    metrics['timing'] = 'Post'
    all_results.append(metrics)
    all_trades['fomc_cal_post'] = fomc_cal_post
    
    if single_stocks and indices:
        print("      - Dispersion...")
        disp_post = backtest_dispersion(stock_metrics, index_metrics, 1, 3)
        metrics = calculate_metrics(disp_post)
        metrics['strategy'] = 'Dispersion (Post)'
        metrics['event_type'] = 'Mixed'
        metrics['timing'] = 'Post'
        all_results.append(metrics)
        all_trades['disp_post'] = disp_post
    
    # Compile results
    results_df = pd.DataFrame(all_results)
    
    print("\n6. Performance Summary:")
    print(results_df[['strategy', 'n_trades', 'total_pnl', 'sharpe', 'win_rate']].to_string(index=False))
    
    # Generate plots
    print("\n7. Generating visualizations...")
    
    # Pre-positioning comparison
    pre_results = results_df[results_df['timing'] == 'Pre']
    fig1 = plot_strategy_comparison(pre_results, 'Pre-Positioning Strategy Performance')
    fig1.savefig(f'{output_prefix}_pre_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_pre_comparison.png")
    
    # Post-reversion comparison
    post_results = results_df[results_df['timing'] == 'Post']
    fig2 = plot_strategy_comparison(post_results, 'Post-Event Reversion Strategy Performance')
    fig2.savefig(f'{output_prefix}_post_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_post_comparison.png")
    
    # Cumulative P&L - Pre
    fig3 = plot_cumulative_pnl(
        [earn_fly_pre, earn_cal_pre, fomc_fly_pre, fomc_cal_pre],
        ['Earn Butterfly', 'Earn Calendar', 'FOMC Butterfly', 'FOMC Calendar'],
        'Cumulative P&L: Pre-Positioning Strategies'
    )
    fig3.savefig(f'{output_prefix}_cumulative_pre.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_cumulative_pre.png")
    
    # Cumulative P&L - Post
    fig4 = plot_cumulative_pnl(
        [earn_fly_post, earn_cal_post, fomc_fly_post, fomc_cal_post],
        ['Earn Butterfly', 'Earn Calendar', 'FOMC Butterfly', 'FOMC Calendar'],
        'Cumulative P&L: Post-Event Reversion Strategies'
    )
    fig4.savefig(f'{output_prefix}_cumulative_post.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_cumulative_post.png")
    
    # Save results
    print("\n8. Saving results...")
    results_df.to_csv(f'{output_prefix}_summary.csv', index=False)
    print(f"   Saved: {output_prefix}_summary.csv")
    
    # Save top trades
    for name, trades in all_trades.items():
        if len(trades) > 0:
            trades.to_csv(f'{output_prefix}_{name}_trades.csv', index=False)
    
    print("\n" + "="*70)
    print("BACKTESTING COMPLETE")
    print("="*70)
    
    return {
        'results': results_df,
        'trades': all_trades
    }


if __name__ == "__main__":
    
    data_file = glob.glob("Data/*.csv")
    data_file = sorted(data_file)

    # Tickers
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
    
    # Run backtest
    output = run_backtest(
        file_paths=data_file,
        tickers=tickers,
        output_prefix='strategy_backtest'
    )
    
    # Display key findings
    results = output['results']
    
    print("\n" + "="*70)
    print("TOP STRATEGIES BY SHARPE RATIO")
    print("="*70)
    top = results.sort_values('sharpe', ascending=False)
    print(top[['strategy', 'n_trades', 'total_pnl', 'sharpe', 'win_rate']].head().to_string(index=False))
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Pre vs Post
    pre_sharpe = results[results['timing'] == 'Pre']['sharpe'].mean()
    post_sharpe = results[results['timing'] == 'Post']['sharpe'].mean()
    print(f"\nPre-positioning avg Sharpe:  {pre_sharpe:.3f}")
    print(f"Post-reversion avg Sharpe:   {post_sharpe:.3f}")
    
    # Event type
    earn_sharpe = results[results['event_type'] == 'Earnings']['sharpe'].mean()
    fomc_sharpe = results[results['event_type'] == 'FOMC']['sharpe'].mean()
    print(f"\nEarnings avg Sharpe:         {earn_sharpe:.3f}")
    print(f"FOMC avg Sharpe:             {fomc_sharpe:.3f}")
    
    # Best strategy
    if len(top) > 0:
        best = top.iloc[0]
        print(f"\n" + "="*70)
        print("BEST STRATEGY DETAILS")
        print("="*70)
        print(f"\nStrategy:          {best['strategy']}")
        print(f"Number of Trades:  {best['n_trades']:.0f}")
        print(f"Total P&L:         ${best['total_pnl']:.2f}")
        print(f"Mean P&L/Trade:    ${best['mean_pnl']:.2f}")
        print(f"Std Dev:           ${best['std_pnl']:.2f}")
        print(f"Sharpe Ratio:      {best['sharpe']:.3f}")
        print(f"Information Ratio: {best['information_ratio']:.3f}")
        print(f"Win Rate:          {best['win_rate']*100:.1f}%")
        print(f"Max Drawdown:      ${best['max_drawdown']:.2f}")
    
    # Structure comparison
    print(f"\n" + "="*70)
    print("PERFORMANCE BY STRUCTURE")
    print("="*70)
    
    butterfly_results = results[results['strategy'].str.contains('Butterfly')]
    calendar_results = results[results['strategy'].str.contains('Calendar')]
    dispersion_results = results[results['strategy'].str.contains('Dispersion')]
    
    print(f"\nButterfly Spreads:")
    print(f"  Total P&L:     ${butterfly_results['total_pnl'].sum():.2f}")
    print(f"  Avg Sharpe:    {butterfly_results['sharpe'].mean():.3f}")
    print(f"  Avg Win Rate:  {butterfly_results['win_rate'].mean()*100:.1f}%")
    
    print(f"\nCalendar Spreads:")
    print(f"  Total P&L:     ${calendar_results['total_pnl'].sum():.2f}")
    print(f"  Avg Sharpe:    {calendar_results['sharpe'].mean():.3f}")
    print(f"  Avg Win Rate:  {calendar_results['win_rate'].mean()*100:.1f}%")
    
    if len(dispersion_results) > 0:
        print(f"\nDispersion Trades:")
        print(f"  Total P&L:     ${dispersion_results['total_pnl'].sum():.2f}")
        print(f"  Avg Sharpe:    {dispersion_results['sharpe'].mean():.3f}")
        print(f"  Avg Win Rate:  {dispersion_results['win_rate'].mean()*100:.1f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)