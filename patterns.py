import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
warnings.filterwarnings('ignore')

# Import from other modules
from earningsDates import preprocess, get_next_earnings_date
from FOMCdates import get_fomc_dates, get_next_fomc_date
from reversion import (
    load_option_data_from_file,
    collect_earnings_dates,
    tag_event_windows,
    compute_iv_surface_metrics
)


# ======================================================
# 1. Pattern identification
# ======================================================

def identify_surface_patterns(metrics_df, event_type='earnings'):
    """
    Identify recurring patterns in IV surface dynamics around events.
    
    Patterns to detect:
      1. Pre-event buildup: ATM IV rising before event
      2. Post-event collapse: Sharp ATM IV drop after event
      3. Skew steepening: Increasing put-call skew before event
      4. Skew normalization: Skew returning to baseline after event
      5. Curvature increase: Wings getting more expensive (smile deepening)
      6. Curvature decrease: Smile flattening post-event
    
    Returns:
      Dictionary with pattern statistics and classification
    """
    patterns = {
        'event_type': event_type,
        'n_observations': len(metrics_df)
    }
    
    if 'days_to_event' not in metrics_df.columns:
        print("[WARN] No days_to_event column, skipping pattern analysis")
        return patterns
    
    # Separate pre and post event
    pre_event = metrics_df[metrics_df['days_to_event'] < 0].copy()
    post_event = metrics_df[metrics_df['days_to_event'] > 0].copy()
    event_day = metrics_df[metrics_df['days_to_event'] == 0].copy()
    
    # Pattern 1: ATM IV buildup (pre-event)
    if len(pre_event) >= 2:
        pre_early = pre_event[pre_event['days_to_event'] <= -2]['atm_iv'].mean()
        pre_late = pre_event[pre_event['days_to_event'] >= -1]['atm_iv'].mean()
        
        if not np.isnan(pre_early) and not np.isnan(pre_late) and pre_early > 0:
            buildup = pre_late - pre_early
            buildup_pct = (buildup / pre_early) * 100
            patterns['atm_buildup_abs'] = buildup
            patterns['atm_buildup_pct'] = buildup_pct
            patterns['atm_buildup_detected'] = buildup_pct > 5  # >5% increase
            patterns['pre_early_iv'] = pre_early
            patterns['pre_late_iv'] = pre_late
    
    # Pattern 2: ATM IV collapse (post-event)
    if len(event_day) > 0 and len(post_event) >= 2:
        event_iv = event_day['atm_iv'].mean()
        post_1d = post_event[post_event['days_to_event'] == 1]['atm_iv'].mean()
        
        if not np.isnan(event_iv) and not np.isnan(post_1d) and event_iv > 0:
            collapse = event_iv - post_1d
            collapse_pct = (collapse / event_iv) * 100
            patterns['atm_collapse_abs'] = collapse
            patterns['atm_collapse_pct'] = collapse_pct
            patterns['atm_collapse_detected'] = collapse_pct > 10  # >10% drop
            patterns['event_day_iv'] = event_iv
            patterns['post_1d_iv'] = post_1d
    
    # Pattern 3: Skew steepening (pre-event)
    if len(pre_event) >= 2 and 'skew' in pre_event.columns:
        pre_early_skew = pre_event[pre_event['days_to_event'] <= -2]['skew'].mean()
        pre_late_skew = pre_event[pre_event['days_to_event'] >= -1]['skew'].mean()
        
        if not np.isnan(pre_early_skew) and not np.isnan(pre_late_skew):
            skew_steepening = pre_late_skew - pre_early_skew
            patterns['skew_steepening'] = skew_steepening
            patterns['skew_steepening_detected'] = skew_steepening > 0.02  # Skew increases
            patterns['pre_early_skew'] = pre_early_skew
            patterns['pre_late_skew'] = pre_late_skew
    
    # Pattern 4: Skew normalization (post-event)
    if len(event_day) > 0 and len(post_event) >= 2 and 'skew' in metrics_df.columns:
        event_skew = event_day['skew'].mean()
        post_2d = post_event[post_event['days_to_event'] >= 2]['skew'].mean()
        
        if not np.isnan(event_skew) and not np.isnan(post_2d):
            skew_normalization = event_skew - post_2d
            patterns['skew_normalization'] = skew_normalization
            patterns['skew_normalization_detected'] = abs(skew_normalization) > 0.02
            patterns['event_day_skew'] = event_skew
            patterns['post_2d_skew'] = post_2d
    
    # Pattern 5: Curvature increase (smile deepening pre-event)
    if len(pre_event) >= 2 and 'curvature' in pre_event.columns:
        pre_early_curv = pre_event[pre_event['days_to_event'] <= -2]['curvature'].mean()
        pre_late_curv = pre_event[pre_event['days_to_event'] >= -1]['curvature'].mean()
        
        if not np.isnan(pre_early_curv) and not np.isnan(pre_late_curv):
            curv_increase = pre_late_curv - pre_early_curv
            patterns['curvature_increase'] = curv_increase
            patterns['curvature_increase_detected'] = curv_increase > 0.01
            patterns['pre_early_curvature'] = pre_early_curv
            patterns['pre_late_curvature'] = pre_late_curv
    
    # Pattern 6: Curvature decrease (smile flattening post-event)
    if len(event_day) > 0 and len(post_event) >= 2 and 'curvature' in metrics_df.columns:
        event_curv = event_day['curvature'].mean()
        post_2d = post_event[post_event['days_to_event'] >= 2]['curvature'].mean()
        
        if not np.isnan(event_curv) and not np.isnan(post_2d):
            curv_decrease = event_curv - post_2d
            patterns['curvature_decrease'] = curv_decrease
            patterns['curvature_decrease_detected'] = curv_decrease > 0.01
            patterns['event_day_curvature'] = event_curv
            patterns['post_2d_curvature'] = post_2d
    
    # Aggregate pattern signature
    detected_patterns = [k for k, v in patterns.items() if k.endswith('_detected') and v]
    patterns['pattern_count'] = len(detected_patterns)
    patterns['pattern_signature'] = ', '.join([p.replace('_detected', '') for p in detected_patterns])
    
    return patterns


# ======================================================
# 2. Pattern classification
# ======================================================

def classify_event_pattern(patterns):
    """
    Classify an event's pattern into human-readable categories.
    
    Returns:
      String classification (e.g., "High Buildup + Sharp Collapse")
    """
    classification_parts = []
    
    # ATM patterns
    buildup = patterns.get('atm_buildup_pct', 0)
    collapse = patterns.get('atm_collapse_pct', 0)
    
    if buildup > 15:
        classification_parts.append("High Buildup")
    elif buildup > 7:
        classification_parts.append("Moderate Buildup")
    elif buildup > 3:
        classification_parts.append("Mild Buildup")
    
    if collapse > 25:
        classification_parts.append("Sharp Collapse")
    elif collapse > 15:
        classification_parts.append("Moderate Collapse")
    elif collapse > 8:
        classification_parts.append("Mild Collapse")
    
    # Skew patterns
    if patterns.get('skew_steepening_detected', False):
        steepening = patterns.get('skew_steepening', 0)
        if steepening > 0.05:
            classification_parts.append("Strong Skew Steepening")
        else:
            classification_parts.append("Skew Steepening")
    
    if patterns.get('skew_normalization_detected', False):
        classification_parts.append("Skew Normalization")
    
    # Curvature patterns
    if patterns.get('curvature_increase_detected', False):
        curv_inc = patterns.get('curvature_increase', 0)
        if curv_inc > 0.03:
            classification_parts.append("Strong Smile Deepening")
        else:
            classification_parts.append("Smile Deepening")
    
    if patterns.get('curvature_decrease_detected', False):
        classification_parts.append("Smile Flattening")
    
    if not classification_parts:
        return "Minimal Pattern"
    
    return " + ".join(classification_parts)


# ======================================================
# 3. Build pattern library
# ======================================================

def build_pattern_library(earnings_metrics, fomc_metrics):
    """
    Build a comprehensive pattern library classifying typical IV responses.
    
    Returns:
      DataFrame with pattern classifications and statistics
    """
    print("\n" + "="*70)
    print("BUILDING PATTERN LIBRARY")
    print("="*70)
    
    # Identify patterns for each event type
    print("\nAnalyzing earnings patterns...")
    earnings_patterns = identify_surface_patterns(earnings_metrics, 'Earnings')
    earnings_classification = classify_event_pattern(earnings_patterns)
    
    print("Analyzing FOMC patterns...")
    fomc_patterns = identify_surface_patterns(fomc_metrics, 'FOMC')
    fomc_classification = classify_event_pattern(fomc_patterns)
    
    # Create pattern library
    library = []
    
    # Earnings patterns
    library.append({
        'Event Type': 'Earnings',
        'N Observations': earnings_patterns.get('n_observations', 0),
        'ATM Buildup (%)': earnings_patterns.get('atm_buildup_pct', np.nan),
        'ATM Collapse (%)': earnings_patterns.get('atm_collapse_pct', np.nan),
        'Skew Steepening': earnings_patterns.get('skew_steepening', np.nan),
        'Skew Normalization': earnings_patterns.get('skew_normalization', np.nan),
        'Curvature Increase': earnings_patterns.get('curvature_increase', np.nan),
        'Curvature Decrease': earnings_patterns.get('curvature_decrease', np.nan),
        'Pattern Count': earnings_patterns.get('pattern_count', 0),
        'Pattern Signature': earnings_patterns.get('pattern_signature', 'None'),
        'Classification': earnings_classification
    })
    
    # FOMC patterns
    library.append({
        'Event Type': 'FOMC',
        'N Observations': fomc_patterns.get('n_observations', 0),
        'ATM Buildup (%)': fomc_patterns.get('atm_buildup_pct', np.nan),
        'ATM Collapse (%)': fomc_patterns.get('atm_collapse_pct', np.nan),
        'Skew Steepening': fomc_patterns.get('skew_steepening', np.nan),
        'Skew Normalization': fomc_patterns.get('skew_normalization', np.nan),
        'Curvature Increase': fomc_patterns.get('curvature_increase', np.nan),
        'Curvature Decrease': fomc_patterns.get('curvature_decrease', np.nan),
        'Pattern Count': fomc_patterns.get('pattern_count', 0),
        'Pattern Signature': fomc_patterns.get('pattern_signature', 'None'),
        'Classification': fomc_classification
    })
    
    library_df = pd.DataFrame(library)
    
    print("\nPattern Library Summary:")
    print(library_df[['Event Type', 'Classification', 'Pattern Count', 'ATM Buildup (%)', 'ATM Collapse (%)']].to_string(index=False))
    
    return library_df, earnings_patterns, fomc_patterns


# ======================================================
# 4. Visualization functions
# ======================================================

def plot_pattern_signatures(earnings_metrics, fomc_metrics, earnings_patterns, fomc_patterns):
    """
    Visualize the identified patterns for each event type.
    6-panel plot showing ATM IV, Skew, and Curvature for both event types.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('IV Surface Pattern Signatures', fontsize=16, fontweight='bold', y=0.995)
    
    # Row 1: Earnings patterns
    # ATM IV trajectory
    ax = axes[0, 0]
    if 'days_to_event' in earnings_metrics.columns and 'atm_iv' in earnings_metrics.columns:
        earn_grouped = earnings_metrics.groupby('days_to_event')['atm_iv'].agg(['mean', 'std', 'count'])
        earn_grouped = earn_grouped[earn_grouped['count'] >= 3]
        
        if len(earn_grouped) > 0:
            ax.plot(earn_grouped.index, earn_grouped['mean'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
            ax.fill_between(earn_grouped.index, 
                            earn_grouped['mean'] - earn_grouped['std'],
                            earn_grouped['mean'] + earn_grouped['std'],
                            alpha=0.3, color='#2E86AB')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2, label='Event')
            ax.axvspan(-3, 0, alpha=0.1, color='#06A77D', label='Pre-event')
            ax.axvspan(0, 3, alpha=0.1, color='#F18F01', label='Post-event')
    
    buildup = earnings_patterns.get('atm_buildup_pct', np.nan)
    collapse = earnings_patterns.get('atm_collapse_pct', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('ATM IV', fontsize=11)
    ax.set_title(f'Earnings: ATM IV Dynamics\nBuildup: {buildup:.1f}% | Collapse: {collapse:.1f}%', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Skew trajectory
    ax = axes[0, 1]
    if 'skew' in earnings_metrics.columns:
        earn_skew = earnings_metrics.groupby('days_to_event')['skew'].agg(['mean', 'std', 'count'])
        earn_skew = earn_skew[earn_skew['count'] >= 3]
        
        if len(earn_skew) > 0:
            ax.plot(earn_skew.index, earn_skew['mean'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
            ax.fill_between(earn_skew.index,
                            earn_skew['mean'] - earn_skew['std'],
                            earn_skew['mean'] + earn_skew['std'],
                            alpha=0.3, color='#2E86AB')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    steep = earnings_patterns.get('skew_steepening', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('Skew (Put - Call)', fontsize=11)
    ax.set_title(f'Earnings: Skew Evolution\nSteepening: {steep:.4f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Curvature trajectory
    ax = axes[0, 2]
    if 'curvature' in earnings_metrics.columns:
        earn_curv = earnings_metrics.groupby('days_to_event')['curvature'].agg(['mean', 'std', 'count'])
        earn_curv = earn_curv[earn_curv['count'] >= 3]
        
        if len(earn_curv) > 0:
            ax.plot(earn_curv.index, earn_curv['mean'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
            ax.fill_between(earn_curv.index,
                            earn_curv['mean'] - earn_curv['std'],
                            earn_curv['mean'] + earn_curv['std'],
                            alpha=0.3, color='#2E86AB')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    curv_inc = earnings_patterns.get('curvature_increase', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('Curvature (Wing - ATM)', fontsize=11)
    ax.set_title(f'Earnings: Smile Curvature\nIncrease: {curv_inc:.4f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Row 2: FOMC patterns
    # ATM IV trajectory
    ax = axes[1, 0]
    if 'days_to_event' in fomc_metrics.columns and 'atm_iv' in fomc_metrics.columns:
        fomc_grouped = fomc_metrics.groupby('days_to_event')['atm_iv'].agg(['mean', 'std', 'count'])
        fomc_grouped = fomc_grouped[fomc_grouped['count'] >= 3]
        
        if len(fomc_grouped) > 0:
            ax.plot(fomc_grouped.index, fomc_grouped['mean'], 'o-', linewidth=2, markersize=6, color='#F18F01')
            ax.fill_between(fomc_grouped.index,
                            fomc_grouped['mean'] - fomc_grouped['std'],
                            fomc_grouped['mean'] + fomc_grouped['std'],
                            alpha=0.3, color='#F18F01')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2, label='Event')
            ax.axvspan(-3, 0, alpha=0.1, color='#06A77D', label='Pre-event')
            ax.axvspan(0, 3, alpha=0.1, color='#F18F01', label='Post-event')
    
    buildup = fomc_patterns.get('atm_buildup_pct', np.nan)
    collapse = fomc_patterns.get('atm_collapse_pct', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('ATM IV', fontsize=11)
    ax.set_title(f'FOMC: ATM IV Dynamics\nBuildup: {buildup:.1f}% | Collapse: {collapse:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Skew trajectory
    ax = axes[1, 1]
    if 'skew' in fomc_metrics.columns:
        fomc_skew = fomc_metrics.groupby('days_to_event')['skew'].agg(['mean', 'std', 'count'])
        fomc_skew = fomc_skew[fomc_skew['count'] >= 3]
        
        if len(fomc_skew) > 0:
            ax.plot(fomc_skew.index, fomc_skew['mean'], 'o-', linewidth=2, markersize=6, color='#F18F01')
            ax.fill_between(fomc_skew.index,
                            fomc_skew['mean'] - fomc_skew['std'],
                            fomc_skew['mean'] + fomc_skew['std'],
                            alpha=0.3, color='#F18F01')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    steep = fomc_patterns.get('skew_steepening', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('Skew (Put - Call)', fontsize=11)
    ax.set_title(f'FOMC: Skew Evolution\nSteepening: {steep:.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Curvature trajectory
    ax = axes[1, 2]
    if 'curvature' in fomc_metrics.columns:
        fomc_curv = fomc_metrics.groupby('days_to_event')['curvature'].agg(['mean', 'std', 'count'])
        fomc_curv = fomc_curv[fomc_curv['count'] >= 3]
        
        if len(fomc_curv) > 0:
            ax.plot(fomc_curv.index, fomc_curv['mean'], 'o-', linewidth=2, markersize=6, color='#F18F01')
            ax.fill_between(fomc_curv.index,
                            fomc_curv['mean'] - fomc_curv['std'],
                            fomc_curv['mean'] + fomc_curv['std'],
                            alpha=0.3, color='#F18F01')
            ax.axvline(0, color='#C1292E', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    curv_inc = fomc_patterns.get('curvature_increase', np.nan)
    ax.set_xlabel('Days from Event', fontsize=11)
    ax.set_ylabel('Curvature (Wing - ATM)', fontsize=11)
    ax.set_title(f'FOMC: Smile Curvature\nIncrease: {curv_inc:.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pattern_comparison(library_df):
    """
    Create bar charts comparing pattern magnitudes across event types.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Pattern Magnitude Comparison: Earnings vs FOMC', fontsize=14, fontweight='bold')
    
    colors = ['#2E86AB', '#F18F01']
    
    # ATM patterns
    ax = axes[0]
    x = np.arange(len(library_df))
    width = 0.35
    
    buildup = library_df['ATM Buildup (%)'].values
    collapse = library_df['ATM Collapse (%)'].values
    
    ax.bar(x - width/2, buildup, width, label='Buildup', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, collapse, width, label='Collapse', color=colors[1], alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('ATM IV Patterns', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(library_df['Event Type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Skew patterns
    ax = axes[1]
    steep = library_df['Skew Steepening'].values
    norm = library_df['Skew Normalization'].values
    
    ax.bar(x - width/2, steep, width, label='Steepening', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, norm, width, label='Normalization', color=colors[1], alpha=0.8)
    
    ax.set_ylabel('Skew Change', fontsize=11)
    ax.set_title('Skew Patterns', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(library_df['Event Type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Curvature patterns
    ax = axes[2]
    inc = library_df['Curvature Increase'].values
    dec = library_df['Curvature Decrease'].values
    
    ax.bar(x - width/2, inc, width, label='Increase', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, dec, width, label='Decrease', color=colors[1], alpha=0.8)
    
    ax.set_ylabel('Curvature Change', fontsize=11)
    ax.set_title('Curvature Patterns', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(library_df['Event Type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ======================================================
# 5. Main pattern analysis pipeline
# ======================================================

def run_pattern_analysis(file_paths, tickers, output_prefix='pattern_analysis'):
    """
    Main pipeline for pattern identification and library building.
    
    Parameters:
      file_paths: path to CSV/Excel file, OR list of file paths
      tickers: list of tickers to analyze
      output_prefix: prefix for output files
    """
    print("="*70)
    print("IV SURFACE PATTERN ANALYSIS")
    print("="*70)
    
    # Step 1: Load option data
    print("\n1. Loading option data...")
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
    
    # Step 5: Compute IV metrics
    print("\n5. Computing IV surface metrics...")
    
    # Earnings
    earn_events = df_earn[df_earn['is_event_period']].copy()
    groupby_cols = ['date', 'ticker'] if 'ticker' in earn_events.columns else ['date']
    earn_metrics = compute_iv_surface_metrics(earn_events, groupby_cols=groupby_cols)
    
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
    
    # FOMC
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
    
    print(f"   Earnings: {len(earn_metrics)} observations")
    print(f"   FOMC: {len(fomc_metrics)} observations")
    
    # Step 6: Build pattern library
    print("\n6. Building pattern library...")
    pattern_library, earn_patterns, fomc_patterns = build_pattern_library(earn_metrics, fomc_metrics)
    
    # Step 7: Generate visualizations
    print("\n7. Generating visualizations...")
    
    # Pattern signatures
    fig1 = plot_pattern_signatures(earn_metrics, fomc_metrics, earn_patterns, fomc_patterns)
    fig1.savefig(f'{output_prefix}_signatures.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_signatures.png")
    
    # Pattern comparison
    fig2 = plot_pattern_comparison(pattern_library)
    fig2.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_comparison.png")
    
    # Step 8: Save outputs
    print("\n8. Saving outputs...")
    pattern_library.to_csv(f'{output_prefix}_library.csv', index=False)
    print(f"   Saved: {output_prefix}_library.csv")
    
    # Save detailed pattern dictionaries as JSON-like format
    pattern_details = pd.DataFrame([
        {'Event Type': 'Earnings', **earn_patterns},
        {'Event Type': 'FOMC', **fomc_patterns}
    ])
    pattern_details.to_csv(f'{output_prefix}_details.csv', index=False)
    print(f"   Saved: {output_prefix}_details.csv")
    
    print("\n" + "="*70)
    print("PATTERN ANALYSIS COMPLETE")
    print("="*70)
    
    return {
        'pattern_library': pattern_library,
        'earnings_patterns': earn_patterns,
        'fomc_patterns': fomc_patterns,
        'earnings_metrics': earn_metrics,
        'fomc_metrics': fomc_metrics
    }

if __name__ == "__main__":
    
    # Load files from Data folder
    data_file = glob.glob("Data/*.csv")
    data_file = sorted(data_file)
    
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
    
    # Run pattern analysis
    results = run_pattern_analysis(
        file_paths=data_file,
        tickers=tickers,
        output_prefix='iv_patterns'
    )
    
    # Display results
    print("\n" + "="*70)
    print("PATTERN LIBRARY")
    print("="*70)
    print(results['pattern_library'].to_string(index=False))
    
    print("\n" + "="*70)
    print("EARNINGS CLASSIFICATION")
    print("="*70)
    print(f"Classification: {results['pattern_library'].loc[0, 'Classification']}")
    print(f"Signature: {results['pattern_library'].loc[0, 'Pattern Signature']}")
    
    print("\n" + "="*70)
    print("FOMC CLASSIFICATION")
    print("="*70)
    print(f"Classification: {results['pattern_library'].loc[1, 'Classification']}")
    print(f"Signature: {results['pattern_library'].loc[1, 'Pattern Signature']}")