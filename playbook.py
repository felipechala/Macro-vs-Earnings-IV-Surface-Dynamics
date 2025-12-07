import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from reversion import (
    load_option_data_from_file,
    collect_earnings_dates,
    tag_event_windows,
    compute_iv_surface_metrics
)
from FOMCdates import get_fomc_dates
from patterns import identify_surface_patterns
from backtests import calculate_metrics


# ======================================================
# 1. Pattern-based structure selection rules
# ======================================================

def recommend_structure(patterns_dict):
    """
    Recommend optimal option structure based on detected patterns.
    
    Decision tree:
      - High ATM buildup + expected collapse → Butterfly
      - Skew steepening + curvature increase → Butterfly (wings benefit)
      - Moderate buildup + time decay dominates → Calendar
      - High dispersion (stock vs index vol divergence) → Dispersion
    
    Returns:
      Dictionary with recommendations and rationale
    """
    recommendations = []
    
    atm_buildup = patterns_dict.get('atm_buildup_pct', 0)
    atm_collapse = patterns_dict.get('atm_collapse_pct', 0)
    skew_steep = patterns_dict.get('skew_steepening', 0)
    curv_increase = patterns_dict.get('curvature_increase', 0)
    
    # Rule 1: High IV buildup + collapse → Butterfly (primary)
    if atm_buildup > 10 and atm_collapse > 15:
        recommendations.append({
            'structure': 'Butterfly',
            'priority': 'Primary',
            'rationale': f'High ATM buildup ({atm_buildup:.1f}%) + sharp collapse ({atm_collapse:.1f}%)',
            'timing': 'Pre-positioning (T-1 → T+1)',
            'confidence': 'High'
        })
    
    # Rule 2: Skew steepening + smile deepening → Butterfly (wings)
    if abs(skew_steep) > 0.03 or curv_increase > 0.02:
        recommendations.append({
            'structure': 'Butterfly',
            'priority': 'Primary',
            'rationale': f'Skew steepening ({skew_steep:.3f}) or smile deepening ({curv_increase:.3f})',
            'timing': 'Pre-positioning (T-1 → T+1)',
            'confidence': 'Medium'
        })
    
    # Rule 3: Moderate buildup + time decay → Calendar
    if 5 < atm_buildup < 12:
        recommendations.append({
            'structure': 'Calendar',
            'priority': 'Secondary',
            'rationale': f'Moderate buildup ({atm_buildup:.1f}%) - time decay beneficial',
            'timing': 'Pre-positioning (T-1 → T+1)',
            'confidence': 'Medium'
        })
    
    # Rule 4: Weak patterns → Post-event reversion
    if atm_buildup < 5:
        recommendations.append({
            'structure': 'Butterfly or Calendar',
            'priority': 'Tertiary',
            'rationale': f'Weak pre-event buildup ({atm_buildup:.1f}%) - prefer post-event mean reversion',
            'timing': 'Post-event (T+1 → T+3)',
            'confidence': 'Low'
        })
    
    # Rule 5: High curvature post-event → Reversion trade
    curv_decrease = patterns_dict.get('curvature_decrease', 0)
    if curv_decrease > 0.015:
        recommendations.append({
            'structure': 'Butterfly',
            'priority': 'Secondary',
            'rationale': f'Smile flattening post-event ({curv_decrease:.3f}) - reversion opportunity',
            'timing': 'Post-event (T+1 → T+3)',
            'confidence': 'Medium'
        })
    
    return recommendations


def generate_playbook_rules(earnings_patterns, fomc_patterns):
    """
    Generate comprehensive trading rules based on pattern analysis.
    """
    playbook = {
        'earnings': {},
        'fomc': {}
    }
    
    # Earnings rules
    playbook['earnings']['patterns'] = earnings_patterns
    playbook['earnings']['recommendations'] = recommend_structure(earnings_patterns)
    
    # FOMC rules
    playbook['fomc']['patterns'] = fomc_patterns
    playbook['fomc']['recommendations'] = recommend_structure(fomc_patterns)
    
    # General principles
    playbook['general_principles'] = [
        {
            'principle': 'Earnings: Butterflies dominate',
            'rule': 'Use butterflies for earnings due to sharp IV collapse patterns',
            'evidence': f"Avg collapse: {earnings_patterns.get('atm_collapse_pct', 0):.1f}%"
        },
        {
            'principle': 'FOMC: Moderate patterns',
            'rule': 'FOMC shows less dramatic IV moves - consider calendars',
            'evidence': f"Avg buildup: {fomc_patterns.get('atm_buildup_pct', 0):.1f}%"
        },
        {
            'principle': 'Pre-positioning works best',
            'rule': 'T-1 entry captures majority of IV buildup and collapse',
            'evidence': 'Pattern analysis shows peak IV at T-0'
        },
        {
            'principle': 'Skew steepening = opportunity',
            'rule': 'When skew steepens pre-event, wings become attractive',
            'evidence': f"Earnings skew steep: {earnings_patterns.get('skew_steepening', 0):.3f}"
        }
    ]
    
    return playbook


# ======================================================
# 2. Reversion regime diagnostics
# ======================================================

def identify_reversion_regime(metrics_df):
    """
    Identify the mean reversion regime based on IV dynamics.
    
    Regimes:
      - Fast reversion: Half-life < 1 day (sharp snap-back)
      - Normal reversion: Half-life 1-2 days (typical)
      - Slow reversion: Half-life > 2 days (sticky IV)
      - No reversion: IV continues trending
    
    Returns:
      Regime classification and diagnostics
    """
    if 'days_to_event' not in metrics_df.columns or 'atm_iv' not in metrics_df.columns:
        return {'regime': 'Unknown', 'diagnostics': {}}
    
    # Get post-event data
    post = metrics_df[metrics_df['days_to_event'] > 0].copy()
    
    if len(post) < 3:
        return {'regime': 'Insufficient Data', 'diagnostics': {}}
    
    # Calculate day-over-day IV changes
    post_sorted = post.sort_values('days_to_event')
    daily_avg = post_sorted.groupby('days_to_event')['atm_iv'].mean()
    
    if len(daily_avg) < 2:
        return {'regime': 'Insufficient Data', 'diagnostics': {}}
    
    # Estimate half-life from exponential decay
    days = daily_avg.index.values
    ivs = daily_avg.values
    
    # Simple exponential fit: IV(t) = IV(0) * exp(-t/tau)
    # Half-life = tau * ln(2)
    
    if ivs[0] <= 0 or len(ivs) < 2:
        return {'regime': 'Invalid Data', 'diagnostics': {}}
    
    # Linear regression on log(IV) vs t
    log_ivs = np.log(ivs / ivs[0])  # Normalized
    
    try:
        # Fit line: log(IV) = -t/tau
        slope = np.polyfit(days, log_ivs, 1)[0]
        
        if slope >= 0:
            regime = 'No Reversion'
            half_life = np.inf
        else:
            tau = -1 / slope
            half_life = tau * np.log(2)
            
            if half_life < 1:
                regime = 'Fast Reversion'
            elif half_life < 2:
                regime = 'Normal Reversion'
            else:
                regime = 'Slow Reversion'
    except:
        regime = 'Uncertain'
        half_life = np.nan
    
    # Additional diagnostics
    iv_at_1d = daily_avg.get(1, np.nan)
    iv_at_3d = daily_avg.get(3, np.nan)
    
    pct_reversion_1d = (ivs[0] - iv_at_1d) / ivs[0] * 100 if not np.isnan(iv_at_1d) and ivs[0] > 0 else np.nan
    pct_reversion_3d = (ivs[0] - iv_at_3d) / ivs[0] * 100 if not np.isnan(iv_at_3d) and ivs[0] > 0 else np.nan
    
    diagnostics = {
        'half_life_days': half_life,
        'iv_peak': ivs[0],
        'iv_at_1d': iv_at_1d,
        'iv_at_3d': iv_at_3d,
        'pct_reversion_1d': pct_reversion_1d,
        'pct_reversion_3d': pct_reversion_3d,
        'slope': slope if 'slope' in locals() else np.nan
    }
    
    return {
        'regime': regime,
        'diagnostics': diagnostics
    }


def regime_based_recommendations(regime_info):
    """
    Provide trading recommendations based on reversion regime.
    """
    regime = regime_info['regime']
    diagnostics = regime_info.get('diagnostics', {})
    half_life = diagnostics.get('half_life_days', np.nan)
    
    recommendations = []
    
    if regime == 'Fast Reversion':
        recommendations.append({
            'action': 'Aggressive post-event entry',
            'rationale': f'Fast mean reversion (t½={half_life:.1f}d) - quick profits',
            'structure': 'Butterfly (short vol)',
            'timing': 'T+1 entry, T+2 exit',
            'risk': 'Low - quick round trip'
        })
    
    elif regime == 'Normal Reversion':
        recommendations.append({
            'action': 'Standard post-event trade',
            'rationale': f'Normal reversion (t½={half_life:.1f}d)',
            'structure': 'Butterfly or Calendar',
            'timing': 'T+1 entry, T+3 exit',
            'risk': 'Medium - standard holding period'
        })
    
    elif regime == 'Slow Reversion':
        recommendations.append({
            'action': 'Extended holding or avoid',
            'rationale': f'Slow reversion (t½={half_life:.1f}d) - long grind',
            'structure': 'Calendar (benefits from extended decay)',
            'timing': 'T+1 entry, T+5+ exit',
            'risk': 'High - extended market exposure'
        })
    
    elif regime == 'No Reversion':
        recommendations.append({
            'action': 'Avoid post-event trades',
            'rationale': 'IV not reverting - trending regime',
            'structure': 'N/A',
            'timing': 'Wait for pattern change',
            'risk': 'Very High - no edge'
        })
    
    return recommendations


# ======================================================
# 3. Trade setup checklist
# ======================================================

def generate_trade_checklist(event_type, patterns, regime_info, days_to_event):
    """
    Generate pre-trade checklist based on current conditions.
    
    Returns:
      Checklist with go/no-go signals
    """
    checklist = {
        'event_type': event_type,
        'days_to_event': days_to_event,
        'timestamp': pd.Timestamp.now(),
        'signals': []
    }
    
    # Signal 1: Pattern strength
    atm_buildup = patterns.get('atm_buildup_pct', 0)
    if atm_buildup > 10:
        checklist['signals'].append({
            'item': 'ATM IV Buildup',
            'status': 'GO',
            'value': f'{atm_buildup:.1f}%',
            'threshold': '>10%',
            'importance': 'High'
        })
    elif atm_buildup > 5:
        checklist['signals'].append({
            'item': 'ATM IV Buildup',
            'status': 'CAUTION',
            'value': f'{atm_buildup:.1f}%',
            'threshold': '5-10%',
            'importance': 'High'
        })
    else:
        checklist['signals'].append({
            'item': 'ATM IV Buildup',
            'status': 'NO-GO',
            'value': f'{atm_buildup:.1f}%',
            'threshold': '<5%',
            'importance': 'High'
        })
    
    # Signal 2: Skew dynamics
    skew = patterns.get('skew_steepening', 0)
    if abs(skew) > 0.03:
        checklist['signals'].append({
            'item': 'Skew Steepening',
            'status': 'GO',
            'value': f'{skew:.3f}',
            'threshold': '>0.03',
            'importance': 'Medium'
        })
    else:
        checklist['signals'].append({
            'item': 'Skew Steepening',
            'status': 'NEUTRAL',
            'value': f'{skew:.3f}',
            'threshold': '<0.03',
            'importance': 'Medium'
        })
    
    # Signal 3: Reversion regime (for post-event)
    if days_to_event > 0:
        regime = regime_info.get('regime', 'Unknown')
        if regime in ['Fast Reversion', 'Normal Reversion']:
            checklist['signals'].append({
                'item': 'Reversion Regime',
                'status': 'GO',
                'value': regime,
                'threshold': 'Fast or Normal',
                'importance': 'High'
            })
        elif regime == 'Slow Reversion':
            checklist['signals'].append({
                'item': 'Reversion Regime',
                'status': 'CAUTION',
                'value': regime,
                'threshold': 'Slow',
                'importance': 'High'
            })
        else:
            checklist['signals'].append({
                'item': 'Reversion Regime',
                'status': 'NO-GO',
                'value': regime,
                'threshold': 'No reversion',
                'importance': 'High'
            })
    
    # Signal 4: Timing
    if days_to_event == -1:
        checklist['signals'].append({
            'item': 'Entry Timing',
            'status': 'GO',
            'value': 'T-1 (optimal pre-positioning)',
            'threshold': 'T-1',
            'importance': 'High'
        })
    elif days_to_event == 1:
        checklist['signals'].append({
            'item': 'Entry Timing',
            'status': 'GO',
            'value': 'T+1 (optimal post-event)',
            'threshold': 'T+1',
            'importance': 'High'
        })
    else:
        checklist['signals'].append({
            'item': 'Entry Timing',
            'status': 'CAUTION',
            'value': f'T{days_to_event:+d} (non-optimal)',
            'threshold': 'T-1 or T+1',
            'importance': 'Medium'
        })
    
    # Overall decision
    go_count = sum(1 for s in checklist['signals'] if s['status'] == 'GO' and s['importance'] == 'High')
    nogo_count = sum(1 for s in checklist['signals'] if s['status'] == 'NO-GO' and s['importance'] == 'High')
    
    if go_count >= 2 and nogo_count == 0:
        checklist['overall_decision'] = 'EXECUTE TRADE'
    elif nogo_count > 0:
        checklist['overall_decision'] = 'DO NOT TRADE'
    else:
        checklist['overall_decision'] = 'DISCRETIONARY'
    
    return checklist


# ======================================================
# 4. Playbook visualization
# ======================================================

def plot_decision_tree(playbook):
    """
    Visualize the decision tree for structure selection.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Event-Driven Option Strategy Decision Tree', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Event type split
    ax.text(0.5, 0.85, 'Event Type?', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2),
            fontsize=12, fontweight='bold')
    
    # Earnings branch
    ax.arrow(0.5, 0.82, -0.15, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.text(0.25, 0.72, 'EARNINGS', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#2E86AB', edgecolor='black', linewidth=2),
            fontsize=11, fontweight='bold', color='white')
    
    # FOMC branch
    ax.arrow(0.5, 0.82, 0.15, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.text(0.75, 0.72, 'FOMC', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#F18F01', edgecolor='black', linewidth=2),
            fontsize=11, fontweight='bold', color='white')
    
    # Earnings path
    ax.text(0.25, 0.62, 'ATM Buildup > 10%\n+ Collapse > 15%?', ha='center', va='center',
            fontsize=9)
    ax.arrow(0.25, 0.59, -0.08, -0.08, head_width=0.015, head_length=0.015, fc='green', ec='green')
    ax.text(0.12, 0.48, 'YES → Butterfly\n(Pre: T-1→T+1)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    ax.arrow(0.25, 0.59, 0.08, -0.08, head_width=0.015, head_length=0.015, fc='orange', ec='orange')
    ax.text(0.38, 0.48, 'NO → Calendar\n(Pre: T-1→T+1)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    # FOMC path
    ax.text(0.75, 0.62, 'Skew Steepening\n> 0.03?', ha='center', va='center',
            fontsize=9)
    ax.arrow(0.75, 0.59, -0.08, -0.08, head_width=0.015, head_length=0.015, fc='green', ec='green')
    ax.text(0.62, 0.48, 'YES → Butterfly\n(Pre: T-1→T+1)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    ax.arrow(0.75, 0.59, 0.08, -0.08, head_width=0.015, head_length=0.015, fc='orange', ec='orange')
    ax.text(0.88, 0.48, 'NO → Calendar\n(Moderate edge)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    # Post-event reversion
    ax.text(0.5, 0.35, 'Post-Event Reversion Check', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2),
            fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.25, 'Reversion Regime?', ha='center', va='center', fontsize=10)
    
    ax.arrow(0.5, 0.22, -0.15, -0.08, head_width=0.015, head_length=0.015, fc='green', ec='green')
    ax.text(0.27, 0.12, 'Fast/Normal\n→ Butterfly (T+1→T+3)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    ax.arrow(0.5, 0.22, 0, -0.08, head_width=0.015, head_length=0.015, fc='orange', ec='orange')
    ax.text(0.5, 0.12, 'Slow\n→ Calendar (Extended)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    ax.arrow(0.5, 0.22, 0.15, -0.08, head_width=0.015, head_length=0.015, fc='red', ec='red')
    ax.text(0.73, 0.12, 'No Reversion\n→ AVOID', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFB3B3', edgecolor='black'),
            fontsize=9, fontweight='bold')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_regime_diagnostics(earnings_regime, fomc_regime):
    """
    Visualize reversion regime characteristics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Earnings regime
    ax = axes[0]
    earn_diag = earnings_regime.get('diagnostics', {})
    
    regime_text = f"Regime: {earnings_regime.get('regime', 'Unknown')}\n"
    regime_text += f"Half-life: {earn_diag.get('half_life_days', np.nan):.2f} days\n"
    regime_text += f"1d Reversion: {earn_diag.get('pct_reversion_1d', np.nan):.1f}%\n"
    regime_text += f"3d Reversion: {earn_diag.get('pct_reversion_3d', np.nan):.1f}%"
    
    ax.text(0.5, 0.5, regime_text, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.set_title('Earnings Reversion Regime', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # FOMC regime
    ax = axes[1]
    fomc_diag = fomc_regime.get('diagnostics', {})
    
    regime_text = f"Regime: {fomc_regime.get('regime', 'Unknown')}\n"
    regime_text += f"Half-life: {fomc_diag.get('half_life_days', np.nan):.2f} days\n"
    regime_text += f"1d Reversion: {fomc_diag.get('pct_reversion_1d', np.nan):.1f}%\n"
    regime_text += f"3d Reversion: {fomc_diag.get('pct_reversion_3d', np.nan):.1f}%"
    
    ax.text(0.5, 0.5, regime_text, ha='center', va='center',
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.set_title('FOMC Reversion Regime', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


# ======================================================
# 5. Main playbook generation
# ======================================================

def generate_playbook(file_paths, tickers, output_prefix='playbook'):
    """
    Generate complete practitioner's playbook.
    """
    print("="*70)
    print("GENERATING PRACTITIONER'S PLAYBOOK")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    option_data = load_option_data_from_file(file_paths, tickers=tickers)
    
    # Collect events
    print("\n2. Collecting events...")
    single_stocks = [t for t in tickers if t not in ['QQQ', 'IWM', 'SPY', 'SPX']]
    earnings_df = collect_earnings_dates(single_stocks if single_stocks else tickers,
                                         start_date='2023-01-01')
    fomc_df = get_fomc_dates()
    fomc_dates = fomc_df['date'].values
    
    # Tag and compute metrics
    print("\n3. Computing IV metrics...")
    df_earn = tag_event_windows(option_data.copy(), earnings_df, 'earnings', window_days=5)
    df_fomc = tag_event_windows(option_data.copy(), fomc_dates, 'fomc', window_days=5)
    
    earn_events = df_earn[df_earn['is_event_period']].copy()
    fomc_events = df_fomc[df_fomc['is_event_period']].copy()
    
    groupby_cols = ['date', 'ticker'] if 'ticker' in earn_events.columns else ['date']
    earn_metrics = compute_iv_surface_metrics(earn_events, groupby_cols=groupby_cols)
    fomc_metrics = compute_iv_surface_metrics(fomc_events, groupby_cols=groupby_cols)
    
    # Merge event info
    if 'ticker' in earn_events.columns:
        earn_info = earn_events.groupby(['date', 'ticker']).agg({'days_to_event': 'first'}).reset_index()
        earn_metrics = earn_metrics.merge(earn_info, on=['date', 'ticker'], how='left')
        fomc_info = fomc_events.groupby(['date', 'ticker']).agg({'days_to_event': 'first'}).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_info, on=['date', 'ticker'], how='left')
    else:
        earn_info = earn_events.groupby('date').agg({'days_to_event': 'first'}).reset_index()
        earn_metrics = earn_metrics.merge(earn_info, on='date', how='left')
        fomc_info = fomc_events.groupby('date').agg({'days_to_event': 'first'}).reset_index()
        fomc_metrics = fomc_metrics.merge(fomc_info, on='date', how='left')
    
    # Identify patterns
    print("\n4. Identifying patterns...")
    earn_patterns = identify_surface_patterns(earn_metrics, 'Earnings')
    fomc_patterns = identify_surface_patterns(fomc_metrics, 'FOMC')
    
    # Identify reversion regimes
    print("\n5. Analyzing reversion regimes...")
    earn_regime = identify_reversion_regime(earn_metrics)
    fomc_regime = identify_reversion_regime(fomc_metrics)
    
    # Generate playbook
    print("\n6. Building playbook...")
    playbook = generate_playbook_rules(earn_patterns, fomc_patterns)
    
    # Add regime info
    playbook['earnings']['regime'] = earn_regime
    playbook['fomc']['regime'] = fomc_regime
    
    # Generate recommendations
    playbook['earnings']['regime_recommendations'] = regime_based_recommendations(earn_regime)
    playbook['fomc']['regime_recommendations'] = regime_based_recommendations(fomc_regime)
    
    # Generate visualizations
    print("\n7. Creating visualizations...")
    
    fig1 = plot_decision_tree(playbook)
    fig1.savefig(f'{output_prefix}_decision_tree.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_decision_tree.png")
    
    fig2 = plot_regime_diagnostics(earn_regime, fomc_regime)
    fig2.savefig(f'{output_prefix}_regime_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_prefix}_regime_diagnostics.png")
    
    # Save playbook
    print("\n8. Saving playbook...")
    
    # Create summary DataFrame
    summary_data = []
    
    for event_type in ['earnings', 'fomc']:
        event_data = playbook[event_type]
        patterns = event_data['patterns']
        regime = event_data['regime']
        
        summary_data.append({
            'Event Type': event_type.title(),
            'ATM Buildup (%)': patterns.get('atm_buildup_pct', np.nan),
            'ATM Collapse (%)': patterns.get('atm_collapse_pct', np.nan),
            'Skew Steepening': patterns.get('skew_steepening', np.nan),
            'Curvature Increase': patterns.get('curvature_increase', np.nan),
            'Reversion Regime': regime.get('regime', 'Unknown'),
            'Half-Life (days)': regime.get('diagnostics', {}).get('half_life_days', np.nan),
            'Primary Structure': event_data['recommendations'][0]['structure'] if event_data['recommendations'] else 'N/A',
            'Recommended Timing': event_data['recommendations'][0]['timing'] if event_data['recommendations'] else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
    print(f"   Saved: {output_prefix}_summary.csv")
    
    # Save detailed rules
    rules_output = []
    
    for principle in playbook['general_principles']:
        rules_output.append({
            'Category': 'General Principle',
            'Rule': principle['principle'],
            'Description': principle['rule'],
            'Evidence': principle['evidence']
        })
    
    for event_type in ['earnings', 'fomc']:
        for rec in playbook[event_type]['recommendations']:
            rules_output.append({
                'Category': f'{event_type.title()} Strategy',
                'Rule': rec['structure'],
                'Description': rec['rationale'],
                'Evidence': f"Priority: {rec['priority']}, Timing: {rec['timing']}"
            })
    
    rules_df = pd.DataFrame(rules_output)
    rules_df.to_csv(f'{output_prefix}_rules.csv', index=False)
    print(f"   Saved: {output_prefix}_rules.csv")
    
    print("\n" + "="*70)
    print("PLAYBOOK GENERATION COMPLETE")
    print("="*70)
    
    return playbook, summary_df, rules_df


def print_playbook_summary(playbook):
    """
    Print human-readable playbook summary.
    """
    print("\n" + "="*70)
    print("PRACTITIONER'S PLAYBOOK SUMMARY")
    print("="*70)
    
    print("\n" + "─"*70)
    print("GENERAL PRINCIPLES")
    print("─"*70)
    for principle in playbook['general_principles']:
        print(f"\n• {principle['principle']}")
        print(f"  Rule: {principle['rule']}")
        print(f"  Evidence: {principle['evidence']}")
    
    print("\n" + "─"*70)
    print("EARNINGS EVENTS")
    print("─"*70)
    
    earn_patterns = playbook['earnings']['patterns']
    print(f"\nPattern Signature:")
    print(f"  ATM Buildup:        {earn_patterns.get('atm_buildup_pct', np.nan):.1f}%")
    print(f"  ATM Collapse:       {earn_patterns.get('atm_collapse_pct', np.nan):.1f}%")
    print(f"  Skew Steepening:    {earn_patterns.get('skew_steepening', np.nan):.3f}")
    print(f"  Curvature Increase: {earn_patterns.get('curvature_increase', np.nan):.3f}")
    
    earn_regime = playbook['earnings']['regime']
    print(f"\nReversion Regime: {earn_regime.get('regime', 'Unknown')}")
    print(f"  Half-Life: {earn_regime.get('diagnostics', {}).get('half_life_days', np.nan):.2f} days")
    
    print("\nRecommended Strategies:")
    for i, rec in enumerate(playbook['earnings']['recommendations'], 1):
        print(f"\n  {i}. {rec['structure']} ({rec['priority']})")
        print(f"     Rationale: {rec['rationale']}")
        print(f"     Timing: {rec['timing']}")
        print(f"     Confidence: {rec['confidence']}")
    
    print("\n" + "─"*70)
    print("FOMC EVENTS")
    print("─"*70)
    
    fomc_patterns = playbook['fomc']['patterns']
    print(f"\nPattern Signature:")
    print(f"  ATM Buildup:        {fomc_patterns.get('atm_buildup_pct', np.nan):.1f}%")
    print(f"  ATM Collapse:       {fomc_patterns.get('atm_collapse_pct', np.nan):.1f}%")
    print(f"  Skew Steepening:    {fomc_patterns.get('skew_steepening', np.nan):.3f}")
    print(f"  Curvature Increase: {fomc_patterns.get('curvature_increase', np.nan):.3f}")
    
    fomc_regime = playbook['fomc']['regime']
    print(f"\nReversion Regime: {fomc_regime.get('regime', 'Unknown')}")
    print(f"  Half-Life: {fomc_regime.get('diagnostics', {}).get('half_life_days', np.nan):.2f} days")
    
    print("\nRecommended Strategies:")
    for i, rec in enumerate(playbook['fomc']['recommendations'], 1):
        print(f"\n  {i}. {rec['structure']} ({rec['priority']})")
        print(f"     Rationale: {rec['rationale']}")
        print(f"     Timing: {rec['timing']}")
        print(f"     Confidence: {rec['confidence']}")
    
    print("\n" + "="*70)
    print("QUICK REFERENCE GUIDE")
    print("="*70)
    
    print("\n┌─ EARNINGS EVENTS ─────────────────────────────────────────┐")
    print("│ Primary Strategy:   Butterfly (Pre-positioning)          │")
    print("│ Entry:              T-1 day                              │")
    print("│ Exit:               T+1 day                              │")
    print("│ Key Signal:         ATM IV buildup > 10%                 │")
    print("│ Expected Collapse:  15-25%                               │")
    print("└───────────────────────────────────────────────────────────┘")
    
    print("\n┌─ FOMC EVENTS ─────────────────────────────────────────────┐")
    print("│ Primary Strategy:   Calendar / Butterfly                 │")
    print("│ Entry:              T-1 day                              │")
    print("│ Exit:               T+1 day                              │")
    print("│ Key Signal:         Moderate IV buildup + skew changes   │")
    print("│ Expected Collapse:  10-18%                               │")
    print("└───────────────────────────────────────────────────────────┘")
    
    print("\n┌─ POST-EVENT REVERSION ────────────────────────────────────┐")
    print("│ When to Use:        Fast/Normal reversion regime         │")
    print("│ Structure:          Butterfly (mean reversion)           │")
    print("│ Entry:              T+1 day                              │")
    print("│ Exit:               T+3 day                              │")
    print("│ Avoid:              No reversion / Slow reversion        │")
    print("└───────────────────────────────────────────────────────────┘")


# ======================================================
# 6. Example usage
# ======================================================

if __name__ == "__main__":
    # Data files
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
    
    # Generate playbook
    playbook, summary_df, rules_df = generate_playbook(
        file_paths=data_file,
        tickers=tickers,
        output_prefix='trading_playbook'
    )
    
    # Print summary
    print_playbook_summary(playbook)
    
    # Example: Generate trade checklist for a specific scenario
    print("\n" + "="*70)
    print("EXAMPLE TRADE CHECKLIST")
    print("="*70)
    
    # Scenario: T-1 day before earnings
    print("\nScenario: T-1 day before AAPL earnings")
    checklist = generate_trade_checklist(
        event_type='Earnings',
        patterns=playbook['earnings']['patterns'],
        regime_info=playbook['earnings']['regime'],
        days_to_event=-1
    )
    
    print(f"\nOverall Decision: {checklist['overall_decision']}")
    print("\nSignal Checklist:")
    for signal in checklist['signals']:
        status_symbol = "✓" if signal['status'] == 'GO' else "⚠" if signal['status'] == 'CAUTION' else "✗"
        print(f"  {status_symbol} {signal['item']}: {signal['value']} (Importance: {signal['importance']})")
    
    print("\n" + "="*70)
    print("PLAYBOOK FILES GENERATED")
    print("="*70)
    print("\nOutput files:")
    print("  • trading_playbook_decision_tree.png")
    print("  • trading_playbook_regime_diagnostics.png")
    print("  • trading_playbook_summary.csv")
    print("  • trading_playbook_rules.csv")
    print("\nUse these files as a quick reference for live trading decisions!")