import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def evaluate_alpha(parquet_path="oos_predictions.parquet"):
    df = pd.read_parquet(parquet_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("=" * 50)
    print("       ALPHA ENGINE PERFORMANCE AUDIT")
    print("=" * 50)
    
    daily_ic = []
    for date, group in df.groupby('Date'):
        if len(group) > 20 and group['Pred_Alpha'].std() > 1e-6:
            ic, _ = spearmanr(group['Pred_Alpha'], group['Target_5D_Z'])
            if not np.isnan(ic):
                daily_ic.append({'Date': date, 'IC': ic})
    
    ic_df = pd.DataFrame(daily_ic).set_index('Date')
    mean_ic = ic_df['IC'].mean()
    ic_std = ic_df['IC'].std()
    icir = (mean_ic / ic_std) * np.sqrt(252)
    
    print(f"Mean Purged IC      : {mean_ic:.5f}")
    print(f"IC Volatility       : {ic_std:.5f}")
    print(f"Annualized ICIR     : {icir:.2f}")
    
    df['Decile'] = df.groupby('Date')['Pred_Alpha'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
    )
    
    decile_returns = df.groupby(['Date', 'Decile'])['Log_Ret'].mean().unstack()
    
    spread = decile_returns[10] - decile_returns[1]
    net_spread = spread - 0.0010 
    
    ann_ret = net_spread.mean() * 252
    ann_vol = net_spread.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-8)
    
    cum_ret = (1 + net_spread).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    print(f"Long/Short Spread   : {ann_ret * 100:.2f}% (Annualized Net)")
    print(f"Annualized Vol      : {ann_vol * 100:.2f}%")
    print(f"Annualized Sharpe   : {sharpe:.2f}")
    print(f"Max Drawdown        : {max_dd * 100:.2f}%")
    
    decile_means = df.groupby('Decile')['Log_Ret'].mean()
    mono_corr, _ = spearmanr(decile_means.index, decile_means.values)
    print(f"Decile Monotonicity : {mono_corr:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_alpha()