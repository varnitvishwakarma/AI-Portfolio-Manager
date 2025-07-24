import numpy as np

def calculate_metrics(daily_returns):
    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return ann_return, ann_vol, sharpe, max_drawdown
