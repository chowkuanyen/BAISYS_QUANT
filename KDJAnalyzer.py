import pandas as pd


def calculate_kdj_signal_from_df(df: pd.DataFrame) -> str:
    """
    根据单只股票的DataFrame计算严格的KDJ信号。
    此函数逻辑与 SignalManager.py 中的KDJ信号逻辑完全一致。

    Args:
        df: 包含 'close', 'high', 'low' 列的股票历史数据DataFrame

    Returns:
        str: KDJ信号描述字符串，如 "极值J线反转 (K=15.2, J=-2.1)"。若无信号则返回空字符串 ""。
    """
    if len(df) < 10:
        return ""

    # 使用 pandas_ta 计算KDJ
    df = df.copy()  # 避免修改原DataFrame
    df.ta.stoch(append=True, close='close', high='high', low='low')
    kdj_cols = [col for col in df.columns if col.startswith('STOCHk_') or col.startswith('STOCHd_')]
    if len(kdj_cols) < 2:
        return ""

    k_col = kdj_cols[0]
    d_col = kdj_cols[1]
    j_col = 'KDJ_J'
    df[j_col] = 3 * df[k_col] - 2 * df[d_col]

    # 检查最新的金叉
    kdj_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # 条件1: 极值J线反转
    if prev_row[j_col] < 0 and last_row[j_col] > 5 and kdj_cross.iloc[-1]:
        return f"极值J线反转 (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"

    # 条件2: 底背离金叉
    window = 10
    curr_low = df['low'].iloc[-1]
    curr_k = df[k_col].iloc[-1]
    min_k_window = df[k_col].iloc[-window:-1].min()
    min_low_window = df['low'].iloc[-window:-1].min()
    is_divergence = (curr_low <= min_low_window * 1.02) & (curr_k > min_k_window * 1.1)
    if kdj_cross.iloc[-1] and is_divergence and last_row[k_col] < 30:
        return f"底背离金叉 (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"

    # 计算5日均线用于后续条件
    ma5 = df['close'].rolling(window=5).mean()
    above_ma5 = df['close'] > ma5

    # 检查过去5天是否有超卖
    kd_oversold = (df[k_col] < 20) & (df[d_col] < 20)
    had_oversold = kd_oversold.iloc[-5:-1].any()

    # 条件3: 趋势确认金叉
    if had_oversold and kdj_cross.iloc[-1] and above_ma5.iloc[-1]:
        return f"趋势确认金叉 (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"

    # 条件4: 低位超卖金叉
    if had_oversold and kdj_cross.iloc[-1]:
        return f"低位超卖金叉 (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"

    # 如果以上条件都不满足，则无信号
    return ""
