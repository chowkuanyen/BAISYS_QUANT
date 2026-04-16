import pandas as pd
import numpy as np
from typing import List, Dict
import pandas_ta as ta  # 确保导入pandas_ta


class MACDAnalyzer:
    def _custom_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [自定义实现] 同时计算 MACD 标准周期 (12, 26, 9) 和加速周期 (6, 13, 5) 的快慢线金叉信号。
        **改进：区分零轴上金叉和零轴下金叉。**
        """
        if 'close' not in df.columns:
            return df

        close = df['close']

        macd_periods = {
            '12269': (12, 26, 9),  # 标准中长线周期
            '6135': (6, 13, 5)  # 短线/加速周期
        }

        for name, (fast, slow, signal) in macd_periods.items():
            ema_fast_col = f'EMA_{fast}_{name}'
            ema_slow_col = f'EMA_{slow}_{name}'

            df[ema_fast_col] = close.ewm(span=fast, adjust=False).mean()
            df[ema_slow_col] = close.ewm(span=slow, adjust=False).mean()

            dif_col = f'DIF_{name}'
            df[dif_col] = df[ema_fast_col] - df[ema_slow_col]

            dea_col = f'DEA_{name}'
            df[dea_col] = df[dif_col].ewm(span=signal, adjust=False).mean()

            # 修正列名：使用正确的信号详情列名
            signal_col = f'MACD_{name}_SIGNAL_DETAIL'  # 修正：这里是关键！

            is_cross = (df[dif_col] > df[dea_col]) & \
                       (df[dif_col].shift(1).fillna(0) <= df[dea_col].shift(1).fillna(0))

            # 核心：区分零轴上/下金叉
            df[signal_col] = np.where(
                is_cross,
                np.where(
                    (df[dif_col] > 0) & (df[dea_col] > 0),
                    '零轴上金叉',
                    '零轴下金叉'
                ),
                ''
            )

            # ✅ 修复：确保列存在，避免后续 merge 失败
            cross_col = f'MACD_{name}_CROSS'
            df[cross_col] = np.where(is_cross, 1, 0)

            df.drop(columns=[ema_fast_col, ema_slow_col], inplace=True, errors='ignore')


        return df

    @staticmethod
    def _calculate_macd_momentum(df: pd.DataFrame, dif_col: str, dea_col: str) -> str:
        """
        计算 MACD 动能状态: 加速上涨/减速上涨/加速下跌/减速下跌
        """
        if len(df) < 2:
            return "N/A (数据不足)"

        # 获取最新的 DIF, DEA 值和前一天的 DIF 值
        latest_dif = df[dif_col].iloc[-1]
        latest_dea = df[dea_col].iloc[-1]
        prev_dif = df[dif_col].iloc[-2]

        # DIF 线的变化 (MACD 柱的变化方向)
        dif_change = latest_dif - prev_dif

        momentum_state = ""

        if latest_dif >= latest_dea:
            # DIF 在 DEA 之上 (多头区域/红柱)
            if dif_change > 0:
                momentum_state = "加速上涨 (红柱加长)"
            elif dif_change <= 0:
                momentum_state = "减速上涨 (红柱缩短)"
        else:
            # DIF 在 DEA 之下 (空头区域/绿柱)
            if dif_change < 0:
                momentum_state = "加速下跌 (绿柱加长)"
            elif dif_change >= 0:
                momentum_state = "减速下跌 (绿柱缩短)"

        return momentum_state


class TASignalProcessor:
    """技术指标信号处理类"""

    def __init__(self, analyzer_instance):
        self.analyzer = analyzer_instance

    def _classify_cci_level(self, cci_value: float) -> str:
        """根据CCI值分类"""
        if pd.isna(cci_value):
            return 'N/A'
        if cci_value > 200:
            return f'极度超买 ({cci_value:.2f})'
        elif cci_value >= 100:
            return f'强势超买 ({cci_value:.2f})'
        elif cci_value > -100:
            return ''
        elif cci_value >= -200:
            return f'弱势超卖 ({cci_value:.2f})'
        else:
            return f'极度超卖 ({cci_value:.2f})'

    def process_signals(self, all_codes: List[str], hist_df_all: pd.DataFrame, spot_df: pd.DataFrame) -> Dict[
        str, pd.DataFrame]:

        print(f"\n正在对 {len(all_codes)} 只股票进行技术分析...")

        # 初始化为 DataFrame，避免后续转
        ta_signals = {
            'MACD_12269': pd.DataFrame(columns=['股票代码', 'MACD_12269_Signal']),
            'MACD_6135': pd.DataFrame(columns=['股票代码', 'MACD_6135_Signal']),
            'KDJ': pd.DataFrame(columns=['股票代码', 'KDJ_Signal']),
            'CCI': pd.DataFrame(columns=['股票代码', 'CCI_Signal']),
            'RSI': pd.DataFrame(columns=['股票代码', 'RSI_Signal']),
            'BOLL': pd.DataFrame(columns=['股票代码', 'BOLL_Signal']),
            'MACD_DIF_MOMENTUM': pd.DataFrame(
                columns=['股票代码', 'MACD_12269_DIF', 'MACD_12269_动能', 'MACD_6135_DIF', 'MACD_6135_动能']),
        }

        if hist_df_all.empty:
            print("[WARN] 历史数据为空，跳过技术分析。")
            return {key: pd.DataFrame(columns=['股票代码', f'{key}_Signal']) for key in ta_signals.keys()}

        # 安全提取 code
        if 'symbol' not in hist_df_all.columns:
            print("[ERROR] K线数据中缺少 'symbol' 列！")
            return {key: pd.DataFrame(columns=['股票代码', f'{key}_Signal']) for key in ta_signals.keys()}

        # 修复：正确提取股票代码，去除交易所前缀
        symbol_str = hist_df_all['symbol'].astype(str)
        extracted_digits = symbol_str.str.extract(r'([0-9]{6})', expand=False).fillna('N/A')
        hist_df_all['股票代码'] = extracted_digits.str.zfill(6)

        if 'date' not in hist_df_all.columns and 'trade_date' in hist_df_all.columns:
            hist_df_all.rename(columns={'trade_date': 'date'}, inplace=True)

        hist_df_all.sort_values(['股票代码', 'date'], inplace=True)

        # 修复：使用纯数字代码进行过滤
        code_set = set([code[2:] if isinstance(code, str) and len(code) > 2 else code for code in all_codes])
        hist_df_all = hist_df_all[hist_df_all['股票代码'].isin(code_set)].copy()

        for code in all_codes:
            # 修复：使用纯数字代码进行匹配
            pure_code = code[2:] if isinstance(code, str) and len(code) > 2 else code
            df = hist_df_all[hist_df_all['股票代码'] == pure_code].copy()

            if df.empty or len(df) < 30:
                print(f"[DEBUG] 股票 {code} (纯代码: {pure_code}) 数据不足，跳过。当前数据长度: {len(df)}")
                continue

            for col in ['close', 'open', 'high', 'low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            if df.empty:
                print(f"[DEBUG] 股票 {code} 清洗后数据为空，跳过。")
                continue

            if 'close' not in df.columns or 'open' not in df.columns:
                print(f"[ERROR] 股票 {code}: 缺少必要的 OHLC 列，跳过。")
                continue

            print(f"[DEBUG] 正在处理股票 {code} (纯代码: {pure_code}), 数据长度: {len(df)}")

            # 调用MACD分析
            df = self._custom_macd(df)

            # 计算MACD动能
            try:
                latest_row = df.iloc[-1]
                mom_12269 = self._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
                mom_6135 = self._calculate_macd_momentum(df, 'DIF_6135', 'DEA_6135')

                # 添加动能数据
                ta_signals['MACD_DIF_MOMENTUM'] = pd.concat([
                    ta_signals['MACD_DIF_MOMENTUM'],
                    pd.DataFrame([{
                        '股票代码': pure_code,
                        'MACD_12269_DIF': latest_row.get('DIF_12269', 0),
                        'MACD_12269_动能': mom_12269,
                        'MACD_6135_DIF': latest_row.get('DIF_6135', 0),
                        'MACD_6135_动能': mom_6135,
                    }])
                ], ignore_index=True)
            except Exception as e:
                print(f"[WARN] {code} 动能计算失败: {e}")

            # MACD 12269 - 修复列名
            detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'  # 修正：使用正确的列名
            if detail_col_12269 in df.columns and df[detail_col_12269].iloc[-1] != '':
                signal_detail = df[detail_col_12269].iloc[-1]
                ta_signals['MACD_12269'] = pd.concat([
                    ta_signals['MACD_12269'],
                    pd.DataFrame([{'股票代码': pure_code, 'MACD_12269_Signal': signal_detail}])
                ], ignore_index=True)
                print(f"[DEBUG] MACD_12269 信号: {signal_detail} for {pure_code}")

            # MACD 6135 - 修复列名
            detail_col_6135 = 'MACD_6135_SIGNAL_DETAIL'  # 修正：使用正确的列名
            if detail_col_6135 in df.columns and df[detail_col_6135].iloc[-1] != '':
                signal_detail = df[detail_col_6135].iloc[-1]
                ta_signals['MACD_6135'] = pd.concat([
                    ta_signals['MACD_6135'],
                    pd.DataFrame([{'股票代码': pure_code, 'MACD_6135_Signal': signal_detail}])
                ], ignore_index=True)
                print(f"[DEBUG] MACD_6135 信号: {signal_detail} for {pure_code}")

            # KDJ 分析
            try:
                df.ta.stoch(close='close', high='high', low='low', append=True)
                kdj_cols = [col for col in df.columns if col.startswith('STOCHk_') or col.startswith('STOCHd_')]
                if len(kdj_cols) >= 2:
                    k_col = kdj_cols[0]
                    d_col = kdj_cols[1]
                    j_col = 'KDJ_J'
                    df[j_col] = 3 * df[k_col] - 2 * df[d_col]

                    kdj_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
                    j_oversold = df[j_col].shift(1).rolling(window=3).min() < 0
                    kd_oversold = (df[k_col] < 20) & (df[d_col] < 20)

                    window = 10
                    curr_low = df['low'].iloc[-1]
                    curr_k = df[k_col].iloc[-1]
                    min_k_window = df[k_col].iloc[-window:-1].min()
                    min_low_window = df['low'].iloc[-window:-1].min()
                    is_divergence = (curr_low <= min_low_window * 1.02) & (curr_k > min_k_window * 1.1)

                    ma5 = df['close'].rolling(window=5).mean()
                    above_ma5 = df['close'] > ma5

                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]

                    signal_msg = ""
                    if prev_row[j_col] < 0 and last_row[j_col] > 5 and kdj_cross.iloc[-1]:
                        signal_msg = "极值J线反转"
                    elif kdj_cross.iloc[-1] and is_divergence and last_row[k_col] < 30:
                        signal_msg = "底背离金叉"
                    elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1] and above_ma5.iloc[-1]:
                        signal_msg = "趋势确认金叉"
                    elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1]:
                        signal_msg = "低位超卖金叉"

                    if signal_msg:
                        ta_signals['KDJ'] = pd.concat([
                            ta_signals['KDJ'],
                            pd.DataFrame([{
                                '股票代码': pure_code,
                                'KDJ_Signal': f"{signal_msg} (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"
                            }])
                        ], ignore_index=True)
                        print(f"[DEBUG] KDJ 信号: {signal_msg} for {pure_code}")
            except Exception as e:
                print(f"[WARN] {code} KDJ 计算失败: {e}")

            # CCI 分析
            try:
                df.ta.cci(close='close', high='high', low='low', append=True)
                cci_cols = [col for col in df.columns if col.startswith('CCI_')]
                if cci_cols:
                    cci_col = cci_cols[0]
                    current_cci = df[cci_col].iloc[-1]
                    cci_signal = self._classify_cci_level(current_cci)
                    if not cci_signal or cci_signal == '':
                        cci_signal = f"常态波动 ({current_cci:.2f})"
                    ta_signals['CCI'] = pd.concat([
                        ta_signals['CCI'],
                        pd.DataFrame([{'股票代码': pure_code, 'CCI_Signal': cci_signal}])
                    ], ignore_index=True)
                    print(f"[DEBUG] CCI 信号: {cci_signal} for {pure_code}")
            except Exception as e:
                print(f"[WARN] {code} CCI 计算失败: {e}")

            # RSI 分析
            try:
                df.ta.rsi(close='close', length=14, append=True)
                rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
                if rsi_cols:
                    rsi_col = rsi_cols[0]
                    curr_rsi = df[rsi_col].iloc[-1]
                    window = 10
                    curr_low = df['low'].iloc[-1]
                    min_low_window = df['low'].iloc[-window:-1].min()
                    min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                    is_price_low = curr_low <= (min_low_window * 1.02)
                    is_rsi_divergence = (is_price_low) and (curr_rsi > min_rsi_window * 1.05) and (curr_rsi < 50)
                    rsi_msg = f"RSI={curr_rsi:.1f}"
                    if is_rsi_divergence:
                        rsi_msg = f"RSI底背离! ({curr_rsi:.1f})"
                    ta_signals['RSI'] = pd.concat([
                        ta_signals['RSI'],
                        pd.DataFrame([{'股票代码': pure_code, 'RSI_Signal': rsi_msg}])
                    ], ignore_index=True)
                    print(f"[DEBUG] RSI 信号: {rsi_msg} for {pure_code}")
            except Exception as e:
                print(f"[WARN] {code} RSI 计算失败: {e}")

            # BOLL 分析
            try:
                df.ta.bbands(length=20, std=2, close='close', append=True)
                boll_cols = [col for col in df.columns if col.startswith('BBL_')]
                if boll_cols:
                    lower_band = boll_cols[0]
                    upper_band = [col for col in df.columns if col.startswith('BBU_')][0]
                    middle_band = [col for col in df.columns if col.startswith('BBM_')][0]
                    df['BOLL_BANDWIDTH'] = (df[upper_band] - df[lower_band]) / df['close']
                    is_narrow = df['BOLL_BANDWIDTH'].iloc[-5:].mean() < df['BOLL_BANDWIDTH'].mean()
                    boll_msg = "低波/缩口" if is_narrow else "常态/张口"
                    ta_signals['BOLL'] = pd.concat([
                        ta_signals['BOLL'],
                        pd.DataFrame([{'股票代码': pure_code, 'BOLL_Signal': boll_msg}])
                    ], ignore_index=True)
                    print(f"[DEBUG] BOLL 信号: {boll_msg} for {pure_code}")
            except Exception as e:
                print(f"[WARN] {code} BOLL 计算失败: {e}")

        # 确保股票代码格式正确
        for key in ta_signals:
            if not ta_signals[key].empty:
                ta_signals[key]['股票代码'] = ta_signals[key]['股票代码'].astype(str).str.zfill(6)

        return ta_signals

    # 将MACD相关方法移到TASignalProcessor类中
    def _custom_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [自定义实现] 同时计算 MACD 标准周期 (12, 26, 9) 和加速周期 (6, 13, 5) 的快慢线金叉信号。
        **改进：区分零轴上金叉和零轴下金叉。**
        """
        if 'close' not in df.columns:
            return df

        close = df['close']

        macd_periods = {
            '12269': (12, 26, 9),  # 标准中长线周期
            '6135': (6, 13, 5)  # 短线/加速周期
        }

        for name, (fast, slow, signal) in macd_periods.items():
            ema_fast_col = f'EMA_{fast}_{name}'
            ema_slow_col = f'EMA_{slow}_{name}'

            df[ema_fast_col] = close.ewm(span=fast, adjust=False).mean()
            df[ema_slow_col] = close.ewm(span=slow, adjust=False).mean()

            dif_col = f'DIF_{name}'
            df[dif_col] = df[ema_fast_col] - df[ema_slow_col]

            dea_col = f'DEA_{name}'
            df[dea_col] = df[dif_col].ewm(span=signal, adjust=False).mean()

            # 修正列名：使用正确的信号详情列名
            signal_col = f'MACD_{name}_SIGNAL_DETAIL'

            is_cross = (df[dif_col] > df[dea_col]) & \
                       (df[dif_col].shift(1).fillna(0) <= df[dea_col].shift(1).fillna(0))

            # 核心：区分零轴上/下金叉
            df[signal_col] = np.where(
                is_cross,
                np.where(
                    (df[dif_col] > 0) & (df[dea_col] > 0),
                    '零轴上金叉',
                    '零轴下金叉'
                ),
                ''
            )

            # ✅ 修复：确保列存在，避免后续 merge 失败
            cross_col = f'MACD_{name}_CROSS'
            df[cross_col] = np.where(is_cross, 1, 0)

            df.drop(columns=[ema_fast_col, ema_slow_col], inplace=True, errors='ignore')
            print(f"✅ 生成信号 for {name}: {df[signal_col].iloc[-1]} (DIF={df[dif_col].iloc[-1]:.4f})")

        return df

    def _calculate_macd_momentum(self, df: pd.DataFrame, dif_col: str, dea_col: str) -> str:
        """
        计算 MACD 动能状态: 加速上涨/减速上涨/加速下跌/减速下跌
        """
        if len(df) < 2:
            return "N/A (数据不足)"

        # 获取最新的 DIF, DEA 值和前一天的 DIF 值
        latest_dif = df[dif_col].iloc[-1]
        latest_dea = df[dea_col].iloc[-1]
        prev_dif = df[dif_col].iloc[-2]

        # DIF 线的变化 (MACD 柱的变化方向)
        dif_change = latest_dif - prev_dif

        momentum_state = ""

        if latest_dif >= latest_dea:
            # DIF 在 DEA 之上 (多头区域/红柱)
            if dif_change > 0:
                momentum_state = "加速上涨 (红柱加长)"
            elif dif_change <= 0:
                momentum_state = "减速上涨 (红柱缩短)"
        else:
            # DIF 在 DEA 之下 (空头区域/绿柱)
            if dif_change < 0:
                momentum_state = "加速下跌 (绿柱加长)"
            elif dif_change >= 0:
                momentum_state = "减速下跌 (绿柱缩短)"

        return momentum_state


# 测试代码
if __name__ == "__main__":
    # 创建模拟测试数据
    test_data = pd.DataFrame({
        'symbol': ['sh000001', 'sz000002'] * 50,
        'date': pd.date_range(start='2023-01-01', periods=100),
        'close': np.random.uniform(10, 50, 100),
        'open': np.random.uniform(10, 50, 100),
        'high': np.random.uniform(10, 55, 100),
        'low': np.random.uniform(5, 45, 100),
    })

    analyzer = TASignalProcessor(None)
    signals = analyzer.process_signals(['sh000001', 'sz000002'], test_data, pd.DataFrame())

    print("测试结果:")
    for key, df in signals.items():
        print(f"{key}: {len(df)} 条记录")
        if not df.empty:
            print(df.head())
