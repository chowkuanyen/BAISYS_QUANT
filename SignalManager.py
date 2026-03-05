import pandas as pd
from typing import List, Dict
from MACDAnalyzer import MACDAnalyzer

class TASignalProcessor:
    """技术指标信号处理类"""

    def __init__(self, analyzer_instance):
        # 传入 StockAnalyzer 的实例以复用配置和方法（如 format_stock_code）
        self.analyzer = analyzer_instance

    def _classify_cci_level(self, cci_value: float) -> str:
        """根据CCI值分类"""
        if pd.isna(cci_value):
            return 'N/A'
        if cci_value > 200: return f'极度超买 ({cci_value:.2f})'
        elif cci_value >= 100: return f'强势超买 ({cci_value:.2f})'
        elif cci_value > -100: return ''
        elif cci_value >= -200: return f'弱势超卖 ({cci_value:.2f})'
        else: return f'极度超卖 ({cci_value:.2f})'


    def process_signals(self, all_codes: List[str], hist_df_all: pd.DataFrame, spot_df: pd.DataFrame) -> Dict[
    str, pd.DataFrame]:

            print(f"\n正在对 {len(all_codes)} 只股票进行技术分析...")

            ta_signals = {'MACD_12269': [], 'MACD_6135': [], 'KDJ': [], 'CCI': [], 'RSI': [], 'BOLL': [],
                          'MACD_DIF_MOMENTUM': []}

            if hist_df_all.empty:
                print("[WARN] 历史数据为空，跳过技术分析。")
                # 返回包含空 DataFrame 的字典
                return {key: pd.DataFrame(columns=['股票代码', f'{key}_Signal']) for key in ta_signals.keys()}

            # 确保数据是连续的且已排序 (使用标准化的 'date' 列)
            hist_df_all.sort_values(['股票代码', 'date'], inplace=True)

            for code in all_codes:
                # df 现在应该已经包含了 'open', 'close', 'high', 'low' 等标准列名
                df = hist_df_all[hist_df_all['股票代码'] == code].copy()

                if df.empty or len(df) < 30:
                    continue

                # 确保所需列是数字类型 (使用英文列名)
                for col in ['close', 'open', 'high', 'low']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df.dropna(subset=['close'], inplace=True)
                if df.empty: continue

                # 确保关键列存在 (现在只检查英文小写)
                if 'close' not in df.columns or 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                    print(f"[ERROR] 股票 {code}: 历史数据中缺少必要的 OHLC 列，跳过 TA 计算。")
                    continue

                df = MACDAnalyzer._custom_macd(self, df)
                try:
                    latest_row = df.iloc[-1]
                    # 计算标准周期动能
                    mom_12269 = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
                    # 计算加速周期动能
                    mom_6135 = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_6135', 'DEA_6135')

                    ta_signals['MACD_DIF_MOMENTUM'].append({
                        '股票代码': code,
                        'MACD_12269_DIF': latest_row.get('DIF_12269', 0),
                        'MACD_12269_动能': mom_12269,
                        'MACD_6135_DIF': latest_row.get('DIF_6135', 0),
                        'MACD_6135_动能': mom_6135,
                    })
                except Exception as e:
                    print(f"[WARN] {code} 动能计算失败: {e}")

                # 提取 MACD 12269 信号
                detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'
                if detail_col_12269 in df.columns:
                    signal_detail = df[detail_col_12269].iloc[-1]
                    if signal_detail != '':
                        ta_signals['MACD_12269'].append({'股票代码': code, 'MACD_12269_Signal': signal_detail})

                    # 提取 MACD 6135 信号
                detail_col_6135 = 'MACD_6135_SIGNAL_DETAIL'
                if detail_col_6135 in df.columns:
                    signal_detail = df[detail_col_6135].iloc[-1]
                    if signal_detail != '':
                        ta_signals['MACD_6135'].append({'股票代码': code, 'MACD_6135_Signal': signal_detail})

                # 2. KDJ

                df.ta.stoch(append=True, close='close', high='high', low='low')
                kdj_cols = [col for col in df.columns if col.startswith('STOCHk_') or col.startswith('STOCHd_')]
                if len(kdj_cols) >= 2:
                    k_col = kdj_cols[0]
                    d_col = kdj_cols[1]
                    j_col = 'KDJ_J'
                    df[j_col] = 3 * df[k_col] - 2 * df[d_col]

                    kdj_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
                    j_oversold = df[j_col].shift(1).rolling(window=3).min() < 0
                    # 普通超卖：K和D都在20以下
                    kd_oversold = (df[k_col] < 20) & (df[d_col] < 20)

                    # 3. 定义高级信号：底背离 (Divergence)
                    window = 10
                    curr_low = df['low'].iloc[-1]
                    curr_k = df[k_col].iloc[-1]
                    min_k_window = df[k_col].iloc[-window:-1].min()
                    min_low_window = df['low'].iloc[-window:-1].min()
                    is_divergence = (curr_low <= min_low_window * 1.02) & (curr_k > min_k_window * 1.1)

                    ma5 = df['close'].rolling(window=5).mean()
                    above_ma5 = df['close'] > ma5

                    current_idx = df.index[-1]  # 最后一行的索引
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]

                    signal_msg = ""

                    # 极值 J 线反转 (最强短线信号)

                    if prev_row[j_col] < 0 and last_row[j_col] > 5 and kdj_cross.iloc[-1]:
                        signal_msg = "极值J线反转"

                    # 底背离金叉 (波段反转信号)

                    elif kdj_cross.iloc[-1] and is_divergence and last_row[k_col] < 30:
                        signal_msg = "底背离金叉"

                    #  超卖区金叉 + 趋势确认 (稳健信号)

                    elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1] and above_ma5.iloc[-1]:
                        signal_msg = "趋势确认金叉"

                    #  普通低位金叉 (保留原有逻辑作为兜底)
                    elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1]:
                        signal_msg = "低位超卖金叉"

                    # 如果有信号，则记录
                    if signal_msg:
                        ta_signals['KDJ'].append({
                            '股票代码': code,
                            'KDJ_Signal': f"{signal_msg} (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"
                        })

                # 3. CCI (专业分类) - 使用 pandas_ta
                df.ta.cci(append=True, close='close', high='high', low='low')
                cci_cols = [col for col in df.columns if col.startswith('CCI_')]
                if cci_cols:
                    cci_col = cci_cols[0]
                    current_cci = df[cci_col].iloc[-1]
                    cci_signal = self._classify_cci_level(current_cci)

                    # 【修复】如果状态为空（常态），则显示数值，确保不为空白
                    if not cci_signal:
                        cci_signal = f"常态波动 ({current_cci:.2f})"

                    ta_signals['CCI'].append({'股票代码': code, 'CCI_Signal': cci_signal})

                # 4. RSI (超卖低位) - 使用 pandas_ta
                df.ta.rsi(append=True, close='close', length=14)
                rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
                if rsi_cols:
                    rsi_col = rsi_cols[0]
                    curr_rsi = df[rsi_col].iloc[-1]
                    window = 10
                    curr_low = df['low'].iloc[-1]
                    min_low_window = df['low'].iloc[-window:-1].min()
                    min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                    is_price_low = curr_low <= (min_low_window * 1.02)
                    is_rsi_divergence = (is_price_low) and \
                                        (curr_rsi > min_rsi_window * 1.05) and \
                                        (curr_rsi < 50)
                    rsi_msg = f"RSI={curr_rsi:.1f}"
                    if is_rsi_divergence:
                        rsi_msg = f"RSI底背离! ({curr_rsi:.1f})"
                    ta_signals['RSI'].append({'股票代码': code, 'RSI_Signal': rsi_msg})

                # 5. BOLL (低波/缩口) - 使用 pandas_ta
                df.ta.bbands(append=True, length=20, std=2, close='close')
                boll_cols = [col for col in df.columns if col.startswith('BBL_')]
                if boll_cols:
                    lower_band = boll_cols[0]
                    upper_band = [col for col in df.columns if col.startswith('BBU_')][0]
                    df['BOLL_BANDWIDTH'] = (df[upper_band] - df[lower_band]) / df['close']

                    is_narrow = df['BOLL_BANDWIDTH'].iloc[-5:].mean() < df['BOLL_BANDWIDTH'].mean()

                    boll_msg = "低波/缩口" if is_narrow else "常态/张口"

                    ta_signals['BOLL'].append({'股票代码': code, 'BOLL_Signal': boll_msg})

            final_ta_dfs = {}
            for key, value in ta_signals.items():
                final_ta_dfs[key] = pd.DataFrame(value)

            return final_ta_dfs
