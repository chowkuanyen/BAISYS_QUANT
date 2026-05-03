import pandas as pd
from typing import List, Dict

import ta

from MACDAnalyzer import MACDAnalyzer
from FormatManager.ShareCodeFormatMgr import format_stock_code
from KDJAnalyzer import AdvancedKDJAnalyzer

class TASignalProcessor:
    """技术指标信号处理类"""

    def __init__(self, analyzer_instance):
        self.analyzer = analyzer_instance
        self.kdj_analyzer = AdvancedKDJAnalyzer()

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

        symbol_str = hist_df_all['symbol'].astype(str)
        extracted_digits = symbol_str.str.extract(r'(\d{6})', expand=False).fillna('N/A')
        hist_df_all['股票代码'] = extracted_digits.str.zfill(6)

        if 'date' not in hist_df_all.columns and 'trade_date' in hist_df_all.columns:
            hist_df_all.rename(columns={'trade_date': 'date'}, inplace=True)

        hist_df_all.sort_values(['股票代码', 'date'], inplace=True)
        pure_codes_list = [c[2:] if str(c).startswith(('sh', 'sz', 'bj')) else c for c in all_codes]
        code_set = set(pure_codes_list)
        hist_df_all = hist_df_all[hist_df_all['股票代码'].isin(code_set)].copy()

        for code in all_codes:
            pure_code = code[2:] if str(code).startswith(('sh', 'sz', 'bj')) else code
            df = hist_df_all[hist_df_all['股票代码'] == pure_code].copy()

            if df.empty or len(df) < 30:
                continue

            for col in ['close', 'open', 'high', 'low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            if df.empty:
                continue

            if 'close' not in df.columns or 'open' not in df.columns:
                print(f"[ERROR] 股票 {code}: 缺少必要的 OHLC 列，跳过。")
                continue

            df = MACDAnalyzer._custom_macd(self, df)

            try:
                latest_row = df.iloc[-1]
                mom_12269 = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
                mom_6135 = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_6135', 'DEA_6135')
                ta_signals['MACD_DIF_MOMENTUM'] = pd.concat([
                    ta_signals['MACD_DIF_MOMENTUM'],
                    pd.DataFrame([{
                        '股票代码': code,
                        'MACD_12269_DIF': latest_row.get('DIF_12269', 0),
                        'MACD_12269_动能': mom_12269,
                        'MACD_6135_DIF': latest_row.get('DIF_6135', 0),
                        'MACD_6135_动能': mom_6135,
                    }])
                ], ignore_index=True)
            except Exception as e:
                print(f"[WARN] {code} 动能计算失败: {e}")

            # MACD 12269
            detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'
            if detail_col_12269 in df.columns and df[detail_col_12269].iloc[-1] != '':
                signal_detail = df[detail_col_12269].iloc[-1]
                ta_signals['MACD_12269'] = pd.concat([
                    ta_signals['MACD_12269'],
                    pd.DataFrame([{'股票代码': code, 'MACD_12269_Signal': signal_detail}])
                ], ignore_index=True)

            # MACD 6135
            detail_col_6135 = 'MACD_6135_SIGNAL_DETAIL'
            if detail_col_6135 in df.columns and df[detail_col_6135].iloc[-1] != '':
                signal_detail = df[detail_col_6135].iloc[-1]
                ta_signals['MACD_6135'] = pd.concat([
                    ta_signals['MACD_6135'],
                    pd.DataFrame([{'股票代码': code, 'MACD_6135_Signal': signal_detail}])
                ], ignore_index=True)

            # KDJ - 现在通过AdvancedKDJAnalyzer计算
            kdj_signal = self.kdj_analyzer.calculate_kdj_signal_from_df(df)
            if kdj_signal:
                ta_signals['KDJ'] = pd.concat([
                    ta_signals['KDJ'],
                    pd.DataFrame([{
                        '股票代码': code,
                        'KDJ_Signal': kdj_signal
                    }])
                ], ignore_index=True)

            # CCI
            df.ta.cci(append=True, close='close', high='high', low='low')
            cci_cols = [col for col in df.columns if col.startswith('CCI_')]
            if cci_cols:
                cci_col = cci_cols[0]
                current_cci = df[cci_col].iloc[-1]
                cci_signal = self._classify_cci_level(current_cci)
                if not cci_signal:
                    cci_signal = f"常态波动 ({current_cci:.2f})"
                ta_signals['CCI'] = pd.concat([
                    ta_signals['CCI'],
                    pd.DataFrame([{'股票代码': code, 'CCI_Signal': cci_signal}])
                ], ignore_index=True)

            # RSI
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
                is_rsi_divergence = (is_price_low) and (curr_rsi > min_rsi_window * 1.05) and (curr_rsi < 50)
                rsi_msg = f"RSI={curr_rsi:.1f}"
                if is_rsi_divergence:
                    rsi_msg = f"RSI底背离! ({curr_rsi:.1f})"
                ta_signals['RSI'] = pd.concat([
                    ta_signals['RSI'],
                    pd.DataFrame([{'股票代码': code, 'RSI_Signal': rsi_msg}])
                ], ignore_index=True)

            # BOLL
            df.ta.bbands(append=True, length=20, std=2, close='close')
            boll_cols = [col for col in df.columns if col.startswith('BBL_')]
            if boll_cols:
                lower_band = boll_cols[0]
                upper_band = [col for col in df.columns if col.startswith('BBU_')][0]
                df['BOLL_BANDWIDTH'] = (df[upper_band] - df[lower_band]) / df['close']
                is_narrow = df['BOLL_BANDWIDTH'].iloc[-5:].mean() < df['BOLL_BANDWIDTH'].mean()
                boll_msg = "低波/缩口" if is_narrow else "常态/张口"
                ta_signals['BOLL'] = pd.concat([
                    ta_signals['BOLL'],
                    pd.DataFrame([{'股票代码': code, 'BOLL_Signal': boll_msg}])
                ], ignore_index=True)

        for key in ta_signals:
            if not ta_signals[key].empty and '股票代码' in ta_signals[key].columns:
                # 提取 6 位数字，去除 sh/sz 等前缀
                ta_signals[key]['股票代码'] = ta_signals[key]['股票代码'].astype(str).str.extract(r'(\d{6})')
        return ta_signals
