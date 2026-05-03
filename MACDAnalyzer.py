import pandas as pd
import numpy as np
from typing import List, Dict
import pandas_ta as ta
from scipy.signal import find_peaks


class MACDAnalyzer:

    @staticmethod
    def _find_peaks_troughs(series, distance=10):
        """
        使用 scipy.find_peaks 寻找序列的波峰和波谷。
        :param series: 输入序列 (如 close, DIF)
        :param distance: 波峰/波谷之间的最小距离
        :return: (peaks_indices, troughs_indices)
        """
        peaks, _ = find_peaks(series, distance=distance)
        troughs, _ = find_peaks(-series, distance=distance)
        return peaks, troughs

    @staticmethod
    def _detect_divergence_single_param(df, price_series, indicator_series, distance=10):
        """
        检测单一参数组的顶/底背离。
        :param df: DataFrame
        :param price_series: 价格序列 (如 df['close'])
        :param indicator_series: 指标序列 (如 df['DIF_12269'])
        :param distance: 用于 find_peaks 的距离参数
        :return: ('顶背离', index) or ('底背离', index) or (None, None)
        """
        peaks_price, troughs_price = MACDAnalyzer._find_peaks_troughs(price_series, distance)
        peaks_indicator, troughs_indicator = MACDAnalyzer._find_peaks_troughs(indicator_series, distance)

        # 检查顶背离 (价格新高，指标没新高)
        if len(peaks_price) >= 2 and len(peaks_indicator) >= 2:
            # 取最后两个波峰
            last_p_idx = peaks_price[-1]
            prev_p_idx = peaks_price[-2]

            last_i_idx = peaks_indicator[-1]
            prev_i_idx = peaks_indicator[-2]

            # 确保波峰出现在相近的时间窗口内，这里简化处理，直接比较最后的两个
            # 更严谨的做法是使用最近的两个价格波峰对应的指标值
            price_higher = price_series.iloc[last_p_idx] > price_series.iloc[prev_p_idx]
            indicator_lower = indicator_series.iloc[last_i_idx] < indicator_series.iloc[prev_i_idx]

            # 为了简化，我们假设最后的几个波峰是对应的
            if price_higher and indicator_lower:
                # 找到最近的价格波峰对应的指标值
                corresponding_prev_i_val = indicator_series.iloc[prev_p_idx]  # 这个索引可能不在 peaks_indicator 中
                corresponding_last_i_val = indicator_series.iloc[last_p_idx]  # 这个索引可能不在 peaks_indicator 中

                # 更严谨的方式：找到与 price peaks 最近的 indicator 峰值
                closest_i_peak_for_last_p = peaks_indicator[peaks_indicator <= last_p_idx][-1] if any(
                    peaks_indicator <= last_p_idx) else None
                closest_i_peak_for_prev_p = peaks_indicator[peaks_indicator <= prev_p_idx][-1] if any(
                    peaks_indicator <= prev_p_idx) else None

                if closest_i_peak_for_last_p is not None and closest_i_peak_for_prev_p is not None:
                    if (price_series.iloc[last_p_idx] > price_series.iloc[prev_p_idx] and
                            indicator_series.iloc[closest_i_peak_for_last_p] < indicator_series.iloc[
                                closest_i_peak_for_prev_p]):
                        return '顶背离', last_p_idx

        # 检查底背离 (价格新低，指标没新低)
        if len(troughs_price) >= 2 and len(troughs_indicator) >= 2:
            # 取最后两个波谷
            last_p_idx = troughs_price[-1]
            prev_p_idx = troughs_price[-2]

            last_i_idx = troughs_indicator[-1]
            prev_i_idx = troughs_indicator[-2]

            # 更严谨的方式：找到与 price troughs 最近的 indicator 谷值
            closest_i_trough_for_last_p = troughs_indicator[troughs_indicator <= last_p_idx][-1] if any(
                troughs_indicator <= last_p_idx) else None
            closest_i_trough_for_prev_p = troughs_indicator[troughs_indicator <= prev_p_idx][-1] if any(
                troughs_indicator <= prev_p_idx) else None

            if closest_i_trough_for_last_p is not None and closest_i_trough_for_prev_p is not None:
                if (price_series.iloc[last_p_idx] < price_series.iloc[prev_p_idx] and
                        indicator_series.iloc[closest_i_trough_for_last_p] > indicator_series.iloc[
                            closest_i_trough_for_prev_p]):
                    return '底背离', last_p_idx

        return None, None

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

    @staticmethod
    def detect_combined_divergence(df: pd.DataFrame) -> Dict[str, str]:
        """
        检测并结合两套 MACD 参数的背离信号。
        :param df: 包含价格和 MACD 指标的数据框
        :return: 包含不同背离组合信号的字典
        """
        # 定义不同的时间范畴
        distance_slow = 25  # 用于 12269 的较大距离
        distance_fast = 12  # 用于 6135 的较小距离

        # 检测 12269 的背离 (战略层)
        div_12269_top, top_idx_12269 = MACDAnalyzer._detect_divergence_single_param(
            df, df['close'], df['DIF_12269'], distance=distance_slow
        )
        div_12269_bot, bot_idx_12269 = MACDAnalyzer._detect_divergence_single_param(
            df, df['close'], df['DIF_12269'], distance=distance_slow
        )

        # 检测 6135 的背离 (战术层)
        div_6135_top, top_idx_6135 = MACDAnalyzer._detect_divergence_single_param(
            df, df['close'], df['DIF_6135'], distance=distance_fast
        )
        div_6135_bot, bot_idx_6135 = MACDAnalyzer._detect_divergence_single_param(
            df, df['close'], df['DIF_6135'], distance=distance_fast
        )

        signals = {}

        # 检查是否为最近的信号
        is_recent = lambda idx: idx is not None and len(df) - idx <= 5  # 假设最近5根K线内有效

        # 双重顶背离
        if div_12269_top == '顶背离' and is_recent(top_idx_12269) and \
                div_6135_top == '顶背离' and is_recent(top_idx_6135):
            signals['combined_signal'] = '双重顶背离 (强烈卖出)'
        # 双重底背离
        elif div_12269_bot == '底背离' and is_recent(bot_idx_12269) and \
                div_6135_bot == '底背离' and is_recent(bot_idx_6135):
            signals['combined_signal'] = '双重底背离 (强烈买入)'

        # 战略预警 + 战术确认 (买入)
        elif div_12269_bot == '底背离' and is_recent(bot_idx_12269) and \
                div_6135_bot == '底背离' and is_recent(bot_idx_6135) and \
                df['DIF_6135'].iloc[-1] > df['DEA_6135'].iloc[-1]:  # 战术金叉确认
            signals['combined_signal'] = '战略底背离+战术金叉确认 (精准买入)'
        # 战略预警 + 战术确认 (卖出)
        elif div_12269_top == '顶背离' and is_recent(top_idx_12269) and \
                div_6135_top == '顶背离' and is_recent(top_idx_6135) and \
                df['DIF_6135'].iloc[-1] < df['DEA_6135'].iloc[-1]:  # 战术死叉确认
            signals['combined_signal'] = '战略顶背离+战术死叉确认 (精准卖出)'

        # 仅 12269 背离 (大趋势预警)
        elif div_12269_top == '顶背离' and is_recent(top_idx_12269):
            signals['combined_signal'] = '12269 顶背离 (卖出预警)'
        elif div_12269_bot == '底背离' and is_recent(bot_idx_12269):
            signals['combined_signal'] = '12269 底背离 (买入关注)'

        # 仅 6135 背离 (小趋势信号，但需过滤)
        elif div_6135_top == '顶背离' and is_recent(top_idx_6135):
            if df['DIF_12269'].iloc[-1] > 0:  # 大趋势向上时，小背离可能无效
                signals['combined_signal'] = '6135 顶背离 (需结合大趋势判断)'
            else:
                signals['combined_signal'] = '6135 顶背离 (可考虑卖出)'
        elif div_6135_bot == '底背离' and is_recent(bot_idx_6135):
            if df['DIF_12269'].iloc[-1] < 0:  # 大趋势向下时，小背离可能无效
                signals['combined_signal'] = '6135 底背离 (需结合大趋势判断)'
            else:
                signals['combined_signal'] = '6135 底背离 (可考虑买入)'

        else:
            signals['combined_signal'] = ''  # 无背离信号

        # 记录原始信号，便于调试
        signals['div_12269_top'] = div_12269_top
        signals['div_12269_bot'] = div_12269_bot
        signals['div_6135_top'] = div_6135_top
        signals['div_6135_bot'] = div_6135_bot

        return signals