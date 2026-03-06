import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List
import akshare as ak
import pandas as pd
import pandas_ta as ta
import Distribution as dist
import Industrytrending as industry
from DataManager import DatabaseWriter
from DataManager import ParallelUtils as utils
from DataManager import QuantDataPerformer
from FormatManager import Parse_Currency
from MACDAnalyzer import MACDAnalyzer
from SignalManager import TASignalProcessor

class Config:
    def __init__(self):
        self.HOME_DIRECTORY = os.path.expanduser('~')
        self.SAVE_DIRECTORY = os.path.join(self.HOME_DIRECTORY, 'Downloads', 'CoreNews_Reports')
        self.TEMP_DATA_DIRECTORY = os.path.join(self.SAVE_DIRECTORY, 'ShareData')
        self.DATA_FETCH_RETRIES = 3
        self.DATA_FETCH_DELAY = 5
        self.MAX_WORKERS = 15
        self.CODE_ALIASES = {'代码': '股票代码', '证券代码': '股票代码', '股票代码': '股票代码'}
        self.NAME_ALIASES = {'名称': '股票简称', '股票名称': '股票简称', '股票简称': '股票简称', '简称': '股票简称'}
        self.PRICE_ALIASES = {'最新价': '最新价', '现价': '最新价', '当前价格': '最新价', '今收盘': '最新价',
                              '收盘': '最新价', '收盘价': '最新价'}


def format_stock_code(code: str) -> str:
    """根据股票代码的开头数字，添加市场前缀。"""
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        return 'sh' + code_str
    elif code_str.startswith(('0', '3')):
        return 'sz' + code_str
    elif code_str.startswith(('4', '8')):
        return 'bj' + code_str
    return code_str


class StockAnalyzer:

    def __init__(self):
        self.config = Config()
        self.today_str = datetime.now().strftime("%Y%m%d")
        self.temp_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.temp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.start_time = time.time()

    def _get_file_path(self, base_name: str, cleaned: bool = False) -> str:
        """
        生成临时数据文件的完整路径。
        如果 cleaned=True, 则添加 "_经清洗" 后缀。
        """
        suffix = "_经清洗" if cleaned else ""
        file_name = f"{base_name}{suffix}_{self.today_str}.txt"
        return os.path.join(self.temp_dir, file_name)

    def _load_data_from_cache(self, file_path: str) -> pd.DataFrame:
        """从缓存加载数据。"""
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='|', encoding='utf-8', dtype={'股票代码': str})
                print(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
                return df
            except Exception as e:
                print(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
        return pd.DataFrame()

    def _save_data_to_cache(self, df: pd.DataFrame, file_path: str):
        """保存数据到缓存。"""
        try:
            df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
        except Exception as e:
            print(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")

    def _safe_ak_fetch(self, fetch_func: Callable, file_base_name: str, **kwargs: Any) -> pd.DataFrame:

        # 1. 尝试从【清洗后的缓存】加载数据
        cleaned_file_path = self._get_file_path(file_base_name, cleaned=True)
        cached_df = self._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        # 2. 如果清洗后的缓存不存在，则尝试从原始获取
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次获取数据: {file_base_name}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    break  # 成功获取数据，跳出重试循环
                else:
                    print(f"[WARN] 数据返回为空或无效: {file_base_name}，重试中。")
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception as e:
                print(f"[ERROR] 获取 {file_base_name} 时出错: {e}，将在 {self.config.DATA_FETCH_DELAY} 秒后重试。")
                time.sleep(self.config.DATA_FETCH_DELAY)

        if df.empty:
            print(f"[FATAL] 所有重试均失败，返回空 DataFrame: {file_base_name}")
            return pd.DataFrame()

        # 3. 清洗数据并保存到带有 "_经清洗" 后缀的缓存文件
        cleaned_df = self._clean_and_standardize(df, file_base_name)
        if not cleaned_df.empty:
            self._save_data_to_cache(cleaned_df, cleaned_file_path)

        return cleaned_df

    def _clean_and_standardize(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """通用数据清洗和列名标准化。"""
        if df.empty: return df
        df = utils._normalize_fund_data(df)
        # 1. 标准化列名：使用 Config 中的别名映射
        for old, new in self.config.CODE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.NAME_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.PRICE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)

        if '股票代码' not in df.columns:
            return pd.DataFrame()

        # 2. 清洗数据
        df.dropna(subset=['股票代码'], inplace=True)
        df.drop_duplicates(subset=['股票代码'], inplace=True)
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)

        if '最新价' in df.columns:
            df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')

        # 3. 过滤ST股，注意您做北京证券的话可以改下
        if '股票简称' not in df.columns:
            cleaned_df = df.copy()
        else:
            cleaned_df = df[~df['股票简称'].str.contains('ST|st|退市|bj|BJ', case=False, na=False)].copy()

        if len(cleaned_df) == 0:
            print(f"[WARN] {df_name} 数据清洗后为空。")
            return pd.DataFrame()

        return cleaned_df

    def _fetch_individual_industry_parallel(self, codes: List[str]) -> pd.DataFrame:

        from Ts_GetStockBasicinfo import TushareStockManager

        # 1. 定义文件路径
        dict_file_path = os.path.join(self.temp_dir, 'StockIndes.txt')

        if not os.path.exists(dict_file_path):
            print(f"未发现本地字典文件，启动 Tushare 下载逻辑...")
            token = "注册tushare可以得到120积分,够用"

            try:
                manager = TushareStockManager(token)
                manager.get_basic_data(list_status='L', save_path=dict_file_path)
            except Exception as e:
                print(f"[ERROR] 调用 Tushare 接口失败: {e}")
                return pd.DataFrame(columns=['股票代码', '行业'])

        # 3. 加载本地字典文件
        try:
            dict_df = pd.read_csv(
                dict_file_path,
                sep='|',
                encoding='utf-8-sig',
                dtype={'股票代码': str, 'ts_code': str}
            )

            if 'ts_code' in dict_df.columns and '股票代码' not in dict_df.columns:
                dict_df.rename(columns={'ts_code': '股票代码'}, inplace=True)
            if 'industry' in dict_df.columns and '行业' not in dict_df.columns:
                dict_df.rename(columns={'industry': '行业'}, inplace=True)

            # 只取匹配需要的核心列
            dict_df = dict_df[['股票代码', '行业']].drop_duplicates(subset=['股票代码'])

        except Exception as e:
            print(f"[ERROR] 读取本地字典文件失败: {e}")
            return pd.DataFrame(columns=['股票代码', '行业'])

        # 4. 匹配个股行业 (使用 Pandas Merge 效率最高)
        input_df = pd.DataFrame(codes, columns=['股票代码'])
        input_df['股票代码'] = input_df['股票代码'].astype(str).str.zfill(6)
        dict_df['股票代码'] = dict_df['股票代码'].astype(str).str.zfill(6)
        final_df = pd.merge(input_df, dict_df, on='股票代码', how='left')

        final_df['行业'] = final_df['行业'].fillna('N/A')

        match_count = final_df[final_df['行业'] != 'N/A'].shape[0]
        print(f"  - 行业数据匹配完成：总计 {len(codes)} 只，成功匹配 {match_count} 只。")

        return final_df

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """集中获取所有数据源。"""
        print("\n>>> 正在初始化数据获取和缓存检查...")

        # 1. 基础行情和研报数据
        data = {
            'spot_data_all': self._safe_ak_fetch(ak.stock_zh_a_spot, "A股实时行情"),
            'main_report_raw': self._safe_ak_fetch(ak.stock_profit_forecast_em, "主力研报盈利预测"),
            'financial_abstract_raw': self._safe_ak_fetch(ak.stock_financial_abstract, "财务摘要数据"),
            'market_fund_flow_raw': self._safe_ak_fetch(ak.stock_fund_flow_individual, "5日市场资金流向",
                                                        symbol="5日排行"),
            'market_fund_flow_raw_10': self._safe_ak_fetch(ak.stock_fund_flow_individual, "10日市场资金流向",
                                                           symbol="10日排行"),
            'market_fund_flow_raw_20': self._safe_ak_fetch(ak.stock_fund_flow_individual, "20日市场资金流向",
                                                           symbol="20日排行"),
            'strong_stocks_raw': self._safe_ak_fetch(ak.stock_zt_pool_strong_em, "强势股池",
                                                     date=datetime.now().strftime('%Y%m%d')),
            'consecutive_rise_raw': self._safe_ak_fetch(ak.stock_rank_lxsz_ths, "连续上涨"),
            'ljqs_raw': self._safe_ak_fetch(ak.stock_rank_ljqs_ths, "量价齐升"),
            'cxfl_raw': self._safe_ak_fetch(ak.stock_rank_cxfl_ths, "持续放量"),
        }

        # 2. 均线突破数据 (Akshare接口参数不同，需分开获取)
        data['xstp_10_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破10日均线", symbol="10日均线")
        data['xstp_30_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破30日均线", symbol="30日均线")
        data['xstp_60_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破60日均线", symbol="60日均线")

        # 3. 行业板块数据
        print("\n>>> 正在获取行业板块名称并保存至本地...")
        # 定义文件路径
        industry_info_filename = f"行业板块信息_{self.today_str}.txt"
        industry_info_path = os.path.join(self.temp_dir, industry_info_filename)
        industry_board_df = pd.DataFrame()

        if os.path.exists(industry_info_path):
            try:
                print(f"  - 发现本地缓存文件，正在读取: {industry_info_filename}")
                industry_board_df = pd.read_csv(industry_info_path, sep='|', encoding='utf-8-sig')
            except Exception as e:
                print(f"  - [WARN] 读取本地缓存失败: {e}，将尝试重新获取...")

        if industry_board_df.empty:
            print(f"  - 本地无有效缓存，正在通过 Akshare 接口获取...")
            try:
                industry_board_df = ak.stock_board_industry_name_em()

                # 获取成功后保存
                if not industry_board_df.empty:
                    try:
                        industry_board_df.to_csv(industry_info_path, sep='|', index=False, encoding='utf-8-sig')
                        print(f"  - 获取成功并已保存至: {industry_info_filename}")
                    except Exception as e:
                        print(f"  - [ERROR] 保存文件失败: {e}")
            except Exception as e:
                print(f"  - [ERROR] 调用行业板块接口失败: {e}")

        # 后续处理保持不变
        data['top_industry_cons_df'] = self._get_top_industry_constituents(industry_board_df)
        data['industry_board_df'] = industry_board_df
        return data

    def _safe_fetch_constituents(self, symbol: str) -> pd.DataFrame:
        """
        带重试机制获取单个行业板块的成分股。
        """
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                # print(f"  - 尝试获取 {symbol} 成分股 (第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次)")
                df = ak.stock_board_industry_cons_em(symbol=symbol)
                if df is not None and not df.empty:
                    return df
                else:
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception:
                time.sleep(self.config.DATA_FETCH_DELAY)
        return pd.DataFrame()

    def _get_top_industry_constituents(self, industry_board_df: pd.DataFrame) -> pd.DataFrame:
        """重构：获取涨幅前10板块的成分股"""
        if industry_board_df.empty or '板块名称' not in industry_board_df.columns:
            return pd.DataFrame()

        # 1. 缓存检查
        cache_name = "前十板块成分股"
        cleaned_file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        top_industries = industry_board_df.sort_values(by='涨跌幅', ascending=False).head(10)
        # 将 DataFrame 的行转为字典列表，便于传给 worker
        industry_list = top_industries.to_dict('records')

        # 2. 定义 worker
        def fetch_worker(row):
            industry_name = row['板块名称']
            # 调用原本类中定义的 _safe_fetch_constituents
            constituents_df = self._safe_fetch_constituents(symbol=industry_name)

            if constituents_df is not None and not constituents_df.empty:
                if '代码' in constituents_df.columns:
                    constituents_df.rename(columns={'代码': '股票代码'}, inplace=True)

                if '股票代码' in constituents_df.columns:
                    constituents_df['股票代码'] = constituents_df['股票代码'].astype(str).str.zfill(6)
                    constituents_df['所属板块'] = industry_name
                    # 返回清洗后的小 DataFrame
                    return constituents_df[['股票代码', '所属板块']].drop_duplicates()
            return None

        # 3. 调用并发工具
        results = utils.run_with_thread_pool(
            items=industry_list,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc="获取板块成分股"
        )

        # 4. 合并结果
        if results:
            final_df = pd.concat(results, ignore_index=True).drop_duplicates(subset=['股票代码'])
            self._save_data_to_cache(final_df, cleaned_file_path)
            return final_df
        return pd.DataFrame()

    def _fetch_hist_data_parallel(self, codes: List[str], days: int) -> pd.DataFrame:
        """重构：并行获取历史数据"""

        # 1. 准备日期参数
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # 2. 缓存检查 (保持原逻辑)
        cache_name = f"MACD_hist_data_cache"
        file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(file_path)

        if not cached_df.empty:
            if 'close' in cached_df.columns and 'open' in cached_df.columns:
                return cached_df
            else:
                print("[WARN] 历史数据缓存文件缺少关键英文列，将重新获取。")

        # 3. 定义 worker
        def fetch_worker(code):
            try:
                hist_df = ak.stock_zh_a_hist_tx(
                    symbol=format_stock_code(code),
                    start_date=start_date_str,
                    end_date=end_date_str,
                    adjust="qfq"
                )
                if hist_df is not None and not hist_df.empty:
                    # 标准化列名
                    hist_df.rename(columns={
                        'date': 'date', '日期': 'date',
                        'open': 'open', '开盘': 'open',
                        'close': 'close', '收盘': 'close',
                        'high': 'high', '最高': 'high',
                        'low': 'low', '最低': 'low',
                        'volume': 'volume', '成交量': 'volume', 'amount': 'volume',
                    }, inplace=True, errors='ignore')

                    hist_df['股票代码'] = code
                    if 'date' in hist_df.columns:
                        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.strftime('%Y-%m-%d')

                    cols_to_keep = ['date', 'open', 'close', 'high', 'low', 'volume', '股票代码']
                    return hist_df[[col for col in cols_to_keep if col in hist_df.columns]]
            except Exception:
                pass
            return None

        # 4. 调用并发工具
        results = utils.run_with_thread_pool(
            items=codes,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc=f"下载 {days} 天历史数据"
        )

        # 5. 合并结果
        if results:
            merged_df = pd.concat(results, ignore_index=True)
            self._save_data_to_cache(merged_df, file_path)
            return merged_df
        return pd.DataFrame()

    def _save_ta_signals_to_txt(self, ta_signals: Dict[str, pd.DataFrame]):
        """
        将技术指标信号结果保存到独立的 TXT 文件。
        """
        print("\n>>> 正在保存技术指标信号到本地 TXT 文件...")

        save_dir = self.config.TEMP_DATA_DIRECTORY
        today_str = self.today_str

        for indicator_name, df in ta_signals.items():
            if df is None or df.empty:
                continue

            # 命名格式: MACD_12269_Signals_YYYYMMDD.txt
            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)

            try:
                df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
                print(f"  - 成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                print(f"[ERROR] 保存 {indicator_name} 信号文件失败: {e}")



    def _process_xstp_and_filter(self, raw_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> pd.DataFrame:
        """处理并合并均线突破数据，并进行多头排列筛选。"""
        print("正在处理并合并均线突破数据...")

        # 1. 清洗均线数据
        processed_df10 = raw_data['xstp_10_raw'].rename(columns={'最新价': '10日均线价'})
        processed_df30 = raw_data['xstp_30_raw'].rename(columns={'最新价': '30日均线价'})
        processed_df60 = raw_data['xstp_60_raw'].rename(columns={'最新价': '60日均线价'})

        # 2. 合并
        # 使用concat时只保留代码和简称，避免价格列冲突
        merged_df = pd.concat([
            processed_df10[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df30[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df60[['股票代码', '股票简称']].dropna(subset=['股票代码'])
        ]).drop_duplicates(subset=['股票代码'])

        # 3. 重新合并均线价格，确保同一行有所有数据
        xstp_base = merged_df[['股票代码', '股票简称']].drop_duplicates()
        xstp_base = pd.merge(xstp_base, processed_df10[['股票代码', '10日均线价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df30[['股票代码', '30日均线价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df60[['股票代码', '60日均线价']], on='股票代码', how='left')

        # 4. 合并实时价格 (此处仍然按代码合并，以便于均线计算的准确性)
        xstp_base = pd.merge(xstp_base, spot_df[['股票代码', '最新价']], on='股票代码', how='left')

        # 5. 类型转换和过滤
        cols_to_convert = [col for col in xstp_base.columns if '最新价' in col or col == '最新价']
        for col in cols_to_convert:
            # 导入 numpy 是为了使用 np.inf，所以在这里重新导入或使用 float('-inf')
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors='coerce')

        # 过滤条件: 1. 最新价>10日均线 2. 多头排列 (10>30 或 30>60)
        filtered_df = xstp_base[
            (xstp_base['最新价'] > xstp_base['10日均线价']) &
            (
                    (xstp_base['10日均线价'] > xstp_base['30日均线价'].fillna(float('-inf'))) |
                    (xstp_base['30日均线价'] > xstp_base['60日均线价'].fillna(float('-inf')))
            )
            ].copy()

        # 添加完全多头排列标记
        filtered_df['完全多头排列'] = filtered_df.apply(
            # 需要确保在计算前处理 NaN 值，这里使用一个较大的负数代替 NaN
            lambda row: '是' if row['10日均线价'] > row['30日均线价'] and row['30日均线价'] > row[
                '60日均线价'] else '否',
            axis=1
        )

        # 重新命名 '最新价' 为 '当前价格' 以避免与均线价混淆
        filtered_df.rename(columns={'最新价': '当前价格'}, inplace=True)
        return filtered_df.fillna('N/A')

    def _consolidate_data(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并所有数据源和信号，生成最终汇总报告。"""
        print("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")

        all_codes = set()
        data_sources = [
            processed_data.get('processed_main_report'),
            processed_data.get('processed_xstp_df'),
            processed_data.get('market_fund_flow_raw'),
            processed_data.get('strong_stocks_raw'),
            processed_data.get('consecutive_rise_raw'),
            processed_data.get('ljqs_raw'),
            processed_data.get('cxfl_raw'),
            processed_data.get('MACD_12269', pd.DataFrame()),  # MACD 标准周期
            processed_data.get('MACD_6135', pd.DataFrame()),  # MACD 加速周期
            processed_data.get('KDJ', pd.DataFrame()),
            processed_data.get('CCI', pd.DataFrame()),
            processed_data.get('RSI', pd.DataFrame()),
            processed_data.get('BOLL', pd.DataFrame()),
        ]

        for df in data_sources:
            if df is not None and not df.empty and '股票代码' in df.columns:
                all_codes.update(df['股票代码'].unique())

        if not all_codes: return pd.DataFrame()
        final_df = pd.DataFrame(list(all_codes), columns=['股票代码'])

        # 强制代码列表为字符串类型
        final_df['股票代码'] = final_df['股票代码'].astype(str)

        spot_df = processed_data.get('spot_data_all', pd.DataFrame())
        report_df_base = processed_data.get('processed_main_report', pd.DataFrame())

        # 确保实时行情代码列表是字符串类型
        if '股票代码' in spot_df.columns:
            spot_df['股票代码'] = spot_df['股票代码'].astype(str)

        # 构建名称源：从实时行情和研报数据中收集名称 (仍以代码为基础确保唯一性)
        name_source_spot = spot_df[['股票代码', '股票简称']].drop_duplicates(
            subset=['股票代码']) if '股票简称' in spot_df.columns else pd.DataFrame()
        name_source_report = report_df_base[['股票代码', '股票简称']].drop_duplicates(
            subset=['股票代码']) if '股票简称' in report_df_base.columns else pd.DataFrame()

        # 合并所有名称源：优先使用实时行情 (spot_df) 的名称
        all_names = pd.concat([name_source_spot, name_source_report]).drop_duplicates(subset=['股票代码'], keep='first')

        # 合并名称到最终报告 (按代码合并，保证基础表的简称是正确的)
        final_df = pd.merge(final_df, all_names, on='股票代码', how='left')

        # 1. 从实时行情数据中提取用于价格合并的列 (简称和最新价)
        if '股票简称' not in spot_df.columns:
            # 严重警告：如果实时行情数据中没有股票简称，则回退到使用股票代码。
            print("[FATAL] 实时行情数据中缺少 '股票简称' 列，无法按要求按简称关联。回退到按代码关联。")
            price_source_key = '股票代码'
            price_source = spot_df[['股票代码', '最新价']].copy()
        else:
            price_source_key = '股票简称'
            price_source = spot_df[['股票简称', '最新价']].copy()

            # 2. 严格筛选：对价格数据进行数值转换，并只保留非空、非零的价格
            price_source['最新价'] = pd.to_numeric(price_source['最新价'], errors='coerce')
            price_source = price_source[
                (price_source['最新价'].notna()) &
                (price_source['最新价'] > 0)
                ].copy()

            # 3. 关键修改点：根据用户要求，按 "股票简称" 进行去重（处理非唯一性问题）
            price_source = price_source.drop_duplicates(subset=[price_source_key], keep='first')

        # 4. 移除旧的 '最新价' 列 (来自XSTP或之前的合并)
        final_df.drop(columns=['最新价'], errors='ignore', inplace=True)

        # 5. 合并价格 (使用 '股票简称' 或回退键)
        if price_source_key in final_df.columns:
            # 确保合并键是字符串类型
            final_df[price_source_key] = final_df[price_source_key].astype(str)
            price_source[price_source_key] = price_source[price_source_key].astype(str)

            final_df = pd.merge(final_df, price_source, on=price_source_key, how='left')

        # 打印诊断信息
        valid_prices_count = final_df['最新价'].notna().sum() if '最新价' in final_df.columns else 0
        print(
            f"  - 实时行情数据 (最新价) 成功通过 '{price_source_key}' 关联的有效价格数量: {valid_prices_count} / {len(final_df)}")

        # 6. 填充空值
        final_df['股票简称'] = final_df['股票简称'].fillna('N/A')
        # 将价格的 NaN 填充为 'N/A'
        final_df['最新价'] = final_df['最新价'].fillna('N/A')

        report_df = processed_data['processed_main_report'][['股票代码', '机构投资评级(近六个月)-买入']]
        report_df.rename(columns={'机构投资评级(近六个月)-买入': '研报买入次数'}, inplace=True)
        final_df = pd.merge(final_df, report_df, on='股票代码', how='left')
        final_df['研报买入次数'] = pd.to_numeric(final_df['研报买入次数'], errors='coerce').fillna(0).astype(int)

        # B. 均线信号 (多头排列)
        xstp_df = processed_data['processed_xstp_df']
        xstp_cols = ['股票代码', '完全多头排列', '当前价格', '10日均线价', '30日均线价', '60日均线价']

        if not xstp_df.empty and '股票代码' in xstp_df.columns:
            # 1. 筛选出实际存在的列
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]

            # 2. 执行去重和合并
            # 此时 xstp_df 肯定包含 '股票代码'，所以 drop_duplicates 是安全的
            merge_df = xstp_df[cols_present].drop_duplicates(subset=['股票代码'])

            # 3. 执行合并
            final_df = pd.merge(final_df, merge_df, on='股票代码', how='left')

        if '完全多头排列' not in final_df.columns:
            final_df['完全多头排列'] = '否'
        else:
            # 如果存在，则将合并产生的 NaN 填充为 '否'
            final_df['完全多头排列'] = final_df['完全多头排列'].fillna('否')

        # C. 资金流向-5day

        fund_flow_df = processed_data.get('market_fund_flow_raw', pd.DataFrame())
        if not fund_flow_df.empty and '股票简称' in fund_flow_df.columns and '资金流入净额' in fund_flow_df.columns:
            merge_df = fund_flow_df[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df, on='股票简称', how='left')
            final_df['5日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        fund_flow_df_10 = processed_data.get('market_fund_flow_raw_10', pd.DataFrame())
        if not fund_flow_df_10.empty and '股票简称' in fund_flow_df_10.columns and '资金流入净额' in fund_flow_df_10.columns:
            merge_df_10 = fund_flow_df_10[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df_10, on='股票简称', how='left')
            final_df['10日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        fund_flow_df_20 = processed_data.get('market_fund_flow_raw_20', pd.DataFrame())
        if not fund_flow_df_20.empty and '股票简称' in fund_flow_df_20.columns and '资金流入净额' in fund_flow_df_20.columns:
            merge_df_20 = fund_flow_df_20[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df_20, on='股票简称', how='left')
            final_df['20日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        f5_col, f10_col, f20_col = '5日资金流入', '10日资金流入', '20日资金流入'
        if all(col in final_df.columns for col in [f5_col, f10_col, f20_col]):
            def calculate_trend(row):
                # 因为之前清洗过，这里直接取 float 即可
                v5 = Parse_Currency.Parse_Currency.parse_money_str(row[f5_col])
                v10 = Parse_Currency.Parse_Currency.parse_money_str(row[f10_col])
                v20 = Parse_Currency.Parse_Currency.parse_money_str(row[f20_col])

                # 业务逻辑：(5日>10日 或 5日>20日) 且 5日为正流入
                if (v5 > v10 or v5 > v20) and v5 > 0:
                    return "动能增强"
                elif v5 > 0:
                    return "流入"
                else:
                    return ""

            # 创建新列
            final_df['资金动能'] = final_df.apply(calculate_trend, axis=1)

            # 调整列顺序，将参考列放到资金列旁边便于对比
            cols = list(final_df.columns)
            if '资金动能' in cols:
                target_idx = cols.index(f5_col)
                cols.insert(target_idx + 1, cols.pop(cols.index('资金动能')))
                final_df = final_df[cols]

        # D. 强势股池
        if not processed_data['strong_stocks_raw'].empty:
            strong_codes = processed_data['strong_stocks_raw']['股票代码'].tolist()
            final_df['强势股'] = final_df['股票代码'].apply(lambda x: '是' if x in strong_codes else '否')
        else:
            final_df['强势股'] = '否'

        # E. 连续上涨天数
        rise_df = processed_data['consecutive_rise_raw']
        if not rise_df.empty:
            rise_df = rise_df[['股票代码', '连涨天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, rise_df, on='股票代码', how='left').fillna({'连涨天数': 0})
        else:
            final_df['连涨天数'] = 0

        final_df['连涨天数'] = final_df['连涨天数'].astype(int)

        # F. 量价齐升
        if not processed_data['ljqs_raw'].empty:
            ljqs_codes = processed_data['ljqs_raw']['股票代码'].tolist()
            final_df['量价齐升'] = final_df['股票代码'].apply(lambda x: '是' if x in ljqs_codes else '否')
        else:
            final_df['量价齐升'] = '否'

        # G. 持续放量天数
        cxfl_df = processed_data['cxfl_raw']
        if not cxfl_df.empty:
            cxfl_df = cxfl_df[['股票代码', '放量天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, cxfl_df, on='股票代码', how='left').fillna({'放量天数': 0})
        else:
            final_df['放量天数'] = 0

        final_df['放量天数'] = final_df['放量天数'].astype(int)

        ta_dfs_to_merge = []

        # 1. MACD 12269 (标准周期)
        macd_df_standard = processed_data.get('MACD_12269', pd.DataFrame())
        if not macd_df_standard.empty:
            # 确保使用正确的信号列名
            ta_dfs_to_merge.append(macd_df_standard[['股票代码', 'MACD_12269_Signal']].rename(
                columns={'MACD_12269_Signal': 'MACD_12269'}))

        # 2. MACD 6135 (加速周期)
        macd_df_fast = processed_data.get('MACD_6135', pd.DataFrame())
        if not macd_df_fast.empty:
            # 确保使用正确的信号列名
            ta_dfs_to_merge.append(macd_df_fast[['股票代码', 'MACD_6135_Signal']].rename(
                columns={'MACD_6135_Signal': 'MACD_6135'}))

        # 3 KDJ
        kdj_df = processed_data.get('KDJ', pd.DataFrame())
        if not kdj_df.empty:
            ta_dfs_to_merge.append(kdj_df[['股票代码', 'KDJ_Signal']].rename(
                columns={'KDJ_Signal': 'KDJ_Signal'}))

        # 4CCI
        cci_df = processed_data.get('CCI', pd.DataFrame())
        if not cci_df.empty:
            # 改进后的CCI信号已包含数值和状态，此处只保留信号字符串
            ta_dfs_to_merge.append(cci_df[['股票代码', 'CCI_Signal']].rename(
                columns={'CCI_Signal': 'CCI_Signal'}))

        # 5 RSI
        rsi_df = processed_data.get('RSI', pd.DataFrame())
        if not rsi_df.empty:
            # 只保留信号的文字描述，移除数值
            rsi_df['RSI_Signal'] = rsi_df['RSI_Signal'].astype(str).str.split(' ').str[0]
            ta_dfs_to_merge.append(rsi_df[['股票代码', 'RSI_Signal']].rename(
                columns={'RSI_Signal': 'RSI_Signal'}))

        # 6. BOLL
        boll_df = processed_data.get('BOLL', pd.DataFrame())
        if not boll_df.empty:
            ta_dfs_to_merge.append(boll_df[['股票代码', 'BOLL_Signal']].rename(
                columns={'BOLL_Signal': 'BOLL_Signal'}))

        # 将所有 TA 信号合并到 final_df
        for ta_df in ta_dfs_to_merge:
            final_df = pd.merge(final_df, ta_df.drop_duplicates(subset=['股票代码']), on='股票代码', how='left')

        # 合并 MACD 动能状态数据

        momentum_df = processed_data.get('MACD_DIF_MOMENTUM', pd.DataFrame())
        if not momentum_df.empty and '股票代码' in momentum_df.columns:
            # 动能数据本身不作为过滤条件，直接 Left Join
            final_df = pd.merge(final_df, momentum_df, on='股票代码', how='left')
            # 填充空值
            for col in ['MACD_12269_动能', 'MACD_6135_动能']:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna('')

        # 对新的 TA 信号列填充空值
        for col in ['MACD_12269', 'MACD_6135', 'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal']:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna('')
            else:
                # 如果某个信号源为空，则补充空列以保持一致性
                final_df[col] = ''
        top_ind_df = processed_data.get('top_industry_cons_df', pd.DataFrame())
        if not top_ind_df.empty:
            top_codes = set(top_ind_df['股票代码'].astype(str).unique())
            final_df['TOP10行业'] = final_df['股票代码'].apply(lambda x: '是' if str(x) in top_codes else '否')
        else:
            final_df['TOP10行业'] = '否'

        industry_df = processed_data.get('individual_industry', pd.DataFrame())
        if not industry_df.empty:
            final_df = pd.merge(final_df, industry_df, on='股票代码', how='left')
            final_df['行业'] = final_df['行业'].fillna('N/A')

        # 4. 过滤掉没有任何主要信号的股票
        def has_any_signal(row):
            # 这里的研报买入次数列在 run() 方法中已过滤，且在 merge 填充 NaN 为 0，故此处检查 > 0 即可
            return (row['研报买入次数'] > 0 or
                    row['完全多头排列'] == '是' or
                    row['强势股'] == '是' or
                    row['量价齐升'] == '是' or
                    row.get('TOP10行业') == '是' or
                    row['MACD_12269'] != '' or
                    row['MACD_6135'] != '' or
                    row['KDJ_Signal'] != '' or
                    row['CCI_Signal'] != '' or
                    row['RSI_Signal'] != '' or
                    row['BOLL_Signal'] != ''
                    )

        final_df = final_df[final_df.apply(has_any_signal, axis=1)].copy()

        # 5. 整理输出并排序 (按研报买入次数、连涨天数、放量天数降序排序)
        final_df.sort_values(by=['研报买入次数', '连涨天数', '放量天数'], ascending=[False, False, False], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        final_df['完整股票代码'] = final_df['股票代码'].apply(format_stock_code)
        final_df['股票链接'] = "https://hybrid.gelonghui.com/stock-check/" + final_df['完整股票代码']

        # 3. 移除临时列
        final_df.drop(columns=['完整股票代码'], inplace=True, errors='ignore')

        if '当前价格' in final_df.columns and '最新价' in final_df.columns:
            final_df.drop(columns=['当前价格'], inplace=True, errors='ignore')

        # 最终列顺序 (手动调整，确保最重要的信息在前)
        base_cols = ['股票代码', '股票简称', '行业', '最新价', '获利比例', '90集中度', '平均成本', '筹码状态']

        signal_cols = [
            '强势股', '量价齐升', '连涨天数', '放量天数', 'TOP10行业',
            'MACD_12269', 'MACD_12269_动能', 'MACD_12269_DIF',
            'MACD_6135', 'MACD_6135_动能', 'MACD_6135_DIF',
            'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal',
        ]
        report_cols = [
            '研报买入次数', '完全多头排列', '10日均线价', '30日均线价', '60日均线价',
            '资金动能', '5日资金流入', '10日资金流入', '20日资金流入'
        ]
        final_cols = base_cols + signal_cols + report_cols + ['股票链接']
        final_df = final_df[[col for col in final_cols if col in final_df.columns]]

        return final_df

    def _merge_industry_signal_to_stocks(self, stock_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        将行业分析的结论('行业信号'列)，精准匹配到每一只股票上。
        """
        if industry_df.empty or stock_df.empty or '行业' not in stock_df.columns:
            stock_df['所属行业信号'] = ''
            return stock_df

        print("  - 正在将行业信号映射至个股...")
        signal_map = industry_df.set_index('行业名称')['行业信号'].to_dict()
        stock_df['所属行业信号'] = stock_df['行业'].map(signal_map).fillna('')

        return stock_df

    def _generate_report(self, sheets_data: Dict[str, pd.DataFrame]):
        """生成 Excel 报告。"""
        print(f"\n>>> 正在生成 Excel 报告...")
        report_path = os.path.join(self.config.SAVE_DIRECTORY, f"股票筛选报告_{self.today_str}.xlsx")

        try:
            writer = pd.ExcelWriter(report_path, engine='xlsxwriter')
            workbook = writer.book

            # 定义格式
            header_format = workbook.add_format(
                {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            currency_format = workbook.add_format({'num_format': '#,##0.00'})
            code_format = workbook.add_format({'num_format': '@'})  # 文本格式

            for sheet_name, df in sheets_data.items():
                if df is None or df.empty:
                    print(f"  - 警告：工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
                worksheet = writer.sheets[sheet_name]

                # 写入表头
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # 设置列宽和格式
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)

                    # 修复：确保 '最新价' 列被正确识别为货币格式
                    if col == '最新价' or '价格' in col or '价' in col or '线' in col or '均线' in col:
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif '代码' in col:
                        worksheet.set_column(i, i, 10, code_format)
                    else:
                        worksheet.set_column(i, i, col_width)

            writer.close()
            print(f"  - 报告已成功生成并保存到: {report_path}")

        except Exception as e:
            print(f"[FATAL] 致命错误：生成 Excel 报告失败。原因: {e}")
            raise

    def run(self):
        """主运行方法。"""
        print(f"股票分析程序启动 (Today: {self.today_str})")

        try:

            # 预处理行业权重数据
            industry_analyzer = industry.IndustryFlowAnalyzer(self.config)
            industry_analysis_df = industry_analyzer.run_analysis()

            # 获取所有原始数据
            raw_data = self._get_all_raw_data()

            # 预处理主力研报数据
            processed_main_report = raw_data.get('main_report_raw', pd.DataFrame())
            spot_df = raw_data.get('spot_data_all', pd.DataFrame())

            report_col = '机构投资评级(近六个月)-买入'
            if not processed_main_report.empty and report_col in processed_main_report.columns:
                # 确保过滤列是数字类型
                processed_main_report[report_col] = pd.to_numeric(processed_main_report[report_col],
                                                                  errors='coerce').fillna(0)

                # 过滤条件：买入评级 >= 1
                processed_main_report = processed_main_report[processed_main_report[report_col] > 1].copy()

                if processed_main_report.empty:
                    print("[WARN] 主力研报数据过滤后为空，回退至使用【A股实时行情】中的股票作为分析基础。")
                else:
                    print(">>> 主力研报数据已过滤，仅保留 '买入' 评级 > 1 的股票。")

            if not processed_main_report.empty and '股票代码' in processed_main_report.columns:
                # 使用过滤后的列表作为分析基础
                all_codes = set(processed_main_report['股票代码'].unique())
            elif not spot_df.empty and '股票代码' in spot_df.columns:
                print("[WARN] 主力研报数据为空或无效，回退至使用【A股实时行情】中的股票作为分析基础。")

                all_codes = set(spot_df['股票代码'].unique())
            else:
                print("[FATAL] 无法获取有效的股票代码列表，流程终止。")
                return
            top_ind_df = raw_data.get('top_industry_raw', pd.DataFrame())
            if not top_ind_df.empty:
                all_codes.update(top_ind_df['股票代码'].astype(str).unique())

            filtered_codes_list = list(all_codes)

            if not filtered_codes_list:
                print("未找到任何有效的股票代码，流程终止。")
                return

            # 历史数据获取和技术指标计算
            hist_df_all = self._fetch_hist_data_parallel(filtered_codes_list, days=365)
            signal_processor = TASignalProcessor(self)
            ta_signals = signal_processor.process_signals (filtered_codes_list, hist_df_all,spot_df)
            self._save_ta_signals_to_txt(ta_signals)
            print(">>> 股票历史数据和技术指标分析完成。")

            industry_info_df = self._fetch_individual_industry_parallel(filtered_codes_list)

            universe_codes_set = set(filtered_codes_list)

            def filter_df_by_universe(df, universe_set):
                if df is None or df.empty or '股票代码' not in df.columns:
                    return pd.DataFrame()
                df['股票代码'] = df['股票代码'].astype(str)
                return df[df['股票代码'].isin(universe_set)].copy()

            # 均线突破数据处理
            processed_xstp_df = self._process_xstp_and_filter(raw_data, spot_df)
            processed_xstp_df = filter_df_by_universe(processed_xstp_df, universe_codes_set)

            # 过滤其他每日排名数据
            raw_data['market_fund_flow_raw'] = filter_df_by_universe(raw_data['market_fund_flow_raw'],
                                                                     universe_codes_set)
            raw_data['strong_stocks_raw'] = filter_df_by_universe(raw_data['strong_stocks_raw'], universe_codes_set)
            raw_data['consecutive_rise_raw'] = filter_df_by_universe(raw_data['consecutive_rise_raw'],
                                                                     universe_codes_set)
            raw_data['ljqs_raw'] = filter_df_by_universe(raw_data['ljqs_raw'], universe_codes_set)
            raw_data['cxfl_raw'] = filter_df_by_universe(raw_data['cxfl_raw'], universe_codes_set)

            # 5. 合并所有数据源和信号
            processed_data = {
                **raw_data,
                **ta_signals,
                'processed_xstp_df': processed_xstp_df,
                'processed_main_report': processed_main_report,
                'individual_industry': industry_info_df
            }

            consolidated_report = self._consolidate_data(processed_data)
            consolidated_report = self._merge_industry_signal_to_stocks(consolidated_report, industry_analysis_df)

            cols = list(consolidated_report.columns)
            if '所属行业信号' in cols and '行业' in cols:
                cols.remove('所属行业信号')
                idx = cols.index('行业')
                cols.insert(idx + 1, '所属行业信号')
                consolidated_report = consolidated_report[cols]

            print(">>> 正在执行最终数据清洗：剔除弱势且加速下跌的个股...")

            if not consolidated_report.empty:
                # 为了安全比较，确保 DIF 列被正确解析为数字，非数字转为 NaN
                dif_12269 = pd.to_numeric(consolidated_report.get('MACD_12269_DIF'), errors='coerce')
                dif_6135 = pd.to_numeric(consolidated_report.get('MACD_6135_DIF'), errors='coerce')
                kdj_col = consolidated_report.get('kdj_singal', pd.Series([''] * len(consolidated_report),
                                                                          index=consolidated_report.index))
                kdj_is_empty = kdj_col.isna() | (kdj_col.astype(str).str.strip().str.lower().isin(['', 'nan', 'none']))
                # 定义剔除条件（所有条件需同时满足）
                drop_condition = (
                        (consolidated_report.get('强势股') == '否') &
                        (consolidated_report.get('量价齐升') == '否') &
                        (consolidated_report.get('连涨天数') == 0) &
                        (consolidated_report.get('放量天数') == 0) &
                        (consolidated_report.get('MACD_12269_动能') == '加速下跌 (绿柱加长)') &
                        (consolidated_report.get('MACD_6135_动能') == '加速下跌 (绿柱加长)') &
                        (dif_12269 < 0) &
                        (dif_6135 < 0) &
                        kdj_is_empty &
                        (consolidated_report.get('5日资金流入', pd.Series(dtype=str)).astype(str).str.contains('-',
                                                                                                               na=False))
                )

                # 记录过滤前的数量
                initial_count = len(consolidated_report)
                consolidated_report = consolidated_report[~drop_condition].copy()
                dropped_count = initial_count - len(consolidated_report)
                print(
                    f"  - 清洗完成：共排除了 {dropped_count} 只符合极度弱势特征的股票。剩余 {len(consolidated_report)} 只。")

            if not consolidated_report.empty:
                print("\n>>> 正在为最终保留的个股获取筹码分布数据...")
                # 提取最终保留的股票代码
                final_codes = consolidated_report['股票代码'].unique().tolist()

                chip_file_name = f"筹码分布数据_精选后_{self.today_str}.txt"
                chip_file_path = os.path.join(self.temp_dir, chip_file_name)
                chip_data_df = pd.DataFrame()

                if os.path.exists(chip_file_path):
                    try:
                        print(f"  - 发现本地筹码分布缓存，正在读取: {chip_file_name}")
                        chip_data_df = pd.read_csv(chip_file_path, sep='|', encoding='utf-8-sig',
                                                   dtype={'股票代码': str})
                    except Exception as e:
                        print(f"  - [WARN] 读取筹码分布缓存失败: {e}，将尝试重新获取...")

                if chip_data_df.empty:
                    chip_analyzer = dist.ChipDistributionAnalyzer(self.config)
                    chip_data_df = chip_analyzer.fetch_chip_data_parallel(final_codes)

                    if not chip_data_df.empty:
                        try:
                            chip_data_df.to_csv(chip_file_path, sep='|', index=False, encoding='utf-8-sig')
                        except Exception as e:
                            print(f"  - [ERROR] 保存筹码分布缓存失败: {e}")

                if not chip_data_df.empty:
                    consolidated_report = pd.merge(consolidated_report, chip_data_df, on='股票代码', how='left')

                    base_cols = ['股票代码', '股票简称', '行业', '所属行业信号', '最新价', '获利比例', '90集中度',
                                 '平均成本', '筹码状态']
                    other_cols = [c for c in consolidated_report.columns if c not in base_cols and c != '股票链接']
                    final_cols = [c for c in base_cols if c in consolidated_report.columns] + other_cols + ['股票链接']

                    consolidated_report = consolidated_report[
                        [c for c in final_cols if c in consolidated_report.columns]]

            # 6. 准备报告数据
            sheets_data = {
                '数据汇总': consolidated_report,
                '行业深度分析': industry_analysis_df,
                '主力研报筛选': processed_data['processed_main_report'],
                '均线多头排列': processed_xstp_df,
                '实时行情': spot_df,
                '5日市场资金流向': raw_data['market_fund_flow_raw'],
                '10日市场资金流向': raw_data['market_fund_flow_raw_10'],
                '20日市场资金流向': raw_data['market_fund_flow_raw_20'],
                '强势股池': raw_data['strong_stocks_raw'],
                '连续上涨': raw_data['consecutive_rise_raw'],
                '量价齐升': raw_data['ljqs_raw'],
                '持续放量': raw_data['cxfl_raw'],
                'MACD_12269金叉': ta_signals.get('MACD_12269', pd.DataFrame()),
                'MACD_6135金叉': ta_signals.get('MACD_6135', pd.DataFrame()),
                'MACD_DIF_动能状态': ta_signals.get('MACD_DIF_MOMENTUM', pd.DataFrame()),
                'KDJ超卖金叉': ta_signals.get('KDJ', pd.DataFrame()),
                'CCI专业状态': ta_signals.get('CCI', pd.DataFrame()),
                'RSI超卖': ta_signals.get('RSI', pd.DataFrame()),
                'BOLL低波': ta_signals.get('BOLL', pd.DataFrame()),
                '前十板块成分股': raw_data['top_industry_cons_df'],
            }

            # 7. 生成报告
            self._generate_report(sheets_data)

            try:

                db_manager = DatabaseWriter.QuantDBManager(
                    user='postgres', password='025874yan',
                    host='127.0.0.1', port='5432', db_name='Corenews'
                )

                sync_task = QuantDataPerformer.QuantDBSyncTask(db_manager)


                sync_task.sync_all(
                    today_str=self.today_str,
                    consolidated_report=consolidated_report,
                    industry_df=industry_analysis_df,
                    raw_data=raw_data
                )

                db_manager.close()
                print("数据库同步成功完成。")

            except Exception as e:
                print(f"!!! [同步中断] 任务运行异常: {e}")

        except Exception as e:
            print(f"\n[FATAL] 致命错误：数据分析流程意外终止。原因: {e}")
            raise

        finally:
            end_time = time.time()
            print(f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}")

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()

