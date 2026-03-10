import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Callable, Dict, Any, List
import akshare as ak
import pandas as pd
import pandas_ta as ta  # 勿删
from sqlalchemy import text
import Distribution as dist
import Industrytrending as industry
from DataManager import DatabaseWriter
from DataManager import ParallelUtils as utils
from DataManager import QuantDataPerformer
from FormatManager import Parse_Currency
from SignalManager import TASignalProcessor
from HistDataEngine import StockSyncEngine
from LoggerManager import LoggerManager


class Config:
    def __init__(self):
        self.HOME_DIRECTORY = os.path.expanduser('~')
        self.SAVE_DIRECTORY = os.path.join(self.HOME_DIRECTORY, 'Downloads', 'CoreNews_Reports')
        self.TEMP_DATA_DIRECTORY = os.path.join(self.SAVE_DIRECTORY, 'ShareData')
        os.makedirs(self.TEMP_DATA_DIRECTORY, exist_ok=True)  # 确保目录存在
        self.DATA_FETCH_RETRIES = 3
        self.DATA_FETCH_DELAY = 5
        self.MAX_WORKERS = 15
        self.CODE_ALIASES = {'代码': '股票代码', '证券代码': '股票代码', '股票代码': '股票代码'}
        self.NAME_ALIASES = {'名称': '股票简称', '股票名称': '股票简称', '股票简称': '股票简称', '简称': '股票简称'}
        self.PRICE_ALIASES = {'最新价': '最新价', '现价': '最新价', '当前价格': '最新价', '今收盘': '最新价',
                              '收盘': '最新价', '收盘价': '最新价'}


def format_stock_code(code: str) -> str:
    """
    根据股票代码的开头数字，添加市场前缀。
    兼容 '000001' 和 'sz000001' 两种输入，并统一输出 'sz000001' 格式。
    """
    code_str = str(code)

    if code_str.startswith(('sh', 'sz', 'bj')):
        return code_str

    code_str = code_str.zfill(6)
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

        
        self.logger = LoggerManager(
            log_dir=os.path.join(self.config.SAVE_DIRECTORY, "Logs"),
            log_filename=f"Corenews_Main_{self.today_str}.log",
            level="INFO"
        )

        self.logger.info("Initializing StockSyncEngine...")
        try:
            self.sync_engine = StockSyncEngine()
            # 显式检查 'db' 属性是否存在且不为 None
            if not hasattr(self.sync_engine, 'db') or self.sync_engine.db is None:
                raise AttributeError(
                    "StockSyncEngine object was created, but its 'db' attribute is missing or None. "
                    "This usually indicates a failure during database engine creation within StockSyncEngine's __init__."
                )
            self.db_engine = self.sync_engine.db
            self.logger.info("Successfully initialized StockSyncEngine and database engine.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize StockSyncEngine or its database engine. Error: {e}")
            raise  # 重新抛出异常，阻止程序继续在不健康的状态下运行

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
                df = pd.read_csv(file_path, sep='|', encoding='utf-8', dtype={'股票代码': str, 'symbol': str})
                if 'symbol' in df.columns and '股票代码' not in df.columns:
                    df.rename(columns={'symbol': '股票代码'}, inplace=True)
                return df
            except Exception as e:
                self.logger.warning(f"加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
        return pd.DataFrame()

    def _save_data_to_cache(self, df: pd.DataFrame, file_path: str):
        """保存数据到缓存。"""
        try:
            df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
        except Exception as e:
            self.logger.error(f"保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")

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
                self.logger.info(f"正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次获取数据: {file_base_name}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    break
                else:
                    self.logger.warning(f"数据返回为空或无效: {file_base_name}，重试中。")
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception as e:
                self.logger.error(f"获取 {file_base_name} 时出错: {e}，将在 {self.config.DATA_FETCH_DELAY} 秒后重试。")
                time.sleep(self.config.DATA_FETCH_DELAY)

        if df.empty:
            self.logger.critical(f"所有重试均失败，返回空 DataFrame: {file_base_name}")
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

        for old, new in self.config.CODE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.NAME_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.PRICE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)

        if '股票代码' not in df.columns:
            return pd.DataFrame()

        df.dropna(subset=['股票代码'], inplace=True)
        df.drop_duplicates(subset=['股票代码'], inplace=True)
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)

        if '最新价' in df.columns:
            df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')

        if '股票简称' not in df.columns:
            cleaned_df = df.copy()
        else:
            cleaned_df = df[~df['股票简称'].str.contains('ST|st|退市', case=False, na=False)].copy()

        if len(cleaned_df) == 0:
            self.logger.warning(f"{df_name} 数据清洗后为空。")
            return pd.DataFrame()

        return cleaned_df

    def _load_industry_info_from_generated_file(self, codes_pure_digits: List[str]) -> pd.DataFrame:
        dict_file_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, f'StockIndes_{self.today_str}.txt')

        industry_df_raw = pd.DataFrame()
        if os.path.exists(dict_file_path):
            try:
                industry_df_raw = pd.read_csv(
                    dict_file_path,
                    sep='|',
                    encoding='utf-8-sig',
                    dtype={'ts_code': str, 'symbol': str, 'name': str, 'industry': str}
                )

                def to_pure_code(code_str):
                    if pd.isna(code_str): return None
                    if '.' in code_str:
                        return code_str.split('.')[0].zfill(6)
                    elif code_str.startswith(('sh', 'sz', 'bj')):
                        return code_str[2:].zfill(6)
                    return str(code_str).zfill(6)

                industry_df_raw['股票代码'] = industry_df_raw['ts_code'].apply(to_pure_code)
                if '股票代码' not in industry_df_raw.columns:
                    self.logger.warning("股票基本信息文件缺少 'ts_code' 或 '股票代码' 列。")
                    return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

                if 'industry' in industry_df_raw.columns and '行业' not in industry_df_raw.columns:
                    industry_df_raw.rename(columns={'industry': '行业'}, inplace=True)
                elif '行业' not in industry_df_raw.columns:
                    self.logger.warning("股票基本信息文件缺少 'industry' 或 '行业' 列。")
                    return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

                if 'name' in industry_df_raw.columns and '股票简称' not in industry_df_raw.columns:
                    industry_df_raw.rename(columns={'name': '股票简称'}, inplace=True)
                elif '股票简称' not in industry_df_raw.columns:
                    pass

                cols_to_keep = ['股票代码', '行业', '股票简称']
                industry_df_cleaned = industry_df_raw[
                    [col for col in cols_to_keep if col in industry_df_raw.columns]].copy()
                industry_df_cleaned.drop_duplicates(subset=['股票代码'], inplace=True)

                input_df_codes = pd.DataFrame(codes_pure_digits, columns=['股票代码'])
                input_df_codes['股票代码'] = input_df_codes['股票代码'].astype(str).str.zfill(6)
                final_industry_df = pd.merge(input_df_codes, industry_df_cleaned, on='股票代码', how='left')
                final_industry_df['行业'] = final_industry_df['行业'].fillna('N/A')
                final_industry_df['股票简称'] = final_industry_df['股票简称'].fillna('N/A')

                match_count = final_industry_df[final_industry_df['行业'] != 'N/A'].shape[0]
                self.logger.info(f"从本地文件加载行业数据匹配完成：总计 {len(codes_pure_digits)} 只，成功匹配 {match_count} 只。")
                return final_industry_df

            except Exception as e:
                self.logger.error(f"读取或处理本地股票基本信息文件失败: {e}，将返回空 DataFrame。")
        else:
            self.logger.warning(f"未找到 HistDataEngine 生成的股票基本信息文件: {dict_file_path}，将返回空 DataFrame。")

        return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("正在初始化数据获取和缓存检查...")

        data = {
            'spot_data_all': self._safe_ak_fetch(ak.stock_zh_a_spot, "A股实时行情"),
            'financial_abstract_raw': self._safe_ak_fetch(ak.stock_financial_abstract, "财务摘要数据"),
            'market_fund_flow_raw': self._safe_ak_fetch(ak.stock_fund_flow_individual, "5日市场资金流向", symbol="5日排行"),
            'market_fund_flow_raw_10': self._safe_ak_fetch(ak.stock_fund_flow_individual, "10日市场资金流向", symbol="10日排行"),
            'market_fund_flow_raw_20': self._safe_ak_fetch(ak.stock_fund_flow_individual, "20日市场资金流向", symbol="20日排行"),
            'strong_stocks_raw': self._safe_ak_fetch(ak.stock_zt_pool_strong_em, "强势股池", date=datetime.now().strftime('%Y%m%d')),
            'consecutive_rise_raw': self._safe_ak_fetch(ak.stock_rank_lxsz_ths, "连续上涨"),
            'ljqs_raw': self._safe_ak_fetch(ak.stock_rank_ljqs_ths, "量价齐升"),
            'cxfl_raw': self._safe_ak_fetch(ak.stock_rank_cxfl_ths, "持续放量"),
        }

        data['xstp_10_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破10日均线", symbol="10日均线")
        data['xstp_30_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破30日均线", symbol="30日均线")
        data['xstp_60_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破60日均线", symbol="60日均线")

        self.logger.info("正在获取行业板块名称并保存至本地...")
        industry_info_filename = f"行业板块信息_{self.today_str}.txt"
        industry_info_path = os.path.join(self.temp_dir, industry_info_filename)
        industry_board_df = pd.DataFrame()

        if os.path.exists(industry_info_path):
            try:
                industry_board_df = pd.read_csv(industry_info_path, sep='|', encoding='utf-8-sig')
            except Exception as e:
                self.logger.warning(f"读取本地缓存失败: {e}，将尝试重新获取...")

        if industry_board_df.empty:
            self.logger.info("本地无有效缓存，正在通过 Akshare 接口获取...")
            try:
                industry_board_df = ak.stock_board_industry_name_em()
                if not industry_board_df.empty:
                    try:
                        industry_board_df.to_csv(industry_info_path, sep='|', index=False, encoding='utf-8-sig')
                        self.logger.info(f"获取成功并已保存至: {industry_info_filename}")
                    except Exception as e:
                        self.logger.error(f"保存文件失败: {e}")
            except Exception as e:
                self.logger.error(f"调用行业板块接口失败: {e}")

        data['top_industry_cons_df'] = self._get_top_industry_constituents(industry_board_df)
        data['industry_board_df'] = industry_board_df
        return data

    def _safe_fetch_constituents(self, symbol: str) -> pd.DataFrame:
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                df = ak.stock_board_industry_cons_em(symbol=symbol)
                if df is not None and not df.empty:
                    return df
                else:
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception:
                time.sleep(self.config.DATA_FETCH_DELAY)
        return pd.DataFrame()

    def _get_top_industry_constituents(self, industry_board_df: pd.DataFrame) -> pd.DataFrame:
        if industry_board_df.empty or '板块名称' not in industry_board_df.columns:
            return pd.DataFrame()

        cache_name = "前十板块成分股"
        cleaned_file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        top_industries = industry_board_df.sort_values(by='涨跌幅', ascending=False).head(10)
        industry_list = top_industries.to_dict('records')

        def fetch_worker(row):
            industry_name = row['板块名称']
            constituents_df = self._safe_fetch_constituents(symbol=industry_name)
            if constituents_df is not None and not constituents_df.empty:
                if '代码' in constituents_df.columns:
                    constituents_df.rename(columns={'代码': '股票代码'}, inplace=True)
                if '股票代码' in constituents_df.columns:
                    constituents_df['股票代码'] = constituents_df['股票代码'].astype(str).zfill(6)
                    constituents_df['所属板块'] = industry_name
                    return constituents_df[['股票代码', '所属板块']].drop_duplicates()
            return None

        results = utils.run_with_thread_pool(
            items=industry_list,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc="获取板块成分股"
        )

        if results:
            final_df = pd.concat(results, ignore_index=True).drop_duplicates(subset=['股票代码'])
            self._save_data_to_cache(final_df, cleaned_file_path)
            return final_df
        return pd.DataFrame()

    def _fetch_hist_data_parallel(self, codes: List[str], days: int) -> pd.DataFrame:
        self.logger.info(f"正在从数据库并行获取 {len(codes)} 只股票的 {days} 天历史数据...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        cache_name = f"MACD_hist_data_cache"
        file_path = self._get_file_path(cache_name, cleaned=True)

        def fetch_worker(symbol_prefixed: str) -> pd.DataFrame:
            try:
                query = text(f"""
                    SELECT trade_date, open, close, high, low, adj_ratio
                    FROM stock_daily_kline
                    WHERE symbol = :symbol
                      AND trade_date BETWEEN :start_date AND :end_date
                    ORDER BY trade_date
                """)
                with self.db_engine.connect() as conn:
                    hist_df = pd.read_sql(query, conn, params={
                        'symbol': symbol_prefixed,
                        'start_date': start_date_str,
                        'end_date': end_date_str
                    })

                if hist_df.empty:
                    return None

                hist_df.rename(columns={
                    'trade_date': 'date',
                    'adj_ratio': 'volume'
                }, inplace=True)
                hist_df['股票代码'] = symbol_prefixed[2:]
                hist_df['date'] = pd.to_datetime(hist_df['date']).dt.strftime('%Y-%m-%d')

                cols_to_keep = ['date', 'open', 'close', 'high', 'low', 'volume', '股票代码']
                return hist_df[[col for col in cols_to_keep if col in hist_df.columns]]
            except Exception as e:
                self.logger.error(f"从数据库获取 {symbol_prefixed} 历史数据失败: {e}")
                return None

        results = utils.run_with_thread_pool(
            items=codes,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc=f"从数据库下载 {days} 天历史数据"
        )

        if results:
            merged_df = pd.concat(results, ignore_index=True)
            self._save_data_to_cache(merged_df, file_path)
            return merged_df
        return pd.DataFrame()

    def _save_ta_signals_to_txt(self, ta_signals: Dict[str, pd.DataFrame]):
        self.logger.info("正在保存技术指标信号到本地 TXT 文件...")
        save_dir = self.config.TEMP_DATA_DIRECTORY
        today_str = self.today_str

        for indicator_name, df in ta_signals.items():
            if df is None or df.empty:
                continue
            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)
            try:
                df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
                self.logger.info(f"成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                self.logger.error(f"保存 {indicator_name} 信号文件失败: {e}")

    def _process_xstp_and_filter(self, raw_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("正在处理并合并均线突破数据...")
        processed_df10 = raw_data['xstp_10_raw'].rename(columns={'最新价': '10日均线价'})
        processed_df30 = raw_data['xstp_30_raw'].rename(columns={'最新价': '30日均线价'})
        processed_df60 = raw_data['xstp_60_raw'].rename(columns={'最新价': '60日均线价'})

        merged_df = pd.concat([
            processed_df10[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df30[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df60[['股票代码', '股票简称']].dropna(subset=['股票代码'])
        ]).drop_duplicates(subset=['股票代码'])

        xstp_base = merged_df[['股票代码', '股票简称']].drop_duplicates()
        xstp_base = pd.merge(xstp_base, processed_df10[['股票代码', '10日均线价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df30[['股票代码', '30日均线价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df60[['股票代码', '60日均线价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, spot_df[['股票代码', '最新价']], on='股票代码', how='left')

        cols_to_convert = [col for col in xstp_base.columns if '最新价' in col or col == '最新价']
        for col in cols_to_convert:
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors='coerce')

        filtered_df = xstp_base[
            (xstp_base['最新价'] > xstp_base['10日均线价']) &
            (
                    (xstp_base['10日均线价'] > xstp_base['30日均线价'].fillna(float('-inf'))) |
                    (xstp_base['30日均线价'] > xstp_base['60日均线价'].fillna(float('-inf')))
            )
        ].copy()

        filtered_df['完全多头排列'] = filtered_df.apply(
            lambda row: '是' if row['10日均线价'] > row['30日均线价'] and row['30日均线价'] > row['60日均线价'] else '否',
            axis=1
        )
        filtered_df.rename(columns={'最新价': '当前价格'}, inplace=True)
        return filtered_df.fillna('N/A')

    def _consolidate_data(self, processed_data: Dict[str, pd.DataFrame], base_stock_codes_pure: List[str]) -> pd.DataFrame:
        self.logger.info("正在汇总所有数据和信号 (技术指标作为独立列)...")

        final_df = pd.DataFrame(base_stock_codes_pure, columns=['股票代码'])
        final_df['股票代码'] = final_df['股票代码'].astype(str)

        spot_df = processed_data.get('spot_data_all', pd.DataFrame())
        file_industry_df = processed_data.get('individual_industry', pd.DataFrame())

        if '股票代码' in spot_df.columns:
            spot_df['股票代码'] = spot_df['股票代码'].astype(str)

        name_source_spot = spot_df[['股票代码', '股票简称']].drop_duplicates(subset=['股票代码']) if '股票简称' in spot_df.columns else pd.DataFrame()
        all_names = pd.concat([name_source_spot, file_industry_df[['股票代码', '股票简称']]]).drop_duplicates(subset=['股票代码'], keep='first')
        final_df = pd.merge(final_df, all_names, on='股票代码', how='left')

        if '股票简称' not in spot_df.columns:
            self.logger.critical("实时行情数据中缺少 '股票简称' 列，无法按要求按简称关联。回退到按代码关联。")
            price_source_key = '股票代码'
            price_source = spot_df[['股票代码', '最新价']].copy()
        else:
            price_source_key = '股票简称'
            price_source = spot_df[['股票简称', '最新价']].copy()
            price_source['最新价'] = pd.to_numeric(price_source['最新价'], errors='coerce')
            price_source = price_source[(price_source['最新价'].notna()) & (price_source['最新价'] > 0)].copy()
            price_source = price_source.drop_duplicates(subset=[price_source_key], keep='first')

        final_df.drop(columns=['最新价'], errors='ignore', inplace=True)

        if price_source_key in final_df.columns:
            final_df[price_source_key] = final_df[price_source_key].astype(str)
            price_source[price_source_key] = price_source[price_source_key].astype(str)
            final_df = pd.merge(final_df, price_source, on=price_source_key, how='left')

        valid_prices_count = final_df['最新价'].notna().sum() if '最新价' in final_df.columns else 0
        self.logger.info(f"实时行情数据 (最新价) 成功通过 '{price_source_key}' 关联的有效价格数量: {valid_prices_count} / {len(final_df)}")

        final_df['股票简称'] = final_df['股票简称'].fillna('N/A')
        final_df['最新价'] = final_df['最新价'].fillna('N/A')

        xstp_df = processed_data['processed_xstp_df']
        xstp_cols = ['股票代码', '完全多头排列', '当前价格', '10日均线价', '30日均线价', '60日均线价']
        if not xstp_df.empty and '股票代码' in xstp_df.columns:
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]
            merge_df = xstp_df[cols_present].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, merge_df, on='股票代码', how='left')

        if '完全多头排列' not in final_df.columns:
            final_df['完全多头排列'] = '否'
        else:
            final_df['完全多头排列'] = final_df['完全多头排列'].fillna('否')

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
                v5 = Parse_Currency.Parse_Currency.parse_money_str(row[f5_col])
                v10 = Parse_Currency.Parse_Currency.parse_money_str(row[f10_col])
                v20 = Parse_Currency.Parse_Currency.parse_money_str(row[f20_col])
                if (v5 > v10 or v5 > v20) and v5 > 0:
                    return "动能增强"
                elif v5 > 0:
                    return "流入"
                else:
                    return ""

            final_df['资金动能'] = final_df.apply(calculate_trend, axis=1)
            cols = list(final_df.columns)
            if '资金动能' in cols:
                target_idx = cols.index(f5_col)
                cols.insert(target_idx + 1, cols.pop(cols.index('资金动能')))
                final_df = final_df[cols]

        if not processed_data['strong_stocks_raw'].empty:
            strong_codes = processed_data['strong_stocks_raw']['股票代码'].tolist()
            final_df['强势股'] = final_df['股票代码'].apply(lambda x: '是' if x in strong_codes else '否')
        else:
            final_df['强势股'] = '否'

        rise_df = processed_data['consecutive_rise_raw']
        if not rise_df.empty:
            rise_df = rise_df[['股票代码', '连涨天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, rise_df, on='股票代码', how='left').fillna({'连涨天数': 0})
        else:
            final_df['连涨天数'] = 0
        final_df['连涨天数'] = final_df['连涨天数'].astype(int)

        if not processed_data['ljqs_raw'].empty:
            ljqs_codes = processed_data['ljqs_raw']['股票代码'].tolist()
            final_df['量价齐升'] = final_df['股票代码'].apply(lambda x: '是' if x in ljqs_codes else '否')
        else:
            final_df['量价齐升'] = '否'

        cxfl_df = processed_data['cxfl_raw']
        if not cxfl_df.empty:
            cxfl_df = cxfl_df[['股票代码', '放量天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, cxfl_df, on='股票代码', how='left').fillna({'放量天数': 0})
        else:
            final_df['放量天数'] = 0
        final_df['放量天数'] = final_df['放量天数'].astype(int)

        ta_dfs_to_merge = []
        macd_df_standard = processed_data.get('MACD_12269', pd.DataFrame())
        if not macd_df_standard.empty:
            ta_dfs_to_merge.append(macd_df_standard[['股票代码', 'MACD_12269_Signal']].rename(columns={'MACD_12269_Signal': 'MACD_12269'}))

        macd_df_fast = processed_data.get('MACD_6135', pd.DataFrame())
        if not macd_df_fast.empty:
            ta_dfs_to_merge.append(macd_df_fast[['股票代码', 'MACD_6135_Signal']].rename(columns={'MACD_6135_Signal': 'MACD_6135'}))

        kdj_df = processed_data.get('KDJ', pd.DataFrame())
        if not kdj_df.empty:
            ta_dfs_to_merge.append(kdj_df[['股票代码', 'KDJ_Signal']].rename(columns={'KDJ_Signal': 'KDJ_Signal'}))

        cci_df = processed_data.get('CCI', pd.DataFrame())
        if not cci_df.empty:
            ta_dfs_to_merge.append(cci_df[['股票代码', 'CCI_Signal']].rename(columns={'CCI_Signal': 'CCI_Signal'}))

        rsi_df = processed_data.get('RSI', pd.DataFrame())
        if not rsi_df.empty:
            rsi_df['RSI_Signal'] = rsi_df['RSI_Signal'].astype(str).str.split(' ').str[0]
            ta_dfs_to_merge.append(rsi_df[['股票代码', 'RSI_Signal']].rename(columns={'RSI_Signal': 'RSI_Signal'}))

        boll_df = processed_data.get('BOLL', pd.DataFrame())
        if not boll_df.empty:
            ta_dfs_to_merge.append(boll_df[['股票代码', 'BOLL_Signal']].rename(columns={'BOLL_Signal': 'BOLL_Signal'}))

        for ta_df in ta_dfs_to_merge:
            final_df = pd.merge(final_df, ta_df.drop_duplicates(subset=['股票代码']), on='股票代码', how='left')

        momentum_df = processed_data.get('MACD_DIF_MOMENTUM', pd.DataFrame())
        if not momentum_df.empty and '股票代码' in momentum_df.columns:
            final_df = pd.merge(final_df, momentum_df, on='股票代码', how='left')
            for col in ['MACD_12269_动能', 'MACD_6135_动能']:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna('')

        for col in ['MACD_12269', 'MACD_6135', 'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal']:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna('')
            else:
                final_df[col] = ''

        top_ind_df = processed_data.get('top_industry_cons_df', pd.DataFrame())
        if not top_ind_df.empty:
            top_codes = set(top_ind_df['股票代码'].astype(str).unique())
            final_df['TOP10行业'] = final_df['股票代码'].apply(lambda x: '是' if str(x) in top_codes else '否')
        else:
            final_df['TOP10行业'] = '否'

        industry_df = processed_data.get('individual_industry', pd.DataFrame())
        if not industry_df.empty and '股票代码' in industry_df.columns and '行业' in industry_df.columns:
            final_df = pd.merge(final_df, industry_df[['股票代码', '行业']], on='股票代码', how='left')
            final_df['行业'] = final_df['行业'].fillna('N/A')
            self.logger.info("行业数据已成功合并到最终报告。")
        else:
            if '行业' not in final_df.columns:
                final_df['行业'] = 'N/A'

        def has_any_signal(row):
            return (row['完全多头排列'] == '是' or
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
        final_df.sort_values(by=['连涨天数', '放量天数'], ascending=[False, False], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        final_df['完整股票代码'] = final_df['股票代码'].apply(format_stock_code)
        final_df['股票链接'] = "https://hybrid.gelonghui.com/stock-check/" + final_df['完整股票代码']
        final_df.drop(columns=['完整股票代码'], inplace=True, errors='ignore')

        if '当前价格' in final_df.columns and '最新价' in final_df.columns:
            final_df.drop(columns=['当前价格'], inplace=True, errors='ignore')

        base_cols = ['股票代码', '股票简称', '行业', '获利比例', '90集中度', '平均成本', '筹码状态']
        if '最新价' in final_df.columns:
            base_cols.insert(3, '最新价')

        signal_cols = [
            '强势股', '量价齐升', '连涨天数', '放量天数', 'TOP10行业',
            'MACD_12269', 'MACD_12269_动能', 'MACD_12269_DIF',
            'MACD_6135', 'MACD_6135_动能', 'MACD_6135_DIF',
            'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal',
        ]
        report_cols = [
            '完全多头排列', '10日均线价', '30日均线价', '60日均线价',
            '资金动能', '5日资金流入', '10日资金流入', '20日资金流入'
        ]
        final_cols = base_cols + signal_cols + report_cols + ['股票链接']
        final_df = final_df[[col for col in final_cols if col in final_df.columns]]

        return final_df

    def _merge_industry_signal_to_stocks(self, stock_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        if industry_df.empty or stock_df.empty or '行业' not in stock_df.columns:
            stock_df['所属行业信号'] = ''
            return stock_df

        self.logger.info("正在将行业信号映射至个股...")
        signal_map = industry_df.set_index('行业名称')['行业信号'].to_dict()
        stock_df['所属行业信号'] = stock_df['行业'].map(signal_map).fillna('')
        return stock_df

    def _generate_report(self, sheets_data: Dict[str, pd.DataFrame]):
        self.logger.info("正在生成 Excel 报告...")
        report_path = os.path.join(self.config.SAVE_DIRECTORY, f"审计报告_{self.today_str}.xlsx")

        try:
            writer = pd.ExcelWriter(report_path, engine='xlsxwriter')
            workbook = writer.book

            header_format = workbook.add_format(
                {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            currency_format = workbook.add_format({'num_format': '#,##0.00'})
            code_format = workbook.add_format({'num_format': '@'})

            for sheet_name, df in sheets_data.items():
                if sheet_name == '主力研报筛选':
                    self.logger.warning(f"工作表 '{sheet_name}' 已根据需求移除。")
                    continue
                if df is None or df.empty:
                    self.logger.warning(f"工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
                worksheet = writer.sheets[sheet_name]

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)
                    if col == '最新价' or '价格' in col or '价' in col or '线' in col or '均线' in col:
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif '代码' in col:
                        worksheet.set_column(i, i, 10, code_format)
                    else:
                        worksheet.set_column(i, i, col_width)

            writer.close()
            self.logger.info(f"报告已成功生成并保存到: {report_path}")

        except Exception as e:
            self.logger.critical(f"致命错误：生成 Excel 报告失败。原因: {e}")
            raise

    def run(self):
        self.logger.info(f"股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.logger.info("\n>>> 调用 HistDataWatchDog 同步股票数据到数据库 (K线池将由研报过滤决定)...")
            self.sync_engine.run_engine()

            synced_codes_df_from_db = pd.DataFrame(columns=['symbol'])

            try:
                if self.db_engine is None:
                    raise RuntimeError("数据库引擎未成功初始化，无法从数据库获取数据。")

                with self.db_engine.connect() as conn:
                    latest_date_query = text("SELECT MAX(trade_date) FROM stock_daily_kline;")
                    latest_db_date_result = conn.execute(latest_date_query).scalar_one_or_none()
                    if latest_db_date_result is None:
                        self.logger.critical("数据库中 'stock_daily_kline' 表没有K线数据，无法获取股票代码列表，流程终止。")
                        return
                    query_symbols = text(f"""
                        SELECT DISTINCT symbol
                        FROM stock_daily_kline
                        WHERE trade_date = :latest_date
                    """)
                    synced_codes_df_from_db = pd.read_sql(query_symbols, conn, params={'latest_date': latest_db_date_result})
                    self.logger.info(f">>> 已从数据库获取 {len(synced_codes_df_from_db)} 只股票代码，基于最新交易日 {latest_db_date_result.strftime('%Y-%m-%d')}。")
            except Exception as e:
                self.logger.critical(f"查询数据库获取股票代码失败: {e}，流程终止。")
                return

            if synced_codes_df_from_db.empty:
                self.logger.critical("从数据库获取已同步股票代码列表失败，流程终止。")
                return

            final_analysis_codes_prefixed = synced_codes_df_from_db['symbol'].tolist()
            final_analysis_codes_pure = [code[2:] for code in final_analysis_codes_prefixed]

            self.logger.info(f">>> HistDataWatchDog 成功同步 {len(final_analysis_codes_prefixed)} 只股票数据到数据库，并作为分析基础。")

            industry_analyzer = industry.IndustryFlowAnalyzer(self.config)
            industry_analysis_df = industry_analyzer.run_analysis()

            raw_data = self._get_all_raw_data()
            processed_main_report = pd.DataFrame()

            if not final_analysis_codes_prefixed:
                self.logger.critical("未找到任何有效的股票代码，流程终止。")
                return

            hist_df_all = self._fetch_hist_data_parallel(final_analysis_codes_prefixed, days=365)

            signal_processor = TASignalProcessor(self)
            ta_signals = signal_processor.process_signals(
                final_analysis_codes_pure,
                hist_df_all,
                raw_data['spot_data_all']
            )
            self._save_ta_signals_to_txt(ta_signals)
            self.logger.info(">>> 股票历史数据和技术指标分析完成。")

            industry_info_df = self._load_industry_info_from_generated_file(final_analysis_codes_pure)

            universe_codes_set_pure = set(final_analysis_codes_pure)

            def filter_df_by_universe(df, universe_set):
                if df is None or df.empty or '股票代码' not in df.columns:
                    return pd.DataFrame()
                df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
                return df[df['股票代码'].isin(universe_set)].copy()

            processed_xstp_df = self._process_xstp_and_filter(raw_data, raw_data['spot_data_all'])
            processed_xstp_df = filter_df_by_universe(processed_xstp_df, universe_codes_set_pure)

            raw_data['market_fund_flow_raw'] = filter_df_by_universe(raw_data['market_fund_flow_raw'], universe_codes_set_pure)
            raw_data['market_fund_flow_raw_10'] = filter_df_by_universe(raw_data['market_fund_flow_raw_10'], universe_codes_set_pure)
            raw_data['market_fund_flow_raw_20'] = filter_df_by_universe(raw_data['market_fund_flow_raw_20'], universe_codes_set_pure)
            raw_data['strong_stocks_raw'] = filter_df_by_universe(raw_data['strong_stocks_raw'], universe_codes_set_pure)
            raw_data['consecutive_rise_raw'] = filter_df_by_universe(raw_data['consecutive_rise_raw'], universe_codes_set_pure)
            raw_data['ljqs_raw'] = filter_df_by_universe(raw_data['ljqs_raw'], universe_codes_set_pure)
            raw_data['cxfl_raw'] = filter_df_by_universe(raw_data['cxfl_raw'], universe_codes_set_pure)

            processed_data = {
                **raw_data,
                **ta_signals,
                'processed_xstp_df': processed_xstp_df,
                'processed_main_report': processed_main_report,
                'individual_industry': industry_info_df
            }

            consolidated_report = self._consolidate_data(processed_data, final_analysis_codes_pure)
            consolidated_report = self._merge_industry_signal_to_stocks(consolidated_report, industry_analysis_df)

            cols = list(consolidated_report.columns)
            if '所属行业信号' in cols and '行业' in cols:
                cols.remove('所属行业信号')
                idx = cols.index('行业')
                cols.insert(idx + 1, '所属行业信号')
                consolidated_report = consolidated_report[cols]

            self.logger.info(">>> 正在执行最终数据清洗：剔除弱势且加速下跌的个股...")

            if not consolidated_report.empty:
                dif_12269 = pd.to_numeric(consolidated_report.get('MACD_12269_DIF'), errors='coerce')
                dif_6135 = pd.to_numeric(consolidated_report.get('MACD_6135_DIF'), errors='coerce')
                kdj_col = consolidated_report.get('KDJ_Signal', pd.Series([''] * len(consolidated_report), index=consolidated_report.index))
                kdj_is_empty = kdj_col.isna() | (kdj_col.astype(str).str.strip().str.lower().isin(['', 'nan', 'none']))

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
                        (consolidated_report.get('5日资金流入', pd.Series(dtype=str)).astype(str).str.contains('-', na=False))
                )

                initial_count = len(consolidated_report)
                consolidated_report = consolidated_report[~drop_condition].copy()
                dropped_count = initial_count - len(consolidated_report)
                self.logger.info(f"  - 清洗完成：共排除了 {dropped_count} 只符合极度弱势特征的股票。剩余 {len(consolidated_report)} 只。")

            if not consolidated_report.empty:
                self.logger.info("\n>>> 正在为最终保留的个股获取筹码分布数据...")
                final_codes_for_chip = consolidated_report['股票代码'].unique().tolist()
                chip_file_name = f"筹码分布数据_精选后_{self.today_str}.txt"
                chip_file_path = os.path.join(self.temp_dir, chip_file_name)
                chip_data_df = pd.DataFrame()

                if os.path.exists(chip_file_path):
                    try:
                        chip_data_df = pd.read_csv(chip_file_path, sep='|', encoding='utf-8-sig', dtype={'股票代码': str})
                    except Exception as e:
                        self.logger.warning(f"读取筹码分布缓存失败: {e}，将尝试重新获取...")

                if chip_data_df.empty:
                    chip_analyzer = dist.ChipDistributionAnalyzer(self.config)
                    chip_data_df = chip_analyzer.fetch_chip_data_parallel(final_codes_for_chip)
                    if not chip_data_df.empty:
                        try:
                            chip_data_df.to_csv(chip_file_path, sep='|', index=False, encoding='utf-8-sig')
                        except Exception as e:
                            self.logger.error(f"保存筹码分布缓存失败: {e}")

                if not chip_data_df.empty:
                    consolidated_report = pd.merge(consolidated_report, chip_data_df, on='股票代码', how='left')
                    base_cols = ['股票代码', '股票简称', '行业', '所属行业信号', '最新价', '获利比例', '90集中度', '平均成本', '筹码状态']
                    other_cols = [c for c in consolidated_report.columns if c not in base_cols and c != '股票链接']
                    final_cols = [c for c in base_cols if c in consolidated_report.columns] + other_cols + ['股票链接']
                    consolidated_report = consolidated_report[[c for c in final_cols if c in consolidated_report.columns]]

            sheets_data = {
                '数据汇总': consolidated_report,
                '行业深度分析': industry_analysis_df,
                '均线多头排列': processed_xstp_df,
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
                self.logger.info("数据库同步成功完成。")
            except Exception as e:
                self.logger.error(f"!!! [同步中断] 任务运行异常: {e}")

        except Exception as e:
            self.logger.critical(f"\n[FATAL] 致命错误：数据分析流程意外终止。原因: {e}")
            raise

        finally:
            end_time = time.time()
            self.logger.info(f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}")


if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()
