import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List
import akshare as ak
import pandas as pd
import pandas_ta as ta  # 勿删
from sqlalchemy import text, create_engine
import Distribution as dist
import Industrytrending as industry
from DataManager import DatabaseWriter
from DataManager import ParallelUtils as utils
from DataManager import QuantDataPerformer
from FormatManager import Parse_Currency
from SignalManager import TASignalProcessor
from HistDataEngine import StockSyncEngine
from LoggerManager import LoggerManager
import os
import configparser
from pathlib import Path
from ConfigParser import Config
from FormatManager.ShareCodeFormatMgr import format_stock_code


class StockAnalyzer:

    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = Config(config_file=config_file)
        self.today_str = datetime.now().strftime("%Y%m%d")
        self.temp_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.temp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.start_time = time.time()
        self.logger = LoggerManager(
            log_dir=self.config.LOG_DIR,
            log_filename=f"Corenews_Main_{self.today_str}.log",
            level=self.config.LOG_LEVEL
        )

        try:
            self.sync_engine = StockSyncEngine()
            self.db_engine = self.sync_engine.db
        except Exception as e:
            self.logger.critical(f"[CRITICAL] Corenews_Main: Failed to initialize StockSyncEngine or its database engine. Error: {e}")
            raise

    def _get_file_path(self, base_name: str, cleaned: bool = False) -> str:
        """
        生成临时数据文件的完整路径。如果 cleaned=True, 则添加 "_经清洗" 后缀。
        """
        suffix = "_经清洗" if cleaned else ""
        file_name = f"{base_name}{suffix}_{self.today_str}.txt"
        return os.path.join(self.temp_dir, file_name)

    def _load_data_from_cache(self, file_path: str) -> pd.DataFrame:
        """从缓存加载数据。"""
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='|', encoding='utf-8', dtype={'股票代码': str, 'symbol': str})
                # 统一列名为 '股票代码'
                if 'symbol' in df.columns and '股票代码' not in df.columns:
                    df.rename(columns={'symbol': '股票代码'}, inplace=True)
                print(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
                return df
            except Exception as e:
                self.logger.warning(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
        return pd.DataFrame()

    def _save_data_to_cache(self, df: pd.DataFrame, file_path: str):
        """保存数据到缓存。"""
        try:
            df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
        except Exception as e:
            self.logger.error(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")

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
                    break
                else:
                    self.logger.warning(f"[WARN] 数据返回为空或无效: {file_base_name}，重试中。")
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception as e:
                self.logger.error(f"[ERROR] 获取 {file_base_name} 时出错: {e}，将在 {self.config.DATA_FETCH_DELAY} 秒后重试。")
                time.sleep(self.config.DATA_FETCH_DELAY)

        if df.empty:
            self.logger.critical(f"[FATAL] 所有重试均失败，返回空 DataFrame: {file_base_name}")
            return pd.DataFrame()

        # 3. 清洗数据并保存到带有 "_经清洗" 后缀的缓存文件
        cleaned_df = self._clean_and_standardize(df, file_base_name)
        if not cleaned_df.empty:
            self._save_data_to_cache(cleaned_df, cleaned_file_path)

        return cleaned_df

    def _clean_and_standardize(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        通用数据清洗和列名标准化。

        """
        if df.empty:
            return df


        if df_name == "financial_abstract_raw":
            print(f" - [INFO] 检测到特殊数据源 {df_name}，应用宽表清洗策略。")

            return df


        def extract_pure_code(code_str):
            """提取纯数字股票代码（去掉 sh/sz/bj 前缀，补齐6位）"""
            if pd.isna(code_str):
                return None
            code_str = str(code_str).strip().upper()
            # 去掉常见前缀
            for prefix in ['SH', 'SZ', 'BJ']:
                if code_str.startswith(prefix):
                    code_str = code_str[2:]
                    break
            # 补齐6位
            return code_str.zfill(6)

        # 1. 列名映射 (通用别名处理)
        # 代码列别名映射
        for old, new in self.config.CODE_ALIASES.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        # 名称列别名映射
        for old, new in self.config.NAME_ALIASES.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        # 价格列别名映射
        for old, new in self.config.PRICE_ALIASES.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)


        # 如果数据中已经有'股票代码'，则格式化它
        if '股票代码' in df.columns:
            df['股票代码'] = df['股票代码'].astype(str).apply(extract_pure_code)
        else:
            # 否则尝试从 code/ts_code/symbol 生成
            code_col = None
            for col in df.columns:
                if col.lower() in ['code', 'ts_code', 'symbol']:
                    code_col = col
                    break

            if code_col:
                df['股票代码'] = df[code_col].astype(str).apply(extract_pure_code)
                print(f"[INFO] 已从 '{code_col}' 生成 '股票代码' 列。")
            else:
                # 对于非财务宽表的其他数据源，如果还没有代码列，则报错返回空
                print(f"[ERROR] {df_name} 无 'code'、'ts_code' 或 'symbol' 字段！列名：{df.columns.tolist()}")
                return pd.DataFrame()

        # 3. ✅ 确保 '最新价' 存在
        if '最新价' not in df.columns:
            if 'price' in df.columns:
                df.rename(columns={'price': '最新价'}, inplace=True)
                print(f"[INFO] 已从 'price' 生成 '最新价' 列。")
            elif 'close' in df.columns:
                df.rename(columns={'close': '最新价'}, inplace=True)
            else:
                print(f"[ERROR] {df_name} 无 'price'/'close' 字段！列名：{df.columns.tolist()}")
                df['最新价'] = 0.0  # 默认值兜底

        # 4. ✅ 确保 '股票简称' 存在
        if '股票简称' not in df.columns:
            if 'name' in df.columns:
                df.rename(columns={'name': '股票简称'}, inplace=True)
            elif '简称' in df.columns:
                df.rename(columns={'简称': '股票简称'}, inplace=True)
            elif 'symbol' in df.columns:  # 防止没有名称时用代码代替
                df.rename(columns={'symbol': '股票简称'}, inplace=True)
            else:
                print(f"[WARN] {df_name} 无简称列，使用 'N/A' 作为占位符。")
                df['股票简称'] = 'N/A'

        if '股票简称' in df.columns:
            # 改进正则：支持 *ST、ST、(ST)、*ST(退市)、ST*、sT 等写法
            st_pattern = r'(?:\s*)(?:[Ss][Tt])|(?:\s*(?:\*|★|★|※|•|·))?(?:[Ss][Tt])'
            mask = df['股票简称'].astype(str).str.contains(
                st_pattern,
                case=False,
                regex=True,
                na=False
            )
            st_count = mask.sum()
            if st_count > 0:
                df = df[~mask].copy()
                print(f"[FILTER] 已过滤 {st_count} 只含 'ST'、'*ST' 的股票。")
        else:
            print(f"[WARN] {df_name} 中 '股票简称' 不存在，跳过ST过滤。")

        # 6. ✅ 最终清洗
        df.dropna(subset=['股票代码'], inplace=True)
        # 去重（保留第一次出现的）
        df.drop_duplicates(subset=['股票代码'], keep='first', inplace=True)
        # 再次确保股票代码是纯数字6位格式
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)

        return df


    def _load_industry_info_from_generated_file(self, codes_pure_digits: List[str]) -> pd.DataFrame:

        from pathlib import Path
        file_name = f"StockIndes_{self.today_str}.txt"
        dict_file_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, file_name)

        if not os.path.exists(dict_file_path):
            print(f" [ERROR] 文件未找到！请检查路径是否正确。")
            return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])


        try:
            print(f"正在读取: {file_name}")
            industry_df_raw = pd.read_csv(
                dict_file_path,
                sep='|',
                encoding='utf-8-sig',
                dtype={
                    'ts_code': str,
                    'symbol': str,
                    'name': str,
                    'industry': str,
                    '股票简称': str,
                    '行业': str,
                    'market': str
                }
            )
            print(f"成功读取 {len(industry_df_raw)} 行行业数据。")

        except Exception as e:
            print(f"[ERROR] 读取 CSV 文件失败: {e}")
            return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

        cols_to_keep = {
            'ts_code': 'ts_code',
            'symbol': 'symbol',
            'name': 'name',
            '股票简称': '股票简称',
            '行业': '行业',
            'market': 'market'
        }

        use_columns = [col for col in cols_to_keep if col in industry_df_raw.columns]
        if not use_columns:
            print(f"❌ [ERROR] 行业数据文件中无可用列（仅含 {list(industry_df_raw.columns)}）")
            return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

        industry_df_cleaned = industry_df_raw[use_columns].copy()


        def to_pure_code(code_str):
            if pd.isna(code_str):
                return None
            code_str = str(code_str).strip()
            # 1. 先处理可能带 '.' 的后缀 (如 000001.SZ)
            code_str = code_str.split('.')[0]
            # 2. 去除可能的前缀 (如 sh600011 -> 600011)
            for prefix in ['sh', 'sz', 'bj', 'SH', 'SZ', 'BJ']:
                if code_str.startswith(prefix):
                    code_str = code_str[2:]
                    break
            return code_str.zfill(6)

        # 🔍 确保 ts_code 是 str 类型，同时能处理 .SZ/.SH
        if 'ts_code' in industry_df_cleaned.columns:
            industry_df_cleaned['股票代码'] = industry_df_cleaned['ts_code'].astype(str).apply(to_pure_code)
        elif 'symbol' in industry_df_cleaned.columns:
            industry_df_cleaned['股票代码'] = industry_df_cleaned['symbol'].astype(str).apply(to_pure_code)
        else:
            print(f"❌ [ERROR] 文件中无 'ts_code' 或 'symbol' 列。可用列: {list(industry_df_cleaned.columns)}")
            return pd.DataFrame(columns=['股票代码', '行业', '股票简称'])

        # ✅ 步骤 6：规范化行业和简称列名
        if 'industry' in industry_df_cleaned.columns:
            industry_df_cleaned.rename(columns={'industry': '行业'}, inplace=True)
        elif '行业' not in industry_df_cleaned.columns:
            industry_df_cleaned['行业'] = 'N/A'
        if 'name' in industry_df_cleaned.columns:
            industry_df_cleaned.rename(columns={'name': '股票简称'}, inplace=True)
        elif '股票简称' not in industry_df_cleaned.columns:
            industry_df_cleaned['股票简称'] = 'N/A'

        # ✅ 步骤 7：去除无效行，确保股票代码合法
        industry_df_cleaned = industry_df_cleaned[
            industry_df_cleaned['股票代码'].notnull()
        ].copy()

        # ✅ 步骤 8：只保留进分析池中的股票
        input_df_codes = pd.DataFrame(codes_pure_digits, columns=['股票代码'])
        input_df_codes['股票代码'] = input_df_codes['股票代码'].astype(str).str.zfill(6)
        final_industry_df = pd.merge(input_df_codes, industry_df_cleaned, on='股票代码', how='left')

        # 补充缺失信息
        final_industry_df['行业'] = final_industry_df['行业'].fillna('N/A')
        final_industry_df['股票简称'] = final_industry_df['股票简称'].fillna('N/A')

        # ✅ 步骤 9：输出统计信息
        match_count = final_industry_df[final_industry_df['行业'] != 'N/A'].shape[0]
        print(f"✅ 行业数据加载完成：总共 {len(codes_pure_digits)} 只股票，成功匹配 {match_count} 只。")

        return final_industry_df

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """集中获取所有数据源 (包括主力研报盈利预测)，并支持缓存机制"""
        print("\n>>> 正在初始化数据获取和缓存检查...")

        data = {
            'spot_data_all': self._safe_ak_fetch(ak.stock_zh_a_spot, "A股实时行情"),
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

        # 均线突破数据 (Akshare接口参数不同，需分开获取)
        data['xstp_10_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破10日均线", symbol="10日均线")
        data['xstp_30_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破30日均线", symbol="30日均线")
        data['xstp_60_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破60日均线", symbol="60日均线")

        # 行业板块数据
        print("\n>>> 正在获取行业板块名称并保存至本地...")
        industry_info_filename = f"行业板块信息_{self.today_str}.txt"
        industry_info_path = os.path.join(self.temp_dir, industry_info_filename)
        industry_board_df = pd.DataFrame()

        if os.path.exists(industry_info_path):
            try:
                print(f"  - 发现本地缓存文件，正在读取: {industry_info_filename}")
                industry_board_df = pd.read_csv(industry_info_path, sep='|', encoding='utf-8-sig')
            except Exception as e:
                self.logger.warning(f"  - [WARN] 读取本地缓存失败: {e}，将尝试重新获取...")
        else:
            print(f"  - 本地无有效缓存，正在通过 Akshare 接口获取...")
            try:
                industry_board_df = ak.stock_board_industry_name_em()
                if not industry_board_df.empty:
                    try:
                        industry_board_df.to_csv(industry_info_path, sep='|', index=False, encoding='utf-8-sig')
                        print(f"  - 获取成功并已保存至: {industry_info_filename}")
                    except Exception as e:
                        self.logger.error(f"  - [ERROR] 保存文件失败: {e}")
            except Exception as e:
                self.logger.error(f"  - [ERROR] 调用行业板块接口失败: {e}")

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

            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)

            try:
                df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
                print(f"  - 成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                self.logger.error(f"[ERROR] 保存 {indicator_name} 信号文件失败: {e}")

    def _process_xstp_and_filter(self, raw_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> pd.DataFrame:
        """处理并合并均线突破数据，并进行多头排列筛选。"""
        print("正在处理并合并均线突破数据...")

        # 1. 清洗均线数据
        processed_df10 = raw_data['xstp_10_raw'].rename(columns={'最新价': '10日均线价'})
        processed_df30 = raw_data['xstp_30_raw'].rename(columns={'最新价': '30日均线价'})
        processed_df60 = raw_data['xstp_60_raw'].rename(columns={'最新价': '60日均线价'})

        # 2. 合并
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
            lambda row: '是' if row['10日均线价'] > row['30日均线价'] and row['30日均线价'] > row[
                '60日均线价'] else '否',
            axis=1
        )

        filtered_df.rename(columns={'最新价': '当前价格'}, inplace=True)
        return filtered_df.fillna('N/A')

    def _consolidate_data(self, processed_data: Dict[str, pd.DataFrame],
                          base_stock_codes_pure: List[str]) -> pd.DataFrame:
        """
        合并所有数据源和信号，生成最终汇总报告。
        参数 base_stock_codes_pure 是最终报告的基准股票代码列表（纯数字）。
        """
        print("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")

        final_df = pd.DataFrame(base_stock_codes_pure, columns=['股票代码'])
        final_df['股票代码'] = final_df['股票代码'].astype(str)

        spot_df = processed_data.get('spot_data_all', pd.DataFrame())
        file_industry_df = processed_data.get('individual_industry', pd.DataFrame())

        if '股票代码' in spot_df.columns:
            spot_df['股票代码'] = spot_df['股票代码'].astype(str)

        name_source_spot = spot_df[['股票代码', '股票简称']].drop_duplicates(
            subset=['股票代码']) if '股票简称' in spot_df.columns else pd.DataFrame()

        all_names = pd.concat([name_source_spot, file_industry_df[['股票代码', '股票简称']]]).drop_duplicates(
            # 调整 concat
            subset=['股票代码'], keep='first')
        final_df = pd.merge(final_df, all_names, on='股票代码', how='left')

        if '股票简称' not in spot_df.columns:
            self.logger.critical("[FATAL] 实时行情数据中缺少 '股票简称' 列，无法按要求按简称关联。回退到按代码关联。")
            price_source_key = '股票代码'
            price_source = spot_df[['股票代码', '最新价']].copy()
        else:
            price_source_key = '股票简称'

            price_source = spot_df[['股票代码', '股票简称', '最新价']].copy()
            price_source['最新价'] = pd.to_numeric(price_source['最新价'], errors='coerce')
            price_source = price_source[
                (price_source['最新价'].notna()) &
                (price_source['最新价'] > 0)
                ].copy()

            price_source = price_source.drop_duplicates(subset=[price_source_key], keep='first')

        final_df.drop(columns=['最新价'], errors='ignore', inplace=True)


        if '股票代码' in price_source.columns and '股票代码' in final_df.columns:

            def extract_pure_code(code_str):
                code_str = str(code_str).strip().upper()
                # 去掉常见前缀
                for prefix in ['SH', 'SZ', 'BJ', 'sh', 'sz', 'bj']:
                    if code_str.startswith(prefix):
                        code_str = code_str[2:]
                        break
                # 补齐6位
                return code_str.zfill(6)

            price_source['股票代码_纯'] = price_source['股票代码'].apply(extract_pure_code)
            final_df['股票代码'] = final_df['股票代码'].astype(str).str.zfill(6)

            # 按纯数字代码合并
            final_df = pd.merge(final_df, price_source[['股票代码_纯', '最新价']],
                                left_on='股票代码', right_on='股票代码_纯', how='left')
            final_df.drop(columns=['股票代码_纯'], errors='ignore', inplace=True)

        else:
            print("[WARN] '股票代码' 不存在于价格源中，跳过最新价合并。")

        valid_prices_count = final_df['最新价'].notna().sum() if '最新价' in final_df.columns else 0
        print(
            f"  - 实时行情数据 (最新价) 成功通过 '{price_source_key}' 关联的有效价格数量: {valid_prices_count} / {len(final_df)}")

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
            ta_dfs_to_merge.append(macd_df_standard[['股票代码', 'MACD_12269_Signal']].rename(
                columns={'MACD_12269_Signal': 'MACD_12269'}))

        macd_df_fast = processed_data.get('MACD_6135', pd.DataFrame())
        if not macd_df_fast.empty:
            ta_dfs_to_merge.append(macd_df_fast[['股票代码', 'MACD_6135_Signal']].rename(
                columns={'MACD_6135_Signal': 'MACD_6135'}))

        kdj_df = processed_data.get('KDJ', pd.DataFrame())
        if not kdj_df.empty:
            ta_dfs_to_merge.append(kdj_df[['股票代码', 'KDJ_Signal']].rename(
                columns={'KDJ_Signal': 'KDJ_Signal'}))

        cci_df = processed_data.get('CCI', pd.DataFrame())
        if not cci_df.empty:
            ta_dfs_to_merge.append(cci_df[['股票代码', 'CCI_Signal']].rename(
                columns={'CCI_Signal': 'CCI_Signal'}))

        rsi_df = processed_data.get('RSI', pd.DataFrame())
        if not rsi_df.empty:
            rsi_df['RSI_Signal'] = rsi_df['RSI_Signal'].astype(str).str.split(' ').str[0]
            ta_dfs_to_merge.append(rsi_df[['股票代码', 'RSI_Signal']].rename(
                columns={'RSI_Signal': 'RSI_Signal'}))

        boll_df = processed_data.get('BOLL', pd.DataFrame())
        if not boll_df.empty:
            ta_dfs_to_merge.append(boll_df[['股票代码', 'BOLL_Signal']].rename(
                columns={'BOLL_Signal': 'BOLL_Signal'}))

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
        if not industry_df.empty:
            # 确保 industry_df 包含 '股票代码' 和 '行业' 列再合并
            if '股票代码' in industry_df.columns and '行业' in industry_df.columns:
                final_df = pd.merge(final_df, industry_df[['股票代码', '行业']], on='股票代码', how='left')  # 只合并相关列
                final_df['行业'] = final_df['行业'].fillna('N/A')
                print(f"  - 行业数据已成功合并到最终报告。")
            else:
                self.logger.warning(f"[WARN] 从 processed_data 获取的行业数据缺少 '股票代码' 或 '行业' 列，跳过合并。")
                if '行业' not in final_df.columns:  # 确保即使跳过合并，列也存在
                    final_df['行业'] = 'N/A'
        else:
            self.logger.info("[INFO] 从 processed_data 获取的行业数据为空，跳过合并。")
            if '行业' not in final_df.columns:  # 确保即使数据为空，列也存在
                final_df['行业'] = 'N/A'


        def has_any_signal(row):
            # 研报买入次数条件已移除
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

        # 这里传入纯数字的股票代码，format_stock_code 会自动添加前缀
        final_df['完整股票代码'] = final_df['股票代码'].apply(format_stock_code)
        final_df['股票链接'] = "https://hybrid.gelonghui.com/stock-check/" + final_df['完整股票代码']

        final_df.drop(columns=['完整股票代码'], inplace=True, errors='ignore')

        if '当前价格' in final_df.columns and '最新价' in final_df.columns:
            final_df.drop(columns=['当前价格'], inplace=True, errors='ignore')

        base_cols = ['股票代码', '股票简称', '行业', '获利比例', '90集中度', '平均成本', '筹码状态']  # 移除所属行业信号
        if '最新价' in final_df.columns:  # 如果最新价存在，确保它在 base_cols 中
            base_cols.insert(3, '最新价')

        signal_cols = [
            '强势股', '量价齐升', '连涨天数', '放量天数', 'TOP10行业',
            'MACD_12269', 'MACD_12269_动能', 'MACD_12269_DIF',
            'MACD_6135', 'MACD_6135_动能', 'MACD_6135_DIF',
            'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal',
        ]
        report_cols = [
            '研报买入次数',
            '完全多头排列', '10日均线价', '30日均线价', '60日均线价',
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
        report_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, f"审计报告_{self.today_str}.xlsx")

        try:
            writer = pd.ExcelWriter(report_path, engine='xlsxwriter')
            workbook = writer.book

            header_format = workbook.add_format(
                {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            currency_format = workbook.add_format({'num_format': '#,##0.00'})
            code_format = workbook.add_format({'num_format': '@'})

            for sheet_name, df in sheets_data.items():

                if df is None or df.empty:
                    print(f"  - 警告：工作表 '{sheet_name}' 数据为空，跳过创建。")
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
            print(f"  - 报告已成功生成并保存到: {report_path}")

        except Exception as e:
            self.logger.critical(f"[FATAL] 致命错误：生成 Excel 报告失败。原因: {e}")
            raise

    def run(self):

        print(f"股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:

            self.sync_engine.run_engine()

            synced_codes_df_from_db = pd.DataFrame(columns=['symbol'])  # 初始化为空，以防查询失败

            try:
                # 确保 self.db_engine 已经被成功初始化
                if self.db_engine is None:
                    raise RuntimeError("数据库引擎未成功初始化，无法从数据库获取数据。")

                with self.db_engine.connect() as conn:
                    # 1. 查询数据库中最新的一个交易日期
                    latest_date_query = text("SELECT MAX(trade_date) FROM stock_daily_kline;")
                    latest_db_date_result = conn.execute(latest_date_query).scalar_one_or_none()
                    if latest_db_date_result is None:
                        self.logger.critical("[FATAL] 数据库中 'stock_daily_kline' 表没有K线数据，无法获取股票代码列表，流程终止。")
                        return
                    # 2. 查询在该最新交易日期有数据的股票代码
                    query_symbols = text(f"""
                                    SELECT DISTINCT symbol
                                    FROM stock_daily_kline
                                    WHERE trade_date = :latest_date
                                """)
                    synced_codes_df_from_db = pd.read_sql(query_symbols, conn,
                                                          params={'latest_date': latest_db_date_result})
                    print(
                        f">>> 已从数据库获取 {len(synced_codes_df_from_db)} 只股票代码，基于最新交易日  ")
            except Exception as e:
                self.logger.critical(f"[FATAL] 查询数据库获取股票代码失败: {e}，流程终止。")
                return  # 异常时也终止流程

            if synced_codes_df_from_db.empty:
                self.logger.critical("[FATAL] 从数据库获取已同步股票代码列表失败，流程终止。")
                return

            final_analysis_codes_prefixed = synced_codes_df_from_db['symbol'].tolist()

            final_analysis_codes_pure = [code[2:] for code in final_analysis_codes_prefixed]

            print(
                f">>> HistDataWatchDog 成功同步 {len(final_analysis_codes_prefixed)} 只股票数据到数据库，并作为分析基础。")

            # 预处理行业权重数据
            industry_analyzer = industry.IndustryFlowAnalyzer(self.config)
            industry_analysis_df = industry_analyzer.run_analysis()
            raw_data = self._get_all_raw_data()
            processed_main_report = pd.DataFrame()


            # 构造查询语句
            if not final_analysis_codes_prefixed:
                print("[WARN] 待分析股票代码列表为空，跳过历史数据查询。")
                hist_df_all = pd.DataFrame()
            else:
                # 构造 IN 子句
                symbols_str = ','.join([f"'{s}'" for s in final_analysis_codes_prefixed])
                query = text(f"""
                    SELECT *
                    FROM stock_daily_kline
                    WHERE symbol IN ({symbols_str})
                    ORDER BY trade_date
                """)

                hist_df_all = pd.DataFrame()  # 初始化为空
                try:
                    with self.db_engine.connect() as conn:

                        hist_df_all = pd.read_sql(query, conn)

                        if not hist_df_all.empty:
                            print(
                                f"[DEBUG] 数据日期范围: {hist_df_all['trade_date'].min()} 至 {hist_df_all['trade_date'].max()}")
                        else:
                            print("[ERROR] 查询结果为空！可能是股票代码不匹配或日期条件过滤了所有数据。")

                except Exception as e:
                    # except 必须紧贴 try 块
                    print(f"[ERROR] 数据库查询失败: {e}")
                    hist_df_all = pd.DataFrame()


            if hist_df_all.empty:
                print("[WARN] 由于历史数据为空，将跳过所有技术指标计算。")
                # 这里可能需要处理空数据的情况，防止后续报错
            else:
                # 正常调用信号处理
                pass

            signal_processor = TASignalProcessor(self)
            ta_signals = signal_processor.process_signals(

                final_analysis_codes_prefixed,
                hist_df_all,
                raw_data['spot_data_all']
            )
            self._save_ta_signals_to_txt(ta_signals)
            print(">>> 股票历史数据和技术指标分析完成。")

            # 行业信息获取，注意这里需要纯数字的代码
            industry_info_df = self._load_industry_info_from_generated_file(final_analysis_codes_pure)
            universe_codes_set_pure = set(final_analysis_codes_pure)

            def filter_df_by_universe(df, universe_set):
                if df is None or df.empty or '股票代码' not in df.columns:
                    return pd.DataFrame()
                df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
                return df[df['股票代码'].isin(universe_set)].copy()

            # 均线突破数据处理
            processed_xstp_df = self._process_xstp_and_filter(raw_data, raw_data['spot_data_all'])
            processed_xstp_df = filter_df_by_universe(processed_xstp_df, universe_codes_set_pure)

            # 过滤其他每日排名数据
            raw_data['market_fund_flow_raw'] = filter_df_by_universe(raw_data['market_fund_flow_raw'],
                                                                     universe_codes_set_pure)
            raw_data['market_fund_flow_raw_10'] = filter_df_by_universe(raw_data['market_fund_flow_raw_10'],
                                                                        universe_codes_set_pure)
            raw_data['market_fund_flow_raw_20'] = filter_df_by_universe(raw_data['market_fund_flow_raw_20'],
                                                                        universe_codes_set_pure)
            raw_data['strong_stocks_raw'] = filter_df_by_universe(raw_data['strong_stocks_raw'],
                                                                  universe_codes_set_pure)
            raw_data['consecutive_rise_raw'] = filter_df_by_universe(raw_data['consecutive_rise_raw'],
                                                                     universe_codes_set_pure)
            raw_data['ljqs_raw'] = filter_df_by_universe(raw_data['ljqs_raw'], universe_codes_set_pure)
            raw_data['cxfl_raw'] = filter_df_by_universe(raw_data['cxfl_raw'], universe_codes_set_pure)

            # 5. 合并所有数据源和信号
            processed_data = {
                **raw_data,
                **ta_signals,
                'processed_xstp_df': processed_xstp_df,
                'processed_main_report': processed_main_report,  # 此时为空DataFrame
                'individual_industry': industry_info_df
            }

            # 调用 _consolidate_data 时，传入基础的纯数字股票代码列表
            consolidated_report = self._consolidate_data(processed_data, final_analysis_codes_pure)
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
                kdj_col = consolidated_report.get('KDJ_Signal',
                                                  pd.Series([''] * len(consolidated_report),
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

                initial_count = len(consolidated_report)
                consolidated_report = consolidated_report[~drop_condition].copy()
                dropped_count = initial_count - len(consolidated_report)
                print(
                    f" 排除极度弱势特征的股票。剩余 {len(consolidated_report)} 只。")

            if not consolidated_report.empty:
                print("\n>>> 正在为最终保留的个股获取筹码分布数据...")
                # 提取最终保留的股票代码 (此时是纯数字代码)
                final_codes_for_chip = consolidated_report['股票代码'].unique().tolist()

                chip_file_name = f"筹码分布数据_精选后_{self.today_str}.txt"
                chip_file_path = os.path.join(self.temp_dir, chip_file_name)
                chip_data_df = pd.DataFrame()

                if os.path.exists(chip_file_path):
                    try:
                        print(f"  - 发现本地筹码分布缓存，正在读取: {chip_file_name}")
                        chip_data_df = pd.read_csv(chip_file_path, sep='|', encoding='utf-8-sig',
                                                   dtype={'股票代码': str})
                    except Exception as e:
                        self.logger.warning(f"  - [WARN] 读取筹码分布缓存失败: {e}，将尝试重新获取...")

                if chip_data_df.empty:
                    chip_analyzer = dist.ChipDistributionAnalyzer(self.config)
                    chip_data_df = chip_analyzer.fetch_chip_data_parallel(final_codes_for_chip)

                    if not chip_data_df.empty:
                        try:
                            chip_data_df.to_csv(chip_file_path, sep='|', index=False, encoding='utf-8-sig')
                        except Exception as e:
                            self.logger.error(f"  - [ERROR] 保存筹码分布缓存失败: {e}")

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
                    user=self.config.DB_USER,
                    password=self.config.DB_PASSWORD,
                    host=self.config.DB_HOST,
                    port=self.config.DB_PORT,
                    db_name=self.config.DB_NAME
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
                self.logger.error(f"!!! [同步中断] 任务运行异常: {e}")

        except Exception as e:
            self.logger.critical(f"\n[FATAL] 致命错误：数据分析流程意外终止。原因: {e}")
            raise

        finally:
            end_time = time.time()
            print(f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}")

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()
