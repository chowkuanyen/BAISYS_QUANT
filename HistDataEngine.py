# HistDataEngine.py

import os
import akshare as ak
import pandas as pd
import datetime
import time
from sqlalchemy import create_engine, text
# 如果未安装 psycopg2，可能需要安装: pip install psycopg2-binary
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import List, Set, Optional


class StockSyncEngine:
    DB_URL = "postgresql+psycopg2://postgres:025874yan@127.0.0.1:5432/Corenews"
    AKSHARE_RETRIES = 3
    AKSHARE_DELAY = 5

    def __init__(self, db_url=DB_URL, base_data_dir=None):
        self.token = "f4422b90a91c02d7dc68dd24f066988064d7307790f200243822cac3"  # 更新为你的 Tushare Token

        # ✅ 修复：初始化数据库引擎！这是你缺失的关键一步！
        self.db = create_engine(
            db_url,
            pool_pre_ping=True,      # 自动检测连接是否存活
            pool_recycle=3600,       # 每小时回收连接
            echo=False               # 生产环境设为 False
        )

        # ✅ 测试连接（可选但强烈推荐）
        try:
            with self.db.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[INFO] ✅ 数据库引擎初始化成功，连接测试通过。")
        except Exception as e:
            print(f"[CRITICAL] ❌ 数据库连接失败！请检查：\n"
                  f"  - URL: {db_url}\n"
                  f"  - PostgreSQL 是否运行？\n"
                  f"  - 数据库 'Corenews' 是否存在？\n"
                  f"  - 用户名/密码是否正确？\n"
                  f"  - 是否安装了 psycopg2-binary？pip install psycopg2-binary\n"
                  f"  - 错误详情: {type(e).__name__}: {e}")
            raise RuntimeError("数据库引擎初始化失败，程序无法继续。") from e

        self.global_start = "20250101"  # 通常需要一个过去的日期来获取历史数据
        self.today = datetime.datetime.now().strftime("%Y%m%d")
        self.today_dt = pd.to_datetime(self.today).normalize()

        self.base_data_dir = base_data_dir if base_data_dir else os.path.join(os.path.expanduser('~'), 'Downloads',
                                                                              'CoreNews_Reports', 'ShareData')
        os.makedirs(self.base_data_dir, exist_ok=True)  # 确保该目录存在

        self.report_filtered_symbols_cache_path = os.path.join(self.base_data_dir,
                                                               f"report_qualified_pure_symbols_{self.today}.json")
        self.raw_report_cache_path = os.path.join(self.base_data_dir, f"主力研报盈利预测_经清洗_{self.today}.txt")

    def get_main_board_pool(self) -> pd.DataFrame:
        """
        获取 Tushare 生成的 StockIndes_{self.today}.txt 文件，并加载为 DataFrame。
        返回包含标准化纯数字股票代码和股票简称、行业等核心信息的 DataFrame。
        """
        try:
             from Ts_GetStockBasicinfo import TushareStockManager
        except ImportError:
            print("[ERROR] 无法导入 Ts_GetStockBasicinfo 模块，请检查文件路径和类名。")
            return pd.DataFrame(columns=['ts_code', 'name', 'industry', '股票代码'])

        date_suffix = self.today
        filename = f"StockIndes_{date_suffix}.txt"
        dict_file_path = os.path.join(self.base_data_dir, filename)  # 使用 self.base_data_dir

        stock_index_df = pd.DataFrame()
        required_final_cols = ['ts_code', 'name', 'industry', '股票代码']

        if not os.path.exists(dict_file_path):
            print(f"[INFO] 未发现本地股票字典文件: {dict_file_path}，尝试通过 Tushare 获取。")
            try:
                manager = TushareStockManager(self.token)
                # TushareStockManager 内部应该处理 save_path，并确保保存到正确的位置
                stock_index_df = manager.get_basic_data(list_status='L', market='主板', save_path=dict_file_path)
                print(f"[INFO] 股票字典保存至：{dict_file_path}")


                if 'name' not in stock_index_df.columns and '股票简称' in stock_index_df.columns:
                    stock_index_df.rename(columns={'股票简称': 'name'}, inplace=True)
                if 'industry' not in stock_index_df.columns and '行业' in stock_index_df.columns:
                    stock_index_df.rename(columns={'行业': 'industry'}, inplace=True)
                if 'ts_code' not in stock_index_df.columns and 'symbol' in stock_index_df.columns:
                    stock_index_df.rename(columns={'symbol': 'ts_code'}, inplace=True)


            except Exception as e:
                print(f"[ERROR] Tushare接口获取主板股票数据失败: {e}")
                return pd.DataFrame(columns=required_final_cols)
        else:
            print(f"[INFO] 本地股票字典文件已存在: {dict_file_path}，正在加载...")
            try:
                stock_index_df = pd.read_csv(
                    dict_file_path,
                    sep='|',
                    encoding='utf-8-sig',
                    dtype={'ts_code': str, 'symbol': str, 'name': str, 'industry': str, '股票简称': str, '行业': str,
                           'market': str, '股票代码': str}  # 添加 '股票代码' 到 dtype
                )


                if 'ts_code' not in stock_index_df.columns:
                    if 'symbol' in stock_index_df.columns:
                        stock_index_df.rename(columns={'symbol': 'ts_code'}, inplace=True)
                    else:
                        stock_index_df['ts_code'] = 'N/A'

                if 'name' not in stock_index_df.columns:
                    if '股票简称' in stock_index_df.columns:
                        stock_index_df.rename(columns={'股票简称': 'name'}, inplace=True)
                    else:
                        stock_index_df['name'] = 'N/A'

                if 'industry' not in stock_index_df.columns:
                    if '行业' in stock_index_df.columns:
                        stock_index_df.rename(columns={'行业': 'industry'}, inplace=True)
                    else:
                        stock_index_df['industry'] = 'N/A'

                if '股票代码' not in stock_index_df.columns:
                    if 'ts_code' in stock_index_df.columns:
                        stock_index_df['股票代码'] = stock_index_df['ts_code'].apply(
                            lambda x: x.split('.')[0] if isinstance(x, str) and '.' in x else str(x))
                    else:
                        stock_index_df['股票代码'] = 'N/A'
                stock_index_df['股票代码'] = stock_index_df['股票代码'].astype(str).str.zfill(6)

                print(f"[INFO] 从本地文件加载 {len(stock_index_df)} 只股票的基本信息。")

            except Exception as e:
                print(f"[ERROR] 加载本地股票字典文件失败: {e}。将返回空 DataFrame。")
                return pd.DataFrame(columns=required_final_cols)

        # 最终检查，确保所有必需列都存在
        for col in required_final_cols:
            if col not in stock_index_df.columns:
                print(f"[CRITICAL] 最终DataFrame缺少列 '{col}'，无法返回。当前列: {stock_index_df.columns.tolist()}")
                return pd.DataFrame(columns=required_final_cols)

        return stock_index_df[required_final_cols]

    def _safe_ak_fetch(self, fetch_func: callable, description: str, cleaned_file_path: str = None,
                       **kwargs) -> pd.DataFrame:
        """带有重试和延时机制的 Akshare 数据获取函数，可选择缓存清洗后的数据."""

        # 辅助函数：标准化列名，将其中的 '代码' 和 '名称' 统一为 '股票代码' 和 '股票简称'
        def _standardize_report_columns(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            # 清理所有列名的首尾空格，防止隐形字符影响
            df.columns = [col.strip() for col in df.columns]

            # 统一股票代码列名
            if '代码' in df.columns and '股票代码' not in df.columns:
                df.rename(columns={'代码': '股票代码'}, inplace=True)
            # 统一股票简称列名 (如果存在)
            if '名称' in df.columns and '股票简称' not in df.columns:
                df.rename(columns={'名称': '股票简称'}, inplace=True)
            return df

        # 1. 尝试从【清洗后的缓存】加载数据
        if cleaned_file_path and os.path.exists(cleaned_file_path):
            try:
                # 在 dtype 中使用实际文件中的列名
                df = pd.read_csv(
                    cleaned_file_path,
                    sep='|',
                    encoding='utf-8-sig',
                    dtype={
                        '序号': 'Int64',
                        '股票代码': str,  # 假设文件在保存时已经统一为 '股票代码'
                        '股票简称': str,  # 假设文件在保存时已经统一为 '股票简称'
                        '研报数': 'Int64',
                        '机构投资评级(近六个月)-买入': float,
                        '机构投资评级(近六个月)-增持': float,
                        '机构投资评级(近六个月)-中性': float,
                        '机构投资评级(近六个月)-减持': float,
                        '机构投资评级(近六个月)-卖出': float,
                        '2024预测每股收益': float,
                        '2025预测每股收益': float,
                        '2026预测每股收益': float,
                        '2027预测每股收益': float
                    }
                )
                print(f"  - 发现缓存，加载: {os.path.basename(cleaned_file_path)}")
                # 在加载缓存后立即标准化列名，以防 dtype 未能完全覆盖或文件被外部修改
                df = _standardize_report_columns(df)
                print(f"  - (DEBUG) 缓存加载并标准化后 DataFrame 列名: {df.columns.tolist()}")
                return df
            except Exception as e:
                print(f"[WARN] 加载缓存 {os.path.basename(cleaned_file_path)} 失败: {e}，将重新获取。")

        # 2. 如果清洗后的缓存不存在或加载失败，则尝试从原始获取
        df = pd.DataFrame()
        for i in range(self.AKSHARE_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.AKSHARE_RETRIES} 次获取数据: {description}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    # 在获取原始数据后立即标准化列名
                    df = _standardize_report_columns(df)
                    break
                else:
                    print(f"[WARN] 数据返回为空或无效: {description}，重试中。")
                    time.sleep(self.AKSHARE_DELAY)
            except Exception as e:
                print(f"[ERROR] 获取 {description} 时出错: {e}，将在 {self.AKSHARE_DELAY} 秒后重试。")
                time.sleep(self.AKSHARE_DELAY)

        if df.empty:
            print(f"[CRITICAL] 所有重试均失败，未能获取 {description}。")
            return pd.DataFrame()

        # 3. 清洗数据并保存到带有 "_经清洗" 后缀的缓存文件
        if '股票代码' in df.columns:
            df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
            df.drop_duplicates(subset=['股票代码'], inplace=True)

        if cleaned_file_path and not df.empty:
            try:
                df.to_csv(cleaned_file_path, sep='|', index=False, encoding='utf-8-sig')
                print(f"  - 成功将 {description} 数据保存到缓存: {os.path.basename(cleaned_file_path)}")
            except Exception as e:
                print(f"[ERROR] 保存 {description} 缓存失败: {e}")

        return df

    def _get_research_report_filtered_symbols(self) -> Set[str]:
        """
        获取主力研报盈利预测数据，并过滤出“机构投资评级(近六个月)-买入” > 2 的股票代码。
        返回符合条件的纯数字股票代码集合 (如 {'000001', '600000'})。
        """
        print("\n>>> 正在获取主力研报盈利预测数据并进行过滤...")

        # 1. 尝试从内部缓存加载过滤后的股票列表 (set of pure codes)
        if os.path.exists(self.report_filtered_symbols_cache_path):
            try:
                with open(self.report_filtered_symbols_cache_path, 'r', encoding='utf-8') as f:
                    cached_symbols = set(json.load(f))
                print(f"[INFO] 从研报过滤结果缓存加载 {len(cached_symbols)} 只符合研报条件的股票。")
                return cached_symbols
            except Exception as e:
                print(f"[WARN] 加载研报过滤结果缓存失败: {e}，将重新获取。")

        # 2. 从 Akshare 获取数据 (或从_经清洗文件加载)
        report_df = self._safe_ak_fetch(ak.stock_profit_forecast_em, "主力研报盈利预测",
                                        cleaned_file_path=self.raw_report_cache_path)

        if report_df.empty:
            print("[WARNING] 未能获取到主力研报盈利预测数据。研报过滤池将为空。")
            return set()

        original_cols_debug = report_df.columns.tolist()

        cleaned_cols_debug = report_df.columns.tolist()  # 获取标准化后的列名
        if original_cols_debug != cleaned_cols_debug:
            print(f"  - (DEBUG) 主力研报数据列名已清理，原: {original_cols_debug}，清理后: {cleaned_cols_debug}")
        else:
            print(f"  - (DEBUG) 主力研报数据当前列名: {cleaned_cols_debug}")

        # 3. 清洗和过滤数据
        report_col = '机构投资评级(近六个月)-买入'
        code_col = '股票代码'  # 现在这里应该能够正确匹配

        # 确保关键列存在
        if code_col not in report_df.columns or report_col not in report_df.columns:
            print(
                f"[ERROR] 主力研报数据缺少预期列 ('{code_col}' 或 '{report_col}')。当前存在的列: {report_df.columns.tolist()}")
            return set()

        report_df[report_col] = pd.to_numeric(report_df[report_col], errors='coerce').fillna(0)

        # 过滤条件：研报数 > 2
        filtered_reports = report_df[report_df[report_col] > 1].copy()

        if filtered_reports.empty:
            print("[INFO] 过滤后没有股票的研报数 > 2。研报过滤池将为空。")
            return set()

        filtered_reports[code_col] = filtered_reports[code_col].astype(str).str.zfill(6)

        qualified_pure_symbols = set(filtered_reports[code_col].unique().tolist())

        if not qualified_pure_symbols:
            print("[WARNING] 研报过滤后没有有效的纯数字股票代码。研报过滤池将为空。")
            return set()

        print(f"[INFO] 研报过滤后，获取到 {len(qualified_pure_symbols)} 只符合条件的纯数字股票代码。")

        # 4. 保存过滤结果到内部缓存
        try:
            with open(self.report_filtered_symbols_cache_path, 'w', encoding='utf-8') as f:
                json.dump(list(qualified_pure_symbols), f, ensure_ascii=False, indent=4)
            print(f"[INFO] 研报过滤结果已保存到内部缓存: {self.report_filtered_symbols_cache_path}")
        except Exception as e:
            print(f"[ERROR] 保存研报过滤结果到内部缓存失败: {e}")

        return qualified_pure_symbols

    def fetch_combined_data(self, symbol, start, end):
        """封装腾讯接口，合并前复权与不复权数据"""
        try:
            df_qfq = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
            time.sleep(0.05)  # 适当延时，避免触发接口频率限制

            if df_qfq.empty:
                # Akshare接口在非交易日或数据尚未更新时可能返回空，这是正常的
                if end == self.today:
                    pass
                return None

            expected_api_cols = ['date', 'open', 'close', 'high', 'low']
            missing_api_cols = [col for col in expected_api_cols if col not in df_qfq.columns]
            if missing_api_cols:
                print(
                    f"[ERROR] {symbol} QFQ 数据结构异常，Akshare腾讯接口缺少预期列：{missing_api_cols} (当前列: {df_qfq.columns.tolist()})")
                return None

            df_norm = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="")
            time.sleep(0.05)  # 适当延时

            if df_norm.empty:
                return None

            expected_cols_norm = ['date', 'close']
            missing_norm_cols = [col for col in expected_cols_norm if col not in df_norm.columns]
            if missing_norm_cols:
                print(
                    f"[ERROR] {symbol} Normal 数据结构异常，缺少必要列：{missing_norm_cols} (当前列: {df_norm.columns.tolist()})")
                return None

            df_norm = df_norm[['date', 'close']].rename(columns={'close': 'close_normal'})
            df = pd.merge(df_qfq, df_norm, on='date', how='inner')

            if df.empty:
                return None

            try:
                df['close'] = pd.to_numeric(df['close'], errors='raise')
                df['close_normal'] = pd.to_numeric(df['close_normal'], errors='raise')
                df['adj_ratio'] = df['close'] / df['close_normal'].replace(0, pd.NA)
                df.dropna(subset=['adj_ratio'], inplace=True)
            except Exception as e:
                print(f"[ERROR] {symbol} 计算 adj_ratio 失败：{e}")
                return None

            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'trade_date'}, inplace=True)

            final_columns_for_db = [
                'trade_date', 'symbol', 'open', 'close', 'high', 'low',
                'close_normal', 'adj_ratio'
            ]
            df = df[final_columns_for_db]
            return df

        except Exception as e:
            print(f"[CRITICAL] Akshare接口获取 {symbol} 在 {start} 至 {end} 数据时发生错误: {e}")
            return None

    def _clear_stock_daily_kline_table(self):
        """清空 stock_daily_kline 表."""
        print("[INFO] 正在清空 'stock_daily_kline' 表...")
        # 再次检查 self.db 是否为 None，以防在 __init__ 中创建失败
        if self.db is None:
            print("[CRITICAL] 数据库引擎未初始化，无法清空表。")
            raise RuntimeError("数据库引擎未初始化，无法执行数据库操作。")
        try:
            with self.db.connect() as conn:
                conn.execute(text("DELETE FROM stock_daily_kline;"))
                conn.commit()
            print("[INFO] 'stock_daily_kline' 表已清空。")
        except Exception as e:
            print(f"[ERROR] 清空 'stock_daily_kline' 表失败: {e}")
            raise

    def _fetch_and_prepare_data_for_symbol(self, symbol):
        """为单个股票下载从 global_start 到 today 的所有历史数据。"""
        try:
            data = self.fetch_combined_data(symbol, self.global_start, self.today)
            if data is None or data.empty:
                return None
            return data
        except Exception as e:
            print(f"[ERROR] 下载 {symbol} 历史数据失败: {e}")
            return None

    def run_engine(self):
        print(f"[INFO] HistDataEngine 启动运行。")
        print(f"[DEBUG] 配置：起始日期={self.global_start}, 今日日期={self.today}")

        # 在执行数据库操作前，再次检查 self.db 是否已成功初始化
        if self.db is None:
            print("[CRITICAL] HistDataEngine 无法运行：数据库引擎未成功初始化。")
            return

        # Step 1: 获取 Tushare 生成的基础股票池
        tushare_stock_index_df = self.get_main_board_pool()
        if tushare_stock_index_df.empty or '股票代码' not in tushare_stock_index_df.columns:
            print("[CRITICAL] Tushare 获取的基础股票池为空或缺少 '股票代码' 列，无法进行K线数据同步，程序退出。")
            return

        tushare_pure_codes = set(tushare_stock_index_df['股票代码'].unique().tolist())
        print(f"[INFO] 从 Tushare 基础股票池获取 {len(tushare_pure_codes)} 只股票的纯数字代码。")
        # >>> 添加调试打印：Tushare 股票代码样本
        print(f"[DEBUG] Tushare pure codes sample ({len(tushare_pure_codes)} items): {list(tushare_pure_codes)[:10]}")

        # Step 2: 获取研报过滤后的股票代码集合 (纯数字代码)
        report_qualified_pure_codes_set = self._get_research_report_filtered_symbols()
        if not report_qualified_pure_codes_set:
            print("[CRITICAL] 研报过滤后无有效股票代码（研报筛选结果为空），无法进行K线数据同步，程序退出。")
            return


        # Step 3: 进行股票列表的交集运算 (在 Tushare 股票池中，且研报数 > 2)
        final_stock_universe_pure_codes = list(tushare_pure_codes.intersection(report_qualified_pure_codes_set))



        if not final_stock_universe_pure_codes:
            print("[CRITICAL] 没有股票同时满足 Tushare 基础股票池和研报过滤条件，无法进行K线数据同步，程序退出。")
            return

        print(f"[INFO] 经过 Tushare 和研报双重过滤，最终 K 线下载池包含 {len(final_stock_universe_pure_codes)} 只股票。")

        # Step 4: 将纯数字代码转换为 Akshare 格式，用于下载
        def format_pure_code_to_akshare_symbol(code: str) -> Optional[str]:
            code_str = str(code).zfill(6)
            if code_str.startswith('6'):
                return 'sh' + code_str
            elif code_str.startswith(('0', '3')):
                return 'sz' + code_str
            elif code_str.startswith(('4', '8')):  # 北京证券的代码前缀
                return 'bj' + code_str
            # print(f"[WARNING] 无法识别股票代码 {code_str} 的市场类型，将跳过此代码的下载。") # 太多警告了，暂时注释
            return None

        symbols_to_fetch_prefixed = [
            format_pure_code_to_akshare_symbol(code)
            for code in final_stock_universe_pure_codes
        ]
        symbols_to_fetch_prefixed = [s for s in symbols_to_fetch_prefixed if s is not None]

        if not symbols_to_fetch_prefixed:
            print("[CRITICAL] 最终过滤后的股票代码无法转换为有效的 Akshare 格式，无法进行K线数据同步，程序退出。")
            return

        print(f"[INFO] 准备下载 {len(symbols_to_fetch_prefixed)} 只股票的K线数据 先清空 stock_daily_kline 表，再全量下载并写入。")



        self._clear_stock_daily_kline_table()

        # Step 6: 并行下载过滤后的股票的历史数据
        MAX_WORKERS = 10
        all_downloaded_data = []
        futures = []

        total_symbols = len(symbols_to_fetch_prefixed)

        print(f"[INFO] 正在为 {total_symbols} 只股票并行下载数据...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for symbol in symbols_to_fetch_prefixed:
                futures.append(executor.submit(self._fetch_and_prepare_data_for_symbol, symbol))

            for i, future in enumerate(as_completed(futures)):
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_downloaded_data.append(data)
                    # 避免在控制台打印过多，仅在一定频率下打印进度
                    if (i + 1) % 100 == 0 or (i + 1) == total_symbols:
                        print(
                            f"\r[PROGRESS] 已处理 {i + 1}/{total_symbols} 只股票，成功下载 {len(all_downloaded_data)} 只...",
                            end='', flush=True)
                except Exception as exc:
                    print(f'\r[ERROR] 股票数据下载任务在线程中失败: {exc}')
        print("\n[INFO] 所有股票的下载任务已完成。")

        if not all_downloaded_data:
            print(
                "[WARNING] 未下载到任何有效历史数据（可能因为最终股票池没有数据，或者 Akshare 下载失败），数据库将保持清空状态。")
            return

        # Step 7: 合并所有下载的数据并批量写入数据库
        final_df = pd.concat(all_downloaded_data, ignore_index=True)
        if not final_df.empty:
            print(f"[INFO] 所有股票数据已下载并合并，共 {len(final_df)} 条记录。准备写入数据库。")
            try:
                final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])

                db_columns = [
                    'trade_date', 'symbol', 'open', 'close', 'high', 'low',
                    'close_normal', 'adj_ratio'
                ]
                df_to_save = final_df[[col for col in db_columns if col in final_df.columns]]

                # 再次检查 self.db 是否为 None，以防在 __init__ 中创建失败
                if self.db is None:
                    print("[CRITICAL] 数据库引擎未初始化，无法写入数据。")
                    raise RuntimeError("数据库引擎未初始化，无法执行数据库操作。")

                with self.db.begin() as conn:
                    df_to_save.to_sql('stock_daily_kline', conn, if_exists='append', index=False)
                print(f"[INFO] 成功将 {len(df_to_save)} 条数据批量写入 'stock_daily_kline' 表。")
            except Exception as e:
                print(f"[ERROR] 批量写入数据到 'stock_daily_kline' 表失败: {e}")
                raise
        else:
            print("[WARNING] 合并后的DataFrame为空，没有数据写入数据库。")

        print("[INFO] HistDataEngine 所有股票同步任务已完成。")

    def get_synced_stock_codes_from_db(self) -> pd.DataFrame:
        """
        从数据库获取当天已同步的股票代码列表 (带前缀，如 'sz000001')。
        这个列表将作为 Corenews_Main 的基础股票池。
        """
        if self.db is None:
            print("[CRITICAL] 数据库引擎未初始化，无法获取已同步股票代码。")
            return pd.DataFrame(columns=['symbol'])

        query = text(f"""
            SELECT DISTINCT symbol
            FROM stock_daily_kline
            WHERE trade_date = '{self.today_dt.strftime('%Y-%m-%d')}'
        """)
        try:
            with self.db.connect() as conn:
                df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"[ERROR] 从数据库获取已同步股票代码失败: {e}")
            return pd.DataFrame(columns=['symbol'])
