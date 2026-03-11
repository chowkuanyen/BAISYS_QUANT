import os
import tushare as ts
import pandas as pd


class TushareStockManager:
    """Tushare 股票数据管理类"""

    def __init__(self, token: str):
        """
        初始化 API 连接
        :param token: Tushare 用户凭证
        """
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.HOME_DIRECTORY = os.path.expanduser('~')

    def get_basic_data(self, list_status='L', market='主板', save_path='stock_basic_data.txt') -> pd.DataFrame:
        """
        获取股票基础数据，并保存到本地
        :param list_status: 上市状态 (L上市, D退市, P暂停上市)
        :param save_path: 本地保存路径
        :return: 包含指定字段和标准化股票代码的 pandas DataFrame
        """
        try:
            # 定义需要的字段
            target_fields = 'ts_code,symbol,name,industry,market'

            # 从接口拉取数据
            df = self.pro.stock_basic(
                exchange='',
                list_status=list_status,
                market=market,
                fields=target_fields
            )

            if df.empty:
                print("⚠️ 警告：获取到的数据为空，请检查权限或参数。")
                return df

            # 1. 格式化 ts_code 为 Akshare 风格 (如 'sh000001')
            # Tushare 的 ts_code 通常是 '000001.SZ' 格式
            # Akshare 通常需要 'sh000001' 或 'sz000001'
            def format_ts_code_to_akshare_style(x):
                if not isinstance(x, str) or '.' not in x:
                    return x # Already in desired format or not a string
                code, market_suffix = x.split('.')
                market_prefix = ''
                if market_suffix == 'SH':
                    market_prefix = 'sh'
                elif market_suffix == 'SZ':
                    market_prefix = 'sz'
                elif market_suffix == 'BJ':
                    market_prefix = 'bj'
                return market_prefix + code.zfill(6) # Ensure 6 digits

            df['ts_code_akshare'] = df['ts_code'].apply(format_ts_code_to_akshare_style)

            # 2. 添加纯数字的 '股票代码' 列，用于后续匹配和分析
            # Tushare 的 symbol 列本身就是纯数字代码 (如 '000001')
            df['股票代码'] = df['symbol'].astype(str).str.zfill(6)

            # 3. 标准化 'name' 列为 '股票简称'
            if 'name' in df.columns and '股票简称' not in df.columns:
                df.rename(columns={'name': '股票简称'}, inplace=True)
            elif '股票简称' not in df.columns and 'name' not in df.columns:
                df['股票简称'] = 'N/A' # 兜底

            # 选择需要保存和返回的列
            # 确保 'ts_code_akshare' 存在, 用于 Corenews_Main 加载
            # 确保 '股票代码', '股票简称', '行业' 存在, 用于 HistDataEngine 内部逻辑
            cols_to_keep = ['ts_code_akshare', '股票代码', '股票简称', 'industry', 'market']
            df_final = df[[col for col in cols_to_keep if col in df.columns]].copy()
            df_final.rename(columns={'ts_code_akshare': 'ts_code', 'industry': '行业'}, inplace=True)


            # 保存到本地，使用 "|" 分隔
            df_final.to_csv(save_path, sep='|', index=False, encoding='utf-8-sig')
            print(f"数据已成功保存至: {save_path}")

            return df_final

        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()  # 发生错误时返回空 DataFrame
