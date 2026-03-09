import tushare
import   tushare as ts
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

    def get_basic_data(self, list_status='L',market ='主板', save_path='stock_basic_data.txt') -> pd.DataFrame:
        """
        获取股票基础数据，并保存到本地
        :param list_status: 上市状态 (L上市, D退市, P暂停上市)
        :param save_path: 本地保存路径
        :return: 包含指定字段的 pandas DataFrame
        """
        try:
            # 定义需要的字段
            target_fields = 'ts_code,symbol,name,industry,market'

            # 从接口拉取数据
            df = self.pro.stock_basic(
                exchange='',
                list_status=list_status,
                market =market,
                fields=target_fields
            )

            def format_ts_code(x):
                if not isinstance(x, str) or '.' not in x:
                    return x
                code, market = x.split('.')
                return market + code

            df['ts_code'] = df['ts_code'].apply(format_ts_code)


            if df.empty:
                print("⚠️ 警告：获取到的数据为空，请检查权限或参数。")
                return df

            # 保存到本地，使用 "|" 分隔
            df.to_csv(save_path, sep='|', index=False, encoding='utf-8-sig')
            print(f"数据已成功保存至: {save_path}")

            return df

        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()  # 发生错误时返回空 DataFrame
