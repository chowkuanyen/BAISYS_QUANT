# DataManager/Config.py

import os
import configparser
from pathlib import Path

class Config:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self._validate_config_file()
        self._load_config()
        self._ensure_directories()

    def _validate_config_file(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件未找到: {os.path.abspath(self.config_file)}")

    def _load_config(self):

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')


            # 读取数据库配置
        db = config['DATABASE']
        self.DB_USER = db.get('user')
        self.DB_PASSWORD = db.get('password')
        self.DB_HOST = db.get('host')
        self.DB_PORT = db.get('port')
        self.DB_NAME = db.get('db_name')

            # 读取 SYSTEM 配置
        system = config['SYSTEM']
        home_dir = system.get('HOME_DIRECTORY', '~/Downloads/CoreNews_Reports')
        self.HOME_DIRECTORY = os.path.expanduser(home_dir)
        temp_dir = system.get('TEMP_DATA_DIR', 'ShareData')
        self.TEMP_DATA_DIRECTORY = os.path.join(self.HOME_DIRECTORY, temp_dir)


        self.MAX_WORKERS = system.getint('MAX_WORKERS', fallback=15)
        self.DATA_FETCH_RETRIES = system.getint('DATA_FETCH_RETRIES', fallback=3)
        self.DATA_FETCH_DELAY = system.getint('DATA_FETCH_DELAY', fallback=5)

            # 其他配置...
        self.CODE_ALIASES = {'代码': '股票代码', '证券代码': '股票代码', '股票代码': '股票代码'}
        self.NAME_ALIASES = {'名称': '股票简称', '股票名称': '股票简称', '股票简称': '股票简称', '简称': '股票简称'}
        self.PRICE_ALIASES = {'最新价': '最新价', '现价': '最新价', '当前价格': '最新价', '今收盘': '最新价',
                              '收盘': '最新价', '收盘价': '最新价'}


        self.TUSHARE_TOKEN = db.get('tushare_token')  # 如果没有配置，默认为 None
        if not self.TUSHARE_TOKEN:
            raise ValueError("配置文件中缺少 'tushare_token'，请在 [DATABASE] 节点下添加。")

        log = config['LOGGING']
        self.LOG_LEVEL = log.get('LOG_LEVEL', 'INFO')
        self.LOG_DIR = os.path.join(self.HOME_DIRECTORY, log.get('LOG_DIR', 'Logs'))

        for key, val in self.__dict__.items():
            if val is None:
                raise ValueError(f"配置项 '{key}' 未设置，请在 {self.config_file} 中检查。")

    def _ensure_directories(self):
        dirs = [self.HOME_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_db_connection_string(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
