#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器

用于加载和解析配置文件
"""

import os
import yaml

class ConfigLoader:
    """
    配置加载器类
    """
    def __init__(self, config_path=None):
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径，默认为项目根目录下的config.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            self.config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config.yaml'
            )
        else:
            self.config_path = config_path
    
    def load_config(self):
        """
        加载配置文件
        
        返回:
            dict: 配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                'qlib': {
                    'provider_uri': "~/.qlib/qlib_data/cn_data",
                    'region': "REG_US"
                },
                'data_source': "yahoo",
                'default_symbols': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
            }