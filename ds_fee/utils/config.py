import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, **kwargs):
        # 过滤掉不可序列化的属性
        safe_kwargs = {}
        for key, value in kwargs.items():
            try:
                import pickle
                pickle.dumps(value)
                safe_kwargs[key] = value
            except Exception as e:
                print(f"警告: 属性 {key} 不可序列化")
                print(f"类型: {type(value)}")
                print(f"错误: {str(e)}")
                print("------------------------")
        self.__dict__.update(safe_kwargs)
    
    def __getstate__(self):
        # 确保对象可以被序列化
        return self.__dict__
    
    def __setstate__(self, state):
        # 从序列化状态恢复
        self.__dict__.update(state)

    """基础配置类"""
    def get(self, key, default=None):
        """获取配置项，支持默认值
        
        Args:
            key: 配置项键名
            default: 默认值，当配置项不存在时返回
            
        Returns:
            配置项值或默认值
        """
        return getattr(self, key, default)
        
    def to_dict(self):
        """将配置对象转换为字典格式
        
        Returns:
            dict: 配置对象的字典表示
        """
        return {key: value for key, value in self.__dict__.items()}
        
    def __iter__(self):
        """使Config对象可迭代，支持dict()转换
        
        Returns:
            iterator: 配置项的迭代器
        """
        return iter(self.to_dict().items())

def load_base_config(config_path: str = None) -> Config:
    """加载基础配置文件
    
    Args:
        config_path (str, optional): 配置文件路径. 默认使用ds_fee/configs/base_config.yaml
        
    Returns:
        Config: 配置对象
    """
    if not config_path:
        config_path = os.path.join(Path(__file__).parent.parent, "conf/base_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            config = Config()
            for key, value in config_data.items():
                setattr(config, key, value)
            return config
    except FileNotFoundError:
        raise Exception(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise Exception(f"配置文件解析失败: {str(e)}")
    except Exception as e:
        raise Exception(f"配置文件加载异常: {str(e)}")
