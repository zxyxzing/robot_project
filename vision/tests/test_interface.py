#!/usr/bin/env python3
"""
test_interface.py
测试 SimVideoClient 的接口是否符合项目要求
"""

import sys
import os

# 添加上级目录到路径，以便导入 vision 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_video_client import SimVideoClient

def test_interface():
    """测试接口是否符合要求"""
    print("测试 SimVideoClient 接口...")
    
    # 1. 创建实例
    client = SimVideoClient()
    
    # 2. 测试 GetImageSample 方法是否存在
    assert hasattr(client, 'GetImageSample'), "缺少 GetImageSample 方法"
    print("✓ GetImageSample 方法存在")
    
    # 3. 调用方法获取图像
    image = client.GetImageSample()
    print("✓ GetImageSample 调用成功")
    
    # 4. 验证返回类型
    assert image is not None, "返回的图像不能为 None"
    print("✓ 返回图像不为空")
    
    # 5. 验证格式
    client.validate_image_format(image)
    
    print("\n所有接口测试通过！")
    return True

if __name__ == "__main__":
    test_interface()