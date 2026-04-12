#!/bin/bash
set -e  # 出错自动退出，避免环境损坏

# 日志输出函数
log_info() {
    echo -e "\033[32m[INFO] $1\033[0m"
}
log_error() {
    echo -e "\033[31m[ERROR] $1\033[0m"
    exit 1
}

# 权限校验：禁止root执行，避免污染系统环境
if [ "$EUID" -eq 0 ]; then
    log_error "请勿使用root权限执行此脚本，普通用户+sudo即可"
fi

# 1. 系统基础环境准备
log_info "=== 1/5 开始更新系统基础环境 ==="
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential cmake git wget curl software-properties-common linux-headers-$(uname -r) python3-pip python3-venv python3-dev -y

# 2. Python版本校验（必须为3.10）
log_info "=== 2/5 校验Python3.10环境 ==="
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VERSION" != "3.10" ]; then
    log_error "当前Python版本为$PY_VERSION，要求Python3.10，请使用Ubuntu22.04原生系统"
fi
log_info "Python3.10版本校验通过"

# 3. 创建并激活Python3.10虚拟环境
log_info "=== 3/5 开始创建项目虚拟环境 ==="
python3 -m venv venv_yolo_follow
source venv_yolo_follow/bin/activate

# 4. 升级工具并安装项目依赖
log_info "=== 4/5 开始安装项目核心依赖 ==="
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ultralytics==8.0.200 gym==0.26.2 opencv-python numpy pybullet rospkg -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装结果校验
log_info "=== 5/5 开始校验环境安装结果 ==="
if pip check; then
    log_info "✅ 环境搭建完成！无依赖冲突"
    log_info "👉 后续使用请执行以下命令激活虚拟环境：source venv_yolo_follow/bin/activate"
else
    log_error "❌ 依赖安装存在冲突，请检查报错信息"
fi

