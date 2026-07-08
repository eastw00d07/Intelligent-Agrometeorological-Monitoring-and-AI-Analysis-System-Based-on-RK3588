#!/bin/bash
# 农业气象预测系统 - Qt看板启动脚本

export XDG_RUNTIME_DIR=/run/user/$(id -u)
export QT_QPA_PLATFORM=wayland
export WAYLAND_DISPLAY=wayland-0
export QT_WAYLAND_DISABLE_WINDOWDECORATION=1

PROJECT_DIR=/userdata/home/elf/weather_predict_system/rk3588_server
cd "$PROJECT_DIR"

python3 /home/elf/sensor_project/huawei_iot_mqtt.py &

exec /usr/bin/python3 "$PROJECT_DIR/main.py" --gui --refresh 5000