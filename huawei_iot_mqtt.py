import os
import json
import time
import hmac
import hashlib
import re
import serial
import paho.mqtt.client as mqtt
from datetime import datetime

# ===== 设备配置（请替换为你的实际参数，或通过环境变量设置）=====
DEVICE_ID     = os.environ.get("HUAWEI_IOT_DEVICE_ID", "your_device_id")
DEVICE_SECRET = os.environ.get("HUAWEI_IOT_DEVICE_SECRET", "your_device_secret")
SERVER        = os.environ.get("HUAWEI_IOT_SERVER", "your_endpoint.st1.iotda-device.cn-north-4.myhuaweicloud.com")
PORT   = 8883  # TLS加密端口
# ==========================================

TOPIC_REPORT = f"$oc/devices/{DEVICE_ID}/sys/properties/report"

def build_auth():
    """生成 ClientId / Username / Password（HMAC-SHA256）"""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H")
    client_id = f"{DEVICE_ID}_0_0_{timestamp}"
    password  = hmac.new(
        timestamp.encode(), DEVICE_SECRET.encode(), hashlib.sha256
    ).hexdigest()
    return client_id, DEVICE_ID, password

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200

# 解析串口数据行，格式: T:28C H:57% L:18lx P:1005.78hPa UV:0.00 Time:.. Lat:41.8239 Lon:123.5663 Alt:..
_PATTERN = re.compile(
    r'T:(\d+)C\s+H:(\d+)%\s+L:([\d.]+)lx\s+P:([\d.]+)hPa\s+UV:([\d.]+)'
    r'(?:.*Lat:([\d.]+)\s+Lon:([\d.]+))?'
)

def open_serial(port=SERIAL_PORT, baud=BAUD_RATE, retry_interval=5):
    """尝试打开串口，失败时持续重试直到成功；成功后等待设备稳定"""
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=2)
            print(f"✅ 串口 {port} 已打开，等待设备稳定…")
            time.sleep(2)  # 给 USB-串口适配器时间完成枚举
            ser.reset_input_buffer()
            print(f"✅ 串口 {port} 就绪")
            return ser
        except serial.SerialException as e:
            print(f"⚠️  无法打开串口 {port}：{e}，{retry_interval}s 后重试…")
            time.sleep(retry_interval)

def read_sensor(ser):
    """清空缓冲区后读取最新一帧传感器数据；设备断线时抛出 SerialException"""
    ser.reset_input_buffer()  # 丢弃积压的旧数据
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            # readline 超时返回空串——检查串口是否仍然在线
            if not ser.is_open:
                raise serial.SerialException("串口已关闭")
            continue
        m = _PATTERN.search(line)
        if m:
            data = {
                "Temp":  float(m.group(1)),
                "Humi":  float(m.group(2)),
                "Light": float(m.group(3)),
                "Press": float(m.group(4)),
                "UV":    float(m.group(5)),
            }
            if m.group(6) and m.group(7):
                data["GPS-Latitude"]  = float(m.group(6))
                data["GPS-longitude"] = float(m.group(7))
            return data

def on_connect(client, userdata, flags, reason_code, properties):
    print("✅ 已连接华为云IoT" if reason_code == 0 else f"❌ 连接失败，rc={reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"   └─ 上报成功 mid={mid}")

client_id, username, password = build_auth()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
client.username_pw_set(username, password)
client.tls_set()  # 启用TLS加密（使用系统CA证书）
client.on_connect = on_connect
client.on_publish  = on_publish

print(f"正在连接 {SERVER}:{PORT} ...")
client.connect(SERVER, PORT, keepalive=60)
client.loop_start()
time.sleep(2)  # 等待连接建立

ser = open_serial()

try:
    while True:
        try:
            props = read_sensor(ser)
        except serial.SerialException as e:
            print(f"⚠️  串口断线：{e}")
            try:
                ser.close()
            except Exception:
                pass
            ser = open_serial()   # 阻塞重连，直到串口恢复
            continue

        payload = {
            "services": [{
                "service_id": "weather",
                "properties": props,
                "event_time": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            }]
        }
        client.publish(TOPIC_REPORT, json.dumps(payload), qos=1)
        print(f"📡 上报: {props}")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n👋 停止上报")
    ser.close()
    client.loop_stop()
    client.disconnect()
