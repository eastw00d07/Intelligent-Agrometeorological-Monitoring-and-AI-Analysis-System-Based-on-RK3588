# 智能天气预测系统

基于 RK3588 边缘计算平台的智能气象预测与远程监控系统，采用 XGBoost + DeepSeek 混合推理架构，集成华为云 IoT 数据上报与微信小程序远程监控。

## 系统架构

```
┌─────────────┐    UART     ┌──────────────────────────────────────┐
│  STM32F1    │────────────▶│           RK3588 主控平台             │
│  传感器采集  │  115200bps  │                                      │
│  温度/湿度   │             │  串口接收 → 数据过滤 → 数据融合       │
│  气压/光照   │             │       ↓                              │
│  UV/降雨    │             │  特征构建(46维) → XGBoost预测(3模型)  │
└─────────────┘             │       ↓                              │
                            │  DeepSeek API整合 → 结构化JSON输出    │
                            │       ↓                              │
                            │  PyQt5看板 / Web看板 / 语音交互       │
                            └──────────┬───────────────────────────┘
                                       │ MQTT TLS
                                       ▼
                            ┌─────────────────────┐
                            │   华为云 IoT 平台    │
                            │  规则引擎 → 数据流转  │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   微信小程序          │
                            │  实时监控/历史曲线    │
                            │  预警推送/远程控制    │
                            └─────────────────────┘
```

## 核心功能

- **多传感器实时采集**：温度、湿度、气压、光照强度、UV紫外线、降雨量，支持文本和二进制帧双协议（CRC16校验）
- **数据清洗与融合**：Z-score 异常过滤 + 传感器/高德天气API 加权融合（0.6/0.4）
- **XGBoost 三模型预测**：温度趋势回归、降水风险回归、暴雨概率分类，基于 46 维特征向量
- **DeepSeek 大模型整合**：调用 DeepSeek Chat API 将数值预测整合为结构化 JSON + 自然语言摘要
- **极端天气预警**：高温/寒潮/暴雨检测，蓝/黄/橙/红四级预警
- **PyQt5 气象看板**：传感器面板、预测结果面板、预警卡片、趋势折线图、UV指数圆盘、置信度环形条、农业AI建议
- **语音交互**：录音 → 百度STT → DeepSeek对话 → TTS播放
- **Web 看板**：Flask REST API 远程访问
- **华为云 IoT 上报**：MQTT over TLS（8883端口），HMAC-SHA256 认证，5秒间隔上报
- **微信小程序**：实时数据监控、历史曲线查询、预警推送、远程命令下发

## 硬件要求

| 组件 | 型号/规格 |
|------|-----------|
| 主控平台 | RK3588（6TOPS NPU，8核CPU） |
| 传感器MCU | STM32F1 |
| 温湿度传感器 | 支持 I2C/UART 输出 |
| 气压传感器 | BMP280 |
| 光照传感器 | 0~150000lux |
| UV传感器 | 紫外线指数 0~15 |
| 降雨量传感器 | 0~100mm |
| 音频 | NAU8822 声卡（语音交互） |
| 通信 | USB转串口（/dev/ttyUSB0） |

## 软件依赖

- Python 3.10+
- PyQt5
- pyserial >= 3.5
- requests >= 2.28.0
- paho-mqtt
- flask + flask-cors
- xgboost
- numpy / pandas

安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

### 1. 环境变量

复制示例配置并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# DeepSeek API
DEEPSEEK_API_KEY=sk-your-deepseek-api-key

# 高德天气 API
AMAP_API_KEY=your-amap-api-key

# 百度语音识别
BAIDU_STT_API_KEY=your-baidu-stt-api-key
BAIDU_STT_SECRET_KEY=your-baidu-stt-secret-key

# 华为云 IoT
HUAWEI_IOT_DEVICE_ID=your_device_id
HUAWEI_IOT_DEVICE_SECRET=your_device_secret
HUAWEI_IOT_SERVER=your_endpoint.st1.iotda-device.cn-north-4.myhuaweicloud.com
```

加载环境变量：

```bash
source .env  # 或 export $(cat .env | xargs)
```

### 2. 系统配置

编辑 `config.json`：

```json
{
    "serial_port": "/dev/ttyUSB0",
    "serial_baudrate": 115200,
    "qweather_api_key": "",
    "qweather_location": "210100",
    "filter_window_size": 10,
    "filter_z_threshold": 2.5,
    "sensor_weight": 0.6,
    "api_weight": 0.4,
    "predict_interval": 300,
    "output_file": "prediction_output.json"
}
```

> `qweather_api_key` 可留空，系统会通过环境变量 `AMAP_API_KEY` 读取。

## 使用方法

### 启动 Qt 看板 + 华为云 IoT 上报

```bash
bash start_dashboard.sh
```

### 仅启动主程序（无GUI）

```bash
python3 main.py --test
```

### 启动 Web 看板

```bash
python3 web_dashboard.py -p 5000 -H 0.0.0.0
```

浏览器访问 `http://<RK3588_IP>:5000`

### 仅启动华为云 IoT 上报

```bash
python3 huawei_iot_mqtt.py
```

## 项目结构

```
├── main.py                    # 主调度程序
├── serial_receiver.py         # 串口数据接收（文本+二进制帧+CRC16）
├── data_filter.py             # 数据过滤融合（Z-score + 加权融合）
├── web_dashboard.py           # Flask Web 看板
├── huawei_iot_mqtt.py         # 华为云 IoT MQTT 上报
├── rkllm_inference.py         # RKLLM 推理封装（备用本地推理）
├── rkllm_wrapper.c            # RKLLM C 语言封装层
├── config.json                # 系统配置
├── requirements.txt           # Python 依赖
├── start_dashboard.sh         # 启动脚本
├── .env.example               # 环境变量示例
├── .gitignore
├── xgboost/
│   └── inference_pipeline.py  # XGBoost 推理管道（3模型46特征）
└── qt_dashboard/
    ├── __init__.py
    ├── main_window.py         # Qt 主窗口布局与交互
    ├── widgets.py             # 自定义组件（仪表盘/圆盘/环形条/折线图/卡片）
    ├── binding_manager.py     # JSON 数据 → Qt 控件绑定
    ├── data_model.py          # 数据模型定义
    ├── voice_dialog.py        # 语音交互对话框
    └── styles.qss             # Qt 样式表
```

## 数据流

1. STM32F1 采集传感器数据 → UART 发送至 RK3588
2. `SerialReceiver` 接收并解析 → `DataFilter` Z-score 过滤 → `DataFusion` 传感器+API加权融合
3. `FeatureBuilder` 构建 46 维特征 → `XGBoostPredictor` 三模型预测
4. `UVIndexProvider` 获取 UV 指数（传感器→API→默认）
5. `DataIntegrator` 整合为结构化 JSON + DeepSeek API 生成自然语言摘要
6. 输出写入 `prediction_output.json` → Qt 看板 5 秒刷新读取
7. `huawei_iot_mqtt.py` 独立读取串口 → MQTT TLS 上报华为云 IoT → 小程序订阅

## XGBoost 特征工程

46 维特征向量包含：

- **原始气象量**：温度、湿度、气压、露点、体感温度、降水、天气编码、云量、风速、风向、阵风、辐射、日照、UV、能见度
- **时间特征**：小时、月份、年中日、是否白天、是否周末
- **变化率特征**：3h/6h 气压变化、3h/6h 温度变化、3h 湿度变化
- **交互特征**：温湿交互项、热指数、风向正余弦分解
- **时序滞后特征**：1/3/6/12/24h 温度、湿度、气压滞后值

## 预警规则

| 条件 | 预警等级 | 类型 |
|------|----------|------|
| 温度 > 35℃ | 橙色 | 高温 |
| 温度 < 5℃ | 蓝色 | 寒潮 |
| 暴雨概率 > 30% | 黄色 | 暴雨 |

## 许可证

MIT License