"""
气象预测系统 - 主调度程序
架构: 串口接收 → 数据过滤融合 → XGBoost预测 + UV传感器 → DeepSeek数据整合 → 输出
"""
import sys
import os
import time
import json
import signal
import logging
import argparse
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('weather_predict.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("WeatherPredictSystem")

from serial_receiver import SerialReceiver, SensorData
from data_filter import DataFilter, QWeatherAPI, DataFusion, FusedWeatherData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xgboost'))
from inference_pipeline import WeatherPredictionPipeline


class WeatherPredictSystem:
    def __init__(self, config: dict, pipeline=None):
        self.config = config
        self._running = False

        serial_port = config.get('serial_port', '/dev/ttyUSB0')
        serial_baud = config.get('serial_baudrate', 115200)
        self.receiver = SerialReceiver(port=serial_port, baudrate=serial_baud)
        self.receiver.set_callback(self._on_sensor_data)

        api_key = config.get('qweather_api_key', '')
        location = config.get('qweather_location', '101010100')
        self.qweather = QWeatherAPI(api_key=api_key, location=location)

        window_size = config.get('filter_window_size', 10)
        z_threshold = config.get('filter_z_threshold', 2.5)
        data_filter = DataFilter(window_size=window_size, z_threshold=z_threshold)

        sensor_weight = config.get('sensor_weight', 0.6)
        api_weight = config.get('api_weight', 0.4)
        self.fusion = DataFusion(
            qweather=self.qweather,
            data_filter=data_filter,
            sensor_weight=sensor_weight,
            api_weight=api_weight
        )

        if pipeline:
            self.pipeline = pipeline
        else:
            pipeline_config = {
                'xgboost_models': os.path.join(os.path.dirname(__file__), 'xgboost', 'trained_models'),
                'rkllm_demo_path': config.get('rkllm_demo_path',
                    '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/demo_Linux_aarch64/llm_demo'),
                'rkllm_model_path': config.get('rkllm_model_path',
                    '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm'),
                'amap_api_key': api_key,
                'amap_location': location,
                'output_file': config.get('output_file', 'prediction_output.json'),
            }
            self.pipeline = WeatherPredictionPipeline(pipeline_config)

        self._predict_interval = config.get('predict_interval', 60)
        self._last_predict_time: float = 0
        self._latest_fused: Optional[FusedWeatherData] = None
        self._data_count = 0
        self._predict_count = 0

        self._output_callback = None

    def set_output_callback(self, callback):
        self._output_callback = callback

    def _on_sensor_data(self, data: SensorData):
        self._data_count += 1
        logger.info(
            f"收到传感器数据 #{self._data_count}: "
            f"T={data.temperature:.1f}°C, H={data.humidity:.1f}%, "
            f"P={data.pressure:.1f}hPa"
        )

        fused = self.fusion.fuse(data)
        self._latest_fused = fused

        sensor_dict = {
            'temperature': fused.temperature,
            'humidity': fused.humidity,
            'pressure': fused.pressure,
            'light_intensity': fused.light_intensity,
            'uv_index': fused.uv_index,

            'rainfall': fused.rainfall,
            'altitude': fused.altitude,
        }
        self.pipeline.update_sensor_data(sensor_dict)

        if fused.api_weather_text:
            api_dict = {
                'weather_text': fused.api_weather_text,
                'temperature': fused.api_temperature,
                'humidity': fused.api_humidity,
                'wind_speed': fused.api_wind_speed,
                'wind_direction': fused.api_wind_direction,
            }
            self.pipeline.update_api_data(api_dict)

        elapsed = time.time() - self._last_predict_time
        if elapsed >= self._predict_interval:
            self._trigger_prediction()

    def _trigger_prediction(self):
        logger.info("触发XGBoost气象预测...")
        self._last_predict_time = time.time()

        result = self.pipeline.run_prediction()
        if result:
            self._latest_prediction = result
            self._predict_count += 1
            logger.info(
                f"预测完成 #{self._predict_count}: "
                f"温度趋势={result.get('prediction', {}).get('temp_trend', {}).get('direction_zh', '?')} "
                f"降水概率={result.get('prediction', {}).get('precipitation', {}).get('probability', 0)}%"
            )

            if self._output_callback:
                self._output_callback(result)
        else:
            logger.warning("预测失败，无结果输出")

    def start(self) -> bool:
        logger.info("=" * 60)
        logger.info("气象预测系统启动中...")
        logger.info("=" * 60)

        if not self.pipeline.integrator.init():
            logger.warning("RKLLM初始化失败，将使用规则整合模式")

        if not self.receiver.start():
            logger.error("串口接收启动失败")
            return False

        self._running = True
        logger.info("气象预测系统已启动，等待传感器数据...")
        logger.info(f"预测间隔: {self._predict_interval}秒")

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            self.stop()

        return True

    def stop(self):
        self._running = False
        self.receiver.stop()
        self.pipeline.integrator.destroy()
        logger.info("气象预测系统已停止")

    def get_status(self) -> dict:
        return {
            'running': self._running,
            'data_count': self._data_count,
            'predict_count': self._predict_count,
            'last_predict_time': self._last_predict_time,
            'latest_fused': self._latest_fused,
            'latest_prediction': self._latest_prediction,
            'pipeline_status': self.pipeline.get_status(),
        }


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _run_test(config: dict):
    logger.info("=" * 60)
    logger.info("测试模式 - 使用XGBoost推理管道")
    logger.info("=" * 60)

    pipeline_config = {
        'xgboost_models': os.path.join(os.path.dirname(__file__), 'xgboost', 'trained_models'),
        'rkllm_demo_path': config.get('rkllm_demo_path',
            '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/demo_Linux_aarch64/llm_demo'),
        'rkllm_model_path': config.get('rkllm_model_path',
            '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm'),
        'amap_api_key': config.get('qweather_api_key', ''),
        'amap_location': config.get('qweather_location', '101010100'),
        'output_file': config.get('output_file', 'prediction_output.json'),
    }

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xgboost'))
    from inference_pipeline import WeatherPredictionPipeline

    pipeline = WeatherPredictionPipeline(pipeline_config)

    sensor = {
        'temperature': 26.5, 'humidity': 72.3, 'pressure': 1013.2,
        'light_intensity': 35000.0, 'uv_index': 5.2,

    }
    api = {
        'weather_text': '多云', 'temperature': 25.0,
        'humidity': 65.0, 'wind_speed': 2.0, 'wind_direction': '东',
    }

    pipeline.update_sensor_data(sensor)
    pipeline.update_api_data(api)

    logger.info("开始XGBoost测试预测...")
    result = pipeline.run_prediction()

    if result:
        logger.info("测试预测成功!")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.error("测试预测失败")

    pipeline.integrator.destroy()


DEFAULT_CONFIG = {
    "serial_port": "/dev/ttyUSB0",
    "serial_baudrate": 115200,
    "qweather_api_key": "",
    "qweather_location": "101010100",
    "filter_window_size": 10,
    "filter_z_threshold": 2.5,
    "sensor_weight": 0.6,
    "api_weight": 0.4,
    "rkllm_model_path": "/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm",
    "rkllm_demo_path": "/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/demo_Linux_aarch64/llm_demo",
    "predict_interval": 10,
    "output_file": "prediction_output.json"
}


def main():
    parser = argparse.ArgumentParser(description='气象预测系统 - RK3588')
    parser.add_argument('-c', '--config', default='config.json', help='配置文件路径')
    parser.add_argument('-p', '--port', default=None, help='串口设备路径')
    parser.add_argument('-b', '--baudrate', type=int, default=None, help='串口波特率')
    parser.add_argument('-k', '--api-key', default=None, help='和风天气API Key')
    parser.add_argument('-l', '--location', default=None, help='和风天气城市代码')
    parser.add_argument('-i', '--interval', type=int, default=None, help='预测间隔(秒)')
    parser.add_argument('-t', '--test', action='store_true', help='测试模式(无需传感器)')
    parser.add_argument('-g', '--gui', action='store_true', help='启动Qt图形看板')
    parser.add_argument('--refresh', type=int, default=5000, help='看板刷新间隔(毫秒)')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    file_config = load_config(args.config)
    config.update(file_config)

    if args.port:
        config['serial_port'] = args.port
    if args.baudrate:
        config['serial_baudrate'] = args.baudrate
    if args.api_key:
        config['qweather_api_key'] = args.api_key
    if args.location:
        config['qweather_location'] = args.location
    if args.interval:
        config['predict_interval'] = args.interval

    if args.test:
        _run_test(config)
        return

    if args.gui:
        _run_with_gui(config, args.refresh)
    else:
        system = WeatherPredictSystem(config)

        def signal_handler(sig, frame):
            logger.info("收到终止信号")
            system.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        system.start()


def _run_with_gui(config: dict, refresh_ms: int):
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        wayland_socket = os.path.join('/run', 'user', str(os.getuid()), 'wayland-0')
        if os.path.exists(wayland_socket):
            os.environ['WAYLAND_DISPLAY'] = 'wayland-0'
            os.environ['QT_QPA_PLATFORM'] = 'wayland'
            logger.info("自动检测Wayland显示环境")
        elif os.environ.get('DISPLAY') is None:
            os.environ['DISPLAY'] = ':0'
            logger.info("自动设置DISPLAY=:0")

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from qt_dashboard.main_window import WeatherDashboard

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    output_file = config.get('output_file', 'prediction_output.json')
    dashboard = WeatherDashboard(json_file_path=output_file)
    dashboard.refresh_timer.setInterval(refresh_ms)
    dashboard.setWindowTitle("气象预测系统 - RK3588 (XGBoost+DeepSeek)")

    class InitAndPredictThread(QThread):
        pipeline_ready = pyqtSignal(object)
        error_signal = pyqtSignal(str)

        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            self.system = None
            self.pipeline = None

        def run(self):
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xgboost'))
                from inference_pipeline import WeatherPredictionPipeline

                pipeline_config = {
                    'xgboost_models': os.path.join(os.path.dirname(__file__), 'xgboost', 'trained_models'),
                    'rkllm_demo_path': self.config.get('rkllm_demo_path',
                        '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/demo_Linux_aarch64/llm_demo'),
                    'rkllm_model_path': self.config.get('rkllm_model_path',
                        '/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm'),
                    'amap_api_key': self.config.get('qweather_api_key', ''),
                    'amap_location': self.config.get('qweather_location', '101010100'),
                    'output_file': self.config.get('output_file', 'prediction_output.json'),
                }

                self.pipeline = WeatherPredictionPipeline(pipeline_config)
                self.pipeline.integrator.init()
                self.pipeline_ready.emit(self.pipeline)

                self.system = WeatherPredictSystem(self.config, pipeline=self.pipeline)
                self.system.start()
            except Exception as e:
                self.error_signal.emit(str(e))

        def stop(self):
            if self.system:
                self.system.stop()

    pred_thread = InitAndPredictThread(config)
    pred_thread.pipeline_ready.connect(lambda p: setattr(dashboard, 'pipeline', p))
    pred_thread.error_signal.connect(lambda msg: logger.error(f"预测线程错误: {msg}"))
    pred_thread.start()
    
    logger.info("=" * 60)
    logger.info("气象预测系统已启动 (GUI模式)")
    logger.info(f"Qt看板刷新间隔: {refresh_ms}ms")
    logger.info("按 Ctrl+C 终止程序")
    logger.info("=" * 60)
    
    import signal
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在关闭...")
        pred_thread.stop()
        pred_thread.wait(3000)
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def cleanup():
        logger.info("正在关闭...")
        pred_thread.stop()
        pred_thread.wait(3000)
    
    app.aboutToQuit.connect(cleanup)
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()