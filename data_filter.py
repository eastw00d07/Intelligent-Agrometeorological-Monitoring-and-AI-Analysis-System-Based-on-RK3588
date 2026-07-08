"""
数据过滤融合处理模块
1. 对传感器原始数据进行异常值过滤(滑动窗口+统计方法)
2. 获取和风天气API数据
3. 融合本地传感器数据与API数据
"""
import time
import logging
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0

    rainfall: float = 0.0
    light_intensity: float = 0.0
    uv_index: float = 0.0
    altitude: float = 0.0
    timestamp: int = 0
    sensor_status: int = 0
    recv_time: float = 0.0


SENSOR_RANGES = {
    'temperature': (-40.0, 60.0),
    'humidity': (0.0, 100.0),
    'pressure': (800.0, 1200.0),

    'rainfall': (0.0, 100.0),
    'light_intensity': (0.0, 150000.0),
    'uv_index': (0.0, 15.0),
    'altitude': (-500.0, 9000.0),
}

WINDOW_SIZE = 10
Z_SCORE_THRESHOLD = 2.5


class DataFilter:
    def __init__(self, window_size: int = WINDOW_SIZE,
                 z_threshold: float = Z_SCORE_THRESHOLD):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._buffers: Dict[str, deque] = {}
        for key in SENSOR_RANGES:
            self._buffers[key] = deque(maxlen=window_size)

    def filter_data(self, data: SensorData) -> Optional[SensorData]:
        filtered = SensorData(
            timestamp=data.timestamp,
            sensor_status=data.sensor_status,
            recv_time=data.recv_time
        )
        fields = ['temperature', 'humidity', 'pressure',
                   'rainfall', 'light_intensity', 'uv_index',
                   'altitude']

        for f in fields:
            val = getattr(data, f)
            lo, hi = SENSOR_RANGES[f]

            if val < lo or val > hi:
                logger.warning(f"传感器数据越界: {f}={val}, 有效范围[{lo},{hi}]")
                if len(self._buffers[f]) > 0:
                    val = sum(self._buffers[f]) / len(self._buffers[f])
                else:
                    val = (lo + hi) / 2.0
                logger.info(f"已替换为: {f}={val:.2f}")

            if len(self._buffers[f]) >= 3:
                mean = sum(self._buffers[f]) / len(self._buffers[f])
                variance = sum((x - mean) ** 2 for x in self._buffers[f]) / len(self._buffers[f])
                std = variance ** 0.5
                min_std = max(0.1, abs(mean) * 0.001)
                if std > min_std:
                    z = abs(val - mean) / std
                    if z > self.z_threshold:
                        logger.warning(f"Z-score异常: {f}={val:.2f}, z={z:.2f}")
                        val = mean
                        logger.info(f"Z-score修正: {f}={val:.2f}")

            self._buffers[f].append(val)
            setattr(filtered, f, val)

        return filtered

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for key, buf in self._buffers.items():
            if len(buf) > 0:
                vals = list(buf)
                mean = sum(vals) / len(vals)
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                stats[key] = {
                    'mean': mean,
                    'std': variance ** 0.5,
                    'min': min(vals),
                    'max': max(vals),
                    'latest': vals[-1],
                    'count': len(vals)
                }
        return stats


@dataclass
class QWeatherData:
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0
    wind_speed: float = 0.0
    wind_direction: str = ""
    weather_text: str = ""
    weather_code: str = ""
    precip: float = 0.0
    visibility: float = 0.0
    cloud_cover: float = 0.0
    update_time: str = ""


class QWeatherAPI:
    BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str, location: str = "101010100"):
        self.api_key = api_key
        self.location = location
        self._cache: Optional[QWeatherData] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 1800.0

    def _parse_adm_location(self, location: str) -> str:
        if len(location) == 9 and location.isdigit():
            province = location[:2]
            city = location[2:4]
            district = location[4:6]
            return f"{province},{city},{district}"
        return location

    def get_weather_now(self) -> Optional[QWeatherData]:
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache

        try:
            adm_location = self._parse_adm_location(self.location)
            url = f"{self.BASE_URL}/weather/weatherInfo"
            params = {
                'city': adm_location,
                'key': self.api_key,
                'extensions': 'base'
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get('status') != '1':
                logger.error(f"高德天气API错误: info={data.get('info')}")
                return self._cache

            lives = data.get('lives', [])
            if not lives:
                logger.error("高德天气API返回空数据")
                return self._cache

            now = lives[0]
            wind_power = now.get('windpower', '0')
            try:
                wind_speed_val = float(wind_power.split('-')[-1] if '-' in wind_power else wind_power)
            except (ValueError, IndexError):
                wind_speed_val = 0.0

            weather = QWeatherData(
                temperature=float(now.get('temperature', 0)),
                humidity=float(now.get('humidity', 0)),
                pressure=0.0,
                wind_speed=wind_speed_val,
                wind_direction=now.get('winddirection', ''),
                weather_text=now.get('weather', ''),
                weather_code=now.get('weather', ''),
                precip=0.0,
                visibility=float(now.get('visibility', 0)) if now.get('visibility') else 0.0,
                cloud_cover=0.0,
                update_time=now.get('reporttime', '')
            )
            self._cache = weather
            self._cache_time = time.time()
            logger.info(f"高德天气数据更新: {weather.weather_text}, {weather.temperature}°C")
            return weather

        except Exception as e:
            logger.error(f"高德天气API请求失败: {e}")
            return self._cache

    def get_weather_24h(self) -> Optional[List[Dict]]:
        try:
            adm_location = self._parse_adm_location(self.location)
            url = f"{self.BASE_URL}/weather/weatherInfo"
            params = {
                'city': adm_location,
                'key': self.api_key,
                'extensions': 'all'
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get('status') != '1':
                logger.error(f"高德天气预报API错误: info={data.get('info')}")
                return None

            forecasts = data.get('forecasts', [])
            if not forecasts:
                return None

            casts = forecasts[0].get('casts', [])
            hourly = []
            for cast in casts:
                hourly.append({
                    'fxTime': cast.get('date', ''),
                    'temp': cast.get('daytemp', '0'),
                    'humidity': cast.get('dayhumidity', '0'),
                    'text': cast.get('dayweather', ''),
                    'windSpeed': cast.get('daypower', '0'),
                })
            return hourly

        except Exception as e:
            logger.error(f"高德天气预报API请求失败: {e}")
            return None


@dataclass
class FusedWeatherData:
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0

    rainfall: float = 0.0
    light_intensity: float = 0.0
    uv_index: float = 0.0
    altitude: float = 0.0
    api_temperature: float = 0.0
    api_humidity: float = 0.0
    api_pressure: float = 0.0
    api_wind_speed: float = 0.0
    api_wind_direction: str = ""
    api_weather_text: str = ""
    api_weather_code: str = ""
    api_precip: float = 0.0
    api_visibility: float = 0.0
    api_cloud_cover: float = 0.0
    temp_diff: float = 0.0
    humidity_diff: float = 0.0
    pressure_diff: float = 0.0
    sensor_status: int = 0
    timestamp: int = 0
    recv_time: float = 0.0
    fusion_time: float = 0.0
    statistics: Dict = field(default_factory=dict)


class DataFusion:
    SENSOR_WEIGHT = 0.6
    API_WEIGHT = 0.4

    def __init__(self, qweather: QWeatherAPI, data_filter: DataFilter,
                 sensor_weight: float = 0.6, api_weight: float = 0.4):
        self.qweather = qweather
        self.data_filter = data_filter
        self.sensor_weight = sensor_weight
        self.api_weight = api_weight

    def fuse(self, raw_data: SensorData) -> FusedWeatherData:
        filtered = self.data_filter.filter_data(raw_data)
        api_data = self.qweather.get_weather_now()
        stats = self.data_filter.get_statistics()

        fused = FusedWeatherData(
            temperature=filtered.temperature,
            humidity=filtered.humidity,
            pressure=filtered.pressure,
            wind_speed=0.0,
            wind_direction=0.0,
            rainfall=filtered.rainfall,
            light_intensity=filtered.light_intensity,
            uv_index=filtered.uv_index,
            altitude=filtered.altitude,
            sensor_status=filtered.sensor_status,
            timestamp=filtered.timestamp,
            recv_time=filtered.recv_time,
            fusion_time=time.time(),
            statistics=stats
        )

        if api_data:
            fused.api_temperature = api_data.temperature
            fused.api_humidity = api_data.humidity
            fused.api_pressure = api_data.pressure
            fused.api_wind_speed = api_data.wind_speed
            fused.api_wind_direction = api_data.wind_direction
            fused.api_weather_text = api_data.weather_text
            fused.api_weather_code = api_data.weather_code
            fused.api_precip = api_data.precip
            fused.api_visibility = api_data.visibility
            fused.api_cloud_cover = api_data.cloud_cover

            fused.temp_diff = filtered.temperature - api_data.temperature
            fused.humidity_diff = filtered.humidity - api_data.humidity
            fused.pressure_diff = filtered.pressure - api_data.pressure

        return fused

    def to_llm_prompt(self, fused: FusedWeatherData) -> str:
        lines = [
            "## 输入数据说明",
            "",
            "### 当前传感器实测数据（本地采集）",
            f"- 温度: {fused.temperature:.1f}℃",
            f"- 湿度: {fused.humidity:.1f}%",
            f"- 气压: {fused.pressure:.1f} hPa",
            f"- 光照强度: {fused.light_intensity:.1f} lux",
            f"- 紫外线指数: {fused.uv_index:.1f}",
            f"- 传感器状态: {fused.sensor_status} (0=异常, 1=正常)",
        ]

        if fused.api_weather_text:
            lines.extend([
                "",
                "### 高德天气API数据（远程获取）",
                f"- 当前天气现象: {fused.api_weather_text}",
                f"- 当前温度: {fused.api_temperature:.1f}℃",
                f"- 湿度: {fused.api_humidity:.1f}%",
                f"- 能见度: {fused.api_visibility:.1f} km",
            ])

        lines.extend([
            "",
            "### 本地与API数据差异分析",
            f"- 温度差: {fused.temp_diff:+.1f}℃",
            f"- 湿度差: {fused.humidity_diff:+.1f}%",
            f"- 气压差: {fused.pressure_diff:+.1f} hPa",
        ])

        return "\n".join(lines)