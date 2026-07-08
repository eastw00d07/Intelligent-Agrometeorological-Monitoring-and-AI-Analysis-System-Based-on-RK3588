"""
串口数据接收模块 - 从STM32F1接收传感器数据
支持两种格式:
1. 文本格式: T:27C H:31% L:40lx P:1001.26hPa UV:0.00
2. 二进制帧格式: [0xAA55][data_len][SensorData][CRC16][0x55AA]
"""
import re
import serial
import struct
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)

FRAME_HEADER = 0xAA55
FRAME_TAIL = 0x55AA
SENSOR_DATA_STRUCT = '<fffffffIB'
SENSOR_DATA_SIZE = struct.calcsize(SENSOR_DATA_STRUCT)
FRAME_HEADER_SIZE = 4
FRAME_CRC_SIZE = 2
FRAME_TAIL_SIZE = 2
FRAME_MIN_SIZE = FRAME_HEADER_SIZE + SENSOR_DATA_SIZE + FRAME_CRC_SIZE + FRAME_TAIL_SIZE

TEXT_PATTERN = re.compile(
    r'T:(?P<T>[^C]+)C\s+'
    r'H:(?P<H>[^%]+)%\s+'
    r'L:(?P<L>[^l]+)lx\s+'
    r'P:(?P<P>[^h]+)hPa\s+'
    r'UV:(?P<UV>[\d.]+)'
    r'(?:.*Alt:(?P<Alt>[\d.]+))?'
)


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
    recv_time: float = field(default_factory=time.time)


def crc16_modbus(buf: bytes) -> int:
    crc = 0xFFFF
    for b in buf:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def parse_text_line(line: str) -> Optional[SensorData]:
    m = TEXT_PATTERN.search(line)
    if not m:
        return None
    try:
        alt_str = m.group('Alt')
        altitude = float(alt_str) if alt_str else 0.0
        return SensorData(
            temperature=float(m.group('T')),
            humidity=float(m.group('H')),
            pressure=float(m.group('P')),
            light_intensity=float(m.group('L')),
            uv_index=float(m.group('UV')),
            altitude=altitude,
            timestamp=int(time.time()),
            sensor_status=1,
            recv_time=time.time()
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"文本行数值解析失败: {e}")
        return None


class SerialReceiver:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200,
                 timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[SensorData], None]] = None
        self._latest_data: Optional[SensorData] = None
        self._lock = threading.Lock()
        self._rx_buffer = bytearray()

    def set_callback(self, callback: Callable[[SensorData], None]):
        self._callback = callback

    def open(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            logger.info(f"串口打开成功: {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            logger.error(f"串口打开失败: {e}")
            return False

    def close(self):
        self.stop()
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("串口已关闭")

    def start(self) -> bool:
        if not self._serial or not self._serial.is_open:
            if not self.open():
                return False
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("串口接收线程已启动")
        return True

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("串口接收线程已停止")

    def get_latest_data(self) -> Optional[SensorData]:
        with self._lock:
            return self._latest_data

    def _recv_loop(self):
        while self._running:
            try:
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    self._rx_buffer.extend(data)
                    self._process_buffer()
                else:
                    time.sleep(0.01)
            except serial.SerialException as e:
                logger.error(f"串口读取异常: {e}")
                time.sleep(0.5)

    def _process_buffer(self):
        while b'\n' in self._rx_buffer:
            idx = self._rx_buffer.index(b'\n')
            line_bytes = bytes(self._rx_buffer[:idx])
            del self._rx_buffer[:idx + 1]

            line = line_bytes.decode('utf-8', errors='replace').strip('\r\n')
            if not line:
                continue

            if line.startswith('Pressure compensation') or line.startswith('pressure compensation'):
                continue

            sensor_data = parse_text_line(line)
            if sensor_data is not None:
                with self._lock:
                    self._latest_data = sensor_data
                if self._callback:
                    self._callback(sensor_data)
                logger.debug(f"文本解析成功: T={sensor_data.temperature}°C H={sensor_data.humidity}%")
                continue

            sensor_data = self._try_parse_binary_frame(line_bytes)
            if sensor_data is not None:
                with self._lock:
                    self._latest_data = sensor_data
                if self._callback:
                    self._callback(sensor_data)
                continue

            logger.debug(f"忽略行: {line[:80]}")

    def _try_parse_binary_frame(self, data: bytes) -> Optional[SensorData]:
        if len(data) < FRAME_MIN_SIZE:
            return None
        try:
            header = struct.unpack_from('<H', data, 0)[0]
            if header != FRAME_HEADER:
                return None

            data_len = struct.unpack_from('<H', data, 2)[0]
            if data_len != SENSOR_DATA_SIZE:
                return None

            offset = FRAME_HEADER_SIZE
            sensor_bytes = data[offset:offset + SENSOR_DATA_SIZE]

            crc_offset = offset + SENSOR_DATA_SIZE
            crc_received = struct.unpack_from('<H', data, crc_offset)[0]
            crc_calc = crc16_modbus(sensor_bytes)
            if crc_received != crc_calc:
                return None

            tail_offset = crc_offset + FRAME_CRC_SIZE
            tail = struct.unpack_from('<H', data, tail_offset)[0]
            if tail != FRAME_TAIL:
                return None

            values = struct.unpack(SENSOR_DATA_STRUCT, sensor_bytes)
            return SensorData(
                temperature=values[0],
                humidity=values[1],
                pressure=values[2],
                wind_speed=0.0,
                wind_direction=0.0,
                rainfall=values[5],
                light_intensity=values[6],
                uv_index=values[7],
                timestamp=values[8],
                sensor_status=values[9],
                recv_time=time.time()
            )
        except (struct.error, IndexError):
            return None
