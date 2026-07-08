"""
串口接收测试程序
1. 扫描可用串口设备
2. 读取并打印原始数据(十六进制)
3. 尝试按数据帧格式解析: [0xAA55][data_len][SensorData][CRC16][0x55AA]
4. 支持虚拟串口回环测试
"""
import serial
import struct
import time
import re
import argparse
import threading
import sys
from dataclasses import dataclass, field
from typing import Optional

FRAME_HEADER = 0xAA55
FRAME_TAIL = 0x55AA
SENSOR_DATA_STRUCT = '<fffffffIB'
SENSOR_DATA_SIZE = struct.calcsize(SENSOR_DATA_STRUCT)
FRAME_HEADER_SIZE = 4
FRAME_CRC_SIZE = 2
FRAME_TAIL_SIZE = 2
FRAME_MIN_SIZE = FRAME_HEADER_SIZE + SENSOR_DATA_SIZE + FRAME_CRC_SIZE + FRAME_TAIL_SIZE


@dataclass
class SensorData:
    temperature: float = 0.0
    humidity: float = 0.0
    pressure: float = 0.0

    rainfall: float = 0.0
    light_intensity: float = 0.0
    uv_index: float = 0.0
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


def build_test_frame(data: SensorData) -> bytes:
    sensor_bytes = struct.pack(
        SENSOR_DATA_STRUCT,
        data.temperature, data.humidity, data.pressure,
        0.0, 0.0, data.rainfall,
        data.light_intensity, data.uv_index,
        data.timestamp, data.sensor_status
    )
    crc = crc16_modbus(sensor_bytes)
    data_len = len(sensor_bytes)
    frame = struct.pack('<H', FRAME_HEADER)
    frame += struct.pack('<H', data_len)
    frame += sensor_bytes
    frame += struct.pack('<H', crc)
    frame += struct.pack('<H', FRAME_TAIL)
    return frame


def scan_ports():
    import glob
    print("=" * 50)
    print("扫描可用串口设备...")
    print("=" * 50)
    patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyS*']
    found = []
    for pat in patterns:
        for dev in sorted(glob.glob(pat)):
            try:
                s = serial.Serial(dev, baudrate=115200, timeout=0.5)
                s.close()
                found.append(dev)
                print(f"  [可用] {dev}")
            except (serial.SerialException, OSError) as e:
                print(f"  [不可用] {dev} - {e}")
    if not found:
        print("  未找到可用串口")
    return found


def print_hex(data: bytes, prefix: str = "  RX"):
    hex_str = ' '.join(f'{b:02X}' for b in data)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
    print(f"{prefix} [{len(data):3d}字节] {hex_str}  | {ascii_str}")


def parse_frame(buf: bytearray) -> Optional[SensorData]:
    if len(buf) < FRAME_MIN_SIZE:
        return None

    header = struct.unpack_from('<H', buf, 0)[0]
    if header != FRAME_HEADER:
        return None

    data_len = struct.unpack_from('<H', buf, 2)[0]
    if data_len != SENSOR_DATA_SIZE:
        return None

    offset = FRAME_HEADER_SIZE
    sensor_bytes = bytes(buf[offset:offset + SENSOR_DATA_SIZE])

    crc_offset = offset + SENSOR_DATA_SIZE
    crc_received = struct.unpack_from('<H', buf, crc_offset)[0]
    crc_calc = crc16_modbus(sensor_bytes)
    if crc_received != crc_calc:
        print(f"  [警告] CRC校验失败: 收到=0x{crc_received:04X}, 计算=0x{crc_calc:04X}")
        return None

    tail_offset = crc_offset + FRAME_CRC_SIZE
    tail = struct.unpack_from('<H', buf, tail_offset)[0]
    if tail != FRAME_TAIL:
        return None

    values = struct.unpack(SENSOR_DATA_STRUCT, sensor_bytes)
    return SensorData(
        temperature=values[0], humidity=values[1], pressure=values[2],
        wind_speed=values[3], wind_direction=values[4], rainfall=values[5],
        light_intensity=values[6], uv_index=values[7],
        timestamp=values[8], sensor_status=values[9],
        recv_time=time.time()
    )


def print_sensor_data(data: SensorData, frame_count: int):
    print(f"\n  ┌─────────── 第 {frame_count} 帧解析结果 ───────────┐")
    print(f"  │ 温度:     {data.temperature:>8.1f} °C")
    print(f"  │ 湿度:     {data.humidity:>8.1f} %")
    print(f"  │ 气压:     {data.pressure:>8.1f} hPa")

    print(f"  │ 降雨量:   {data.rainfall:>8.2f} mm")
    print(f"  │ 光照强度: {data.light_intensity:>8.1f} lux")
    print(f"  │ 紫外线:   {data.uv_index:>8.1f}")
    print(f"  │ 时间戳:   {data.timestamp:>8d}")
    print(f"  │ 传感器状态: {data.sensor_status}")
    print(f"  └──────────────────────────────────────┘")


TEXT_PATTERN = re.compile(
    r'T:(?P<T>[^C]+)C\s+'
    r'H:(?P<H>[^%]+)%\s+'
    r'L:(?P<L>[^l]+)lx\s+'
    r'P:(?P<P>[^h]+)hPa\s+'
    r'UV:(?P<UV>\S+)'
)


def parse_text_line(line: str) -> Optional[SensorData]:
    m = TEXT_PATTERN.search(line)
    if not m:
        return None
    try:
        return SensorData(
            temperature=float(m.group('T')),
            humidity=float(m.group('H')),
            pressure=float(m.group('P')),
            light_intensity=float(m.group('L')),
            uv_index=float(m.group('UV')),
            timestamp=int(time.time()),
            sensor_status=1,
            recv_time=time.time()
        )
    except (ValueError, TypeError):
        return None


def monitor_port(port: str, baudrate: int, raw: bool = False):
    print(f"\n打开串口: {port} @ {baudrate}")
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
    except serial.SerialException as e:
        print(f"串口打开失败: {e}")
        return

    print(f"串口已打开, 等待数据... (Ctrl+C 退出)")
    print(f"支持文本格式: T:27C H:31% L:40lx P:1001.26hPa UV:0.00")
    print(f"支持二进制帧: [AA55][data_len][SensorData({SENSOR_DATA_SIZE}B)][CRC16][55AA]\n")

    rx_buffer = bytearray()
    frame_count = 0
    raw_count = 0

    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                raw_count += len(data)

                if raw:
                    print_hex(data)

                rx_buffer.extend(data)

                while b'\n' in rx_buffer:
                    idx = rx_buffer.index(b'\n')
                    line_bytes = bytes(rx_buffer[:idx])
                    del rx_buffer[:idx + 1]

                    line = line_bytes.decode('utf-8', errors='replace').strip('\r\n')
                    if not line:
                        continue

                    if line.lower().startswith('pressure compensation'):
                        continue

                    parsed = parse_text_line(line)
                    if parsed is not None:
                        frame_count += 1
                        print_sensor_data(parsed, frame_count)
                        continue

                    if len(line_bytes) >= FRAME_MIN_SIZE:
                        parsed = parse_frame(bytearray(line_bytes))
                        if parsed is not None:
                            frame_count += 1
                            print_sensor_data(parsed, frame_count)
                            continue

                    if not raw:
                        print(f"  [忽略] {line[:80]}")
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n\n统计: 共接收 {raw_count} 字节, 解析 {frame_count} 帧")
    finally:
        ser.close()
        print("串口已关闭")


def loopback_test(port: str, baudrate: int, count: int = 5, interval: float = 1.0):
    print(f"\n串口回环测试: {port} @ {baudrate}")
    print(f"将发送 {count} 帧测试数据, 间隔 {interval}s")
    print("请确保 TX-RX 短接或连接了回环设备\n")

    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.5)
    except serial.SerialException as e:
        print(f"串口打开失败: {e}")
        return

    try:
        for i in range(count):
            test_data = SensorData(
                temperature=20.0 + i * 0.5,
                humidity=60.0 + i * 2.0,
                pressure=1013.0 + i * 0.1,

                rainfall=0.0,
                light_intensity=30000.0 + i * 1000.0,
                uv_index=3.0 + i * 0.5,
                timestamp=int(time.time()),
                sensor_status=1
            )

            frame = build_test_frame(test_data)
            ser.write(frame)
            ser.flush()

            print(f"发送第 {i+1}/{count} 帧 ({len(frame)} 字节):")
            print(f"  T={test_data.temperature:.1f}°C, H={test_data.humidity:.1f}%, "
                  f"P={test_data.pressure:.1f}hPa")

            time.sleep(0.1)

            rx_data = bytearray()
            deadline = time.time() + 0.5
            while time.time() < deadline:
                if ser.in_waiting > 0:
                    rx_data.extend(ser.read(ser.in_waiting))
                else:
                    time.sleep(0.01)

            if rx_data:
                parsed = parse_frame(rx_data)
                if parsed:
                    print(f"  回环接收成功!")
                    print(f"  T={parsed.temperature:.1f}°C, H={parsed.humidity:.1f}%, "
                          f"P={parsed.pressure:.1f}hPa")
                else:
                    print(f"  回环接收 {len(rx_data)} 字节, 但解析失败:")
                    print_hex(rx_data, "  ")
            else:
                print(f"  未收到回环数据")

            if i < count - 1:
                time.sleep(interval)

        print(f"\n回环测试完成, 共发送 {count} 帧")

    except KeyboardInterrupt:
        print("\n测试中断")
    finally:
        ser.close()
        print("串口已关闭")


def main():
    parser = argparse.ArgumentParser(description='串口接收测试程序')
    sub = parser.add_subparsers(dest='command', help='子命令')

    scan_p = sub.add_parser('scan', help='扫描可用串口')

    mon_p = sub.add_parser('monitor', help='监听串口数据')
    mon_p.add_argument('-p', '--port', default='/dev/ttyUSB0', help='串口设备')
    mon_p.add_argument('-b', '--baudrate', type=int, default=115200, help='波特率')
    mon_p.add_argument('--raw', action='store_true', help='显示原始十六进制数据')

    loop_p = sub.add_parser('loopback', help='回环测试(需TX-RX短接)')
    loop_p.add_argument('-p', '--port', default='/dev/ttyUSB0', help='串口设备')
    loop_p.add_argument('-b', '--baudrate', type=int, default=115200, help='波特率')
    loop_p.add_argument('-n', '--count', type=int, default=5, help='测试帧数')
    loop_p.add_argument('-i', '--interval', type=float, default=1.0, help='发送间隔(秒)')

    args = parser.parse_args()

    if args.command == 'scan':
        scan_ports()
    elif args.command == 'monitor':
        monitor_port(args.port, args.baudrate, args.raw)
    elif args.command == 'loopback':
        loopback_test(args.port, args.baudrate, args.count, args.interval)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python3 serial_test.py scan              # 扫描串口")
        print("  python3 serial_test.py monitor           # 监听ttyUSB0")
        print("  python3 serial_test.py monitor --raw     # 监听并显示原始数据")
        print("  python3 serial_test.py monitor -p /dev/ttyACM0  # 指定串口")
        print("  python3 serial_test.py loopback          # 回环测试")


if __name__ == '__main__':
    main()