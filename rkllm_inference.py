"""
RKLLM气象预测推理模块
通过llm_demo子进程调用RK3588 NPU上的DeepSeek-R1-Distill-Qwen-1.5B模型
避免ctypes处理ARM64结构体返回的问题
"""
import os
import time
import logging
import threading
import subprocess
import select
from typing import Optional, Callable

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
DEFAULT_DEMO_PATH = "/userdata/home/elf/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM/demo_Linux_aarch64/llm_demo"


class RKLLMInference:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 demo_path: str = DEFAULT_DEMO_PATH,
                 max_new_tokens: int = 2048,
                 max_context: int = 4096,
                 lib_path: str = "",
                 **kwargs):
        self.model_path = model_path
        self.demo_path = demo_path
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._stream_callback: Optional[Callable[[str], None]] = None

    def set_stream_callback(self, callback: Callable[[str], None]):
        self._stream_callback = callback

    def init(self) -> bool:
        if self._initialized:
            return True

        if not os.path.exists(self.demo_path):
            logger.error(f"llm_demo不存在: {self.demo_path}")
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"RKLLM模型不存在: {self.model_path}")
            return False

        try:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = os.path.dirname(self.demo_path) + "/lib" + \
                (":" + env.get("LD_LIBRARY_PATH", ""))
            self._process = subprocess.Popen(
                [self.demo_path, self.model_path,
                 str(self.max_new_tokens), str(self.max_context)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env
            )
            logger.info(f"llm_demo进程已启动, PID={self._process.pid}")
        except OSError as e:
            logger.error(f"llm_demo启动失败: {e}")
            return False

        init_output = self._read_until_prompt(timeout=120)
        if init_output is None:
            logger.error("llm_demo初始化超时或失败")
            stderr_out = ""
            try:
                stderr_out = self._process.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if stderr_out:
                logger.error(f"stderr: {stderr_out}")
            self.destroy()
            return False

        self._initialized = True
        logger.info(f"RKLLM初始化成功, 模型: {self.model_path}")
        return True

    def _read_until_prompt(self, timeout: float = 300) -> Optional[str]:
        output_parts = []
        deadline = time.time() + timeout
        buf = b""
        json_depth = 0
        in_json = False

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                ready, _, _ = select.select([self._process.stdout], [], [], min(remaining, 0.5))
            except (ValueError, OSError):
                break

            if ready:
                try:
                    chunk = os.read(self._process.stdout.fileno(), 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace")
                output_parts.append(text)
                if self._stream_callback:
                    self._stream_callback(text + "\n")
                
                # 检测JSON对象开始和结束
                for char in text:
                    if char == '{':
                        json_depth += 1
                        in_json = True
                    elif char == '}':
                        json_depth -= 1
                        if json_depth == 0 and in_json:
                            # JSON完整，可以返回
                            result = "\n".join(output_parts)
                            return result

            if self._process.poll() is not None:
                if buf:
                    text = buf.decode("utf-8", errors="replace")
                    output_parts.append(text)
                break

        return None if not output_parts else "\n".join(output_parts)

    def run(self, prompt: str) -> Optional[str]:
        if not self._initialized or not self._process:
            logger.error("RKLLM未初始化")
            return None

        try:
            self._process.stdin.write((prompt + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            logger.error(f"写入llm_demo失败: {e}")
            return None

        result = self._read_until_prompt(timeout=300)
        if result is None:
            logger.error("推理超时或进程异常")
            return None

        return result.strip() if result.strip() else None

    def run_async(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> threading.Thread:
        def _run_thread():
            result = self.run(prompt)
            if callback and result:
                callback(result)

        thread = threading.Thread(target=_run_thread, daemon=True)
        thread.start()
        return thread

    def destroy(self):
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._initialized = False
        logger.info("RKLLM已销毁")


class WeatherPredictor:
    SYSTEM_PROMPT = """根据气象数据，输出预测JSON。直接输出JSON，不要解释。

必须包含以下字段：prediction(含temp_trend,precipitation,weather), alert, confidence, advice。

示例：
{"prediction":{"temp_trend":{"direction":"stable","direction_zh":"平稳","delta":0.0,"confidence":70},"precipitation":{"probability":20,"intensity":"none","intensity_zh":"无","confidence":70},"weather":{"type":"overcast","type_zh":"阴","uv_index":2,"uv_level":"low","uv_level_zh":"低","confidence":75}},"alert":{"has_alert":false,"level":"none","level_zh":"无","type":"none","type_zh":"无","description":"无","action":"无"},"confidence":{"overall":70,"factors":{"sensor_status":80,"data_consistency":70,"history_completeness":60},"explanation":"正常"},"advice":{"travel":"适合","clothing":"薄外套","protection":"防晒","activity":"户外"}}

现在根据以下数据输出JSON："""

    def __init__(self, rkllm: RKLLMInference):
        self.rkllm = rkllm
        self._last_result: Optional[str] = None
        self._last_time: float = 0

    def predict(self, data_prompt: str) -> Optional[str]:
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{data_prompt}"
        logger.info("开始RKLLM气象推理...")
        logger.info(f"完整Prompt:\n{full_prompt}")
        start_time = time.time()

        result = self.rkllm.run(full_prompt)

        elapsed = time.time() - start_time
        if result:
            self._last_result = result
            self._last_time = time.time()
            logger.info(f"推理完成, 耗时{elapsed:.2f}s, 输出{len(result)}字符")
            logger.info(f"LLM完整输出:\n{result}")
        else:
            logger.error("推理失败")

        return result

    def predict_async(self, data_prompt: str,
                      callback: Optional[Callable[[str], None]] = None) -> threading.Thread:
        def _predict_thread():
            result = self.predict(data_prompt)
            if callback:
                callback(result)

        thread = threading.Thread(target=_predict_thread, daemon=True)
        thread.start()
        return thread

    def get_last_result(self) -> Optional[str]:
        return self._last_result

    def get_last_time(self) -> float:
        return self._last_time
