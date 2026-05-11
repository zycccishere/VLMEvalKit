import openai
import time
import os
import random
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
import requests  # 引入 requests 库以处理超时异常

# Configure logger
logger = logging.getLogger('OpenAIChatClient')

# 模型名称常量
MODEL_GPT4 = "gpt-4"
MODEL_GPT4_TURBO = "gpt-4-turbo"
MODEL_GPT4o = "gpt-4o"
MODEL_GPT4o_mini = "gpt-4o-mini"
MODEL_GPT35_TURBO = "gpt-3.5-turbo"
MODEL_GPT41 = "gpt-4.1"

DEFAULT_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_API_KEY = ""

# Connection error types that need exponential backoff
CONNECTION_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionResetError,
    ConnectionError,
    TimeoutError,
)
def resolve_default_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    )


def normalize_openai_sdk_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return base
    low = base.lower()
    if low.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
        low = base.lower()
    elif low.endswith("/completions") and low.endswith("/v1/completions"):
        base = base[: -len("/completions")]
        low = base.lower()
    if low.endswith("/llm"):
        base = base + "/v1"
    elif low.endswith("/v1/chat"):
        base = base[: -len("/chat")]
    return base


def resolve_default_base_url() -> str:
    raw = (
        os.environ.get("OPENAI_API_BASE_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_BASE", "").strip()
        or DEFAULT_BASE_URL
    )
    return normalize_openai_sdk_base_url(raw)


class OpenAIChatClient:
    """
    OpenAI 聊天客户端封装类，支持超时机制。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 organization: Optional[str] = None):
        """
        初始化 OpenAI 聊天客户端。

        Args:
            api_key: OpenAI API Key
            base_url: 可选的 base_url（如使用代理或自建代理）
            organization: 可选的 organization ID
        """
        resolved_key = api_key or resolve_default_api_key()
        resolved_base = normalize_openai_sdk_base_url(base_url or resolve_default_base_url())
        openai.api_key = resolved_key
        if resolved_base:
            openai.base_url = resolved_base
        if organization:
            openai.organization = organization

    def chat_sync(self,
                  user_prompt: str,
                  system_prompt: str = 'You are a helpful assistant.',
                  model: str = MODEL_GPT4o,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  tools: Optional[list] = None,
                  timeout: float = 60.0) -> Tuple[str, Dict[str, Any]]:
        """
        同步聊天请求，添加超时机制。

        Args:
            user_prompt: 用户输入
            system_prompt: 系统提示
            model: 使用的模型名称
            max_tokens: 最大回复长度
            temperature: 控制生成的随机程度
            tools: 可选，传入 tool_calls 时使用
            timeout: 请求超时时间（秒）

        Returns:
            Tuple[str, Dict[str, Any]]: 返回模型回复与完整响应

        Raises:
            TimeoutError: 如果请求超时
            RuntimeError: 其他 OpenAI API 错误
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools or [],
                tool_choice="auto" if tools else None,
                timeout=timeout  # 设置请求超时
            )
            reply = response.choices[0].message.content
            raw_response = response.model_dump()

            # 记录 token 使用情况（如果设置了环境变量）
            token_log_file = os.environ.get('TOKEN_USAGE_LOG_FILE', None)
            if token_log_file is not None:
                try:
                    import json
                    from datetime import datetime

                    # 从响应中提取 token 使用信息
                    usage_info = {
                        'input_tokens': 0,
                        'cached_input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                    }

                    # 处理字典格式的响应
                    if isinstance(raw_response, dict) and 'usage' in raw_response:
                        usage = raw_response['usage']
                        usage_info['input_tokens'] = usage.get('prompt_tokens', 0)
                        usage_info['output_tokens'] = usage.get('completion_tokens', 0)
                        usage_info['total_tokens'] = usage.get('total_tokens', 0)

                        # 从 prompt_tokens_details 中提取 cached_tokens
                        prompt_details = usage.get('prompt_tokens_details', {})
                        if isinstance(prompt_details, dict):
                            usage_info['cached_input_tokens'] = prompt_details.get('cached_tokens', 0)

                    # 如果 total_tokens 为 0，尝试计算
                    if usage_info['total_tokens'] == 0:
                        usage_info['total_tokens'] = usage_info['input_tokens'] + usage_info['output_tokens']

                    # 添加其他元信息
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'model': model,
                        'usage': usage_info,
                    }

                    # 确保日志目录存在
                    log_dir = os.path.dirname(token_log_file)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)

                    # 追加写入 JSONL 文件
                    with open(token_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                        f.flush()

                    logger.debug(f"Token usage logged: {usage_info}")
                except Exception as e:
                    logger.warning(f"Failed to log token usage: {e}")

            return reply, raw_response

        except requests.exceptions.Timeout:
            raise TimeoutError(f"OpenAI API request timed out after {timeout} seconds")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"OpenAI chat failed: {e}")

    def chat_sync_retry(self,
                        user_prompt: str,
                        system_prompt: str = 'You are a helpful assistant.',
                        model: str = MODEL_GPT4o,
                        max_tokens: int = 1024,
                        temperature: float = 0.1,
                        max_retry: int = 10,
                        timeout: float = 60.0) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        带重试机制的聊天，支持超时和指数退避。

        Args:
            user_prompt: 用户输入
            system_prompt: 系统提示
            model: 使用的模型名称
            max_tokens: 最大回复长度
            temperature: 控制生成的随机程度
            max_retry: 最大重试次数
            timeout: 单次请求超时时间（秒）

        Returns:
            成功返回 (回复, 完整响应)，失败返回 None
        """
        for attempt in range(max_retry):
            try:
                return self.chat_sync(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
            except CONNECTION_ERRORS as err:
                # Connection errors need exponential backoff
                base_delay = min(2 ** (attempt + 1), 60)  # 2, 4, 8, 16, 32, 60, 60...
                delay = base_delay * (0.5 + random.random())  # Add jitter
                logger.warning(
                    f"{model} 连接错误 {attempt + 1}/{max_retry}: {type(err).__name__}: {str(err)[:80]}"
                )
                logger.info(f"等待 {delay:.1f}s 后重试...")
                if attempt < max_retry - 1:
                    time.sleep(delay)
            except Exception as err:
                # Other errors use shorter delay
                logger.error(f"{model} 尝试 {attempt + 1}/{max_retry} 失败: {err}")
                if attempt < max_retry - 1:
                    time.sleep(2 + random.random() * 2)
        return None


client = OpenAIChatClient()

# 示例用法
if __name__ == "__main__":
    try:
        reply, raw = client.chat_sync("What is your model series id?", model='gpt-4o-mini', timeout=10.0)
        print("模型回复：", reply)
    except TimeoutError as e:
        print(f"请求超时: {e}")
    except RuntimeError as e:
        print(f"请求失败: {e}")
