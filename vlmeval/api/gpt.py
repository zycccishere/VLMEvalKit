from ..smp import *
import json
import os
import sys
import time
import fcntl
from .base import BaseAPI
from .ori_gpt_client import OpenAIChatClient
import code, traceback, signal

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
    'MODELBEST': 'https://llm-center.ali.modelbest.cn/llm/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
        'gpt-4o': 16384,
        'gpt-5': 128000
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


def resolve_openai_key_from_env() -> str:
    return (
        os.environ.get('OPENAI_API_KEY', '').strip()
        or os.environ.get('OPENAI_API_KEY_JUDGE', '').strip()
        or os.environ.get('OPENAI_COMPATIBLE_API_KEY', '').strip()
        or os.environ.get('MODELBEST_API_KEY', '').strip()
        or os.environ.get('LLM_CENTER_API_KEY', '').strip()
    )


def normalize_openai_compatible_api_base(api_base: str) -> str:
    base = (api_base or '').strip().rstrip('/')
    if not base:
        return base
    low = base.lower()
    if low.endswith('/llm'):
        base = base + '/v1/chat/completions'
    elif low.endswith('/v1'):
        base = base + '/chat/completions'
    elif low.endswith('/v1/chat'):
        base = base + '/completions'
    return base


def throttle_openai_compatible_request():
    """Optional cross-process request throttle for low-RPM OpenAI-compatible keys."""
    min_interval = float(os.environ.get('VLMEVAL_API_MIN_INTERVAL_SECONDS', '0') or 0)
    if min_interval <= 0:
        return
    lock_path = os.environ.get(
        'VLMEVAL_API_RATE_LIMIT_LOCK',
        os.path.expanduser('~/.cache/vlmeval_api_rate_limit.lock'),
    )
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, 'a+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        try:
            last_time = float(raw)
        except ValueError:
            last_time = 0.0
        now = time.time()
        wait_s = min_interval - (now - last_time)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.time()
        f.seek(0)
        f.truncate()
        f.write(f'{now:.6f}')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _find_replay_meta(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('replay_meta'), dict):
            return obj['replay_meta']
        for value in obj.values():
            found = _find_replay_meta(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_replay_meta(value)
            if found:
                return found
    return {}


def _estimate_cost_from_usage(model: str, usage: dict) -> dict:
    # USD per 1M tokens. These defaults are for monitoring estimates only;
    # final accounting should replace them with the billing source of record.
    rates = {
        'gpt-4o-mini': {'input': 0.15, 'cached_input': 0.075, 'output': 0.60},
        'gpt-4o-mini-2024-07-18': {'input': 0.15, 'cached_input': 0.075, 'output': 0.60},
        'gpt-5-mini': {'input': 0.25, 'cached_input': 0.025, 'output': 2.00},
        'gpt-5-chat': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
        'claude-haiku-4-5-20251001': {'input': 1.00, 'cached_input': 0.10, 'output': 5.00},
        'gemini-2.5-flash-lite': {'input': 0.10, 'cached_input': 0.025, 'output': 0.40},
        'gemini-2.5-flash-nothinking': {'input': 0.30, 'cached_input': 0.075, 'output': 2.50},
        'gemini-2.5-flash-thinking': {'input': 0.30, 'cached_input': 0.075, 'output': 2.50},
        'gemini-3-flash-preview-nothinking': {'input': 0.50, 'cached_input': 0.05, 'output': 3.00},
        'gemini-3.1-flash-lite': {'input': 0.25, 'cached_input': 0.25, 'output': 1.50},
    }
    rate = rates.get(model)
    if not rate or not isinstance(usage, dict):
        return {'cost_usd': None, 'rate_note': 'missing_rate_or_usage'}
    prompt_tokens = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
    completion_tokens = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
    prompt_details = usage.get('prompt_tokens_details') or {}
    cached_tokens = int(prompt_details.get('cached_tokens') or usage.get('cached_input_tokens') or 0)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    cost_usd = (
        uncached_tokens * rate['input']
        + cached_tokens * rate['cached_input']
        + completion_tokens * rate['output']
    ) / 1_000_000
    return {
        'cost_usd': cost_usd,
        'rate_note': 'default_monitoring_rate_table',
        'uncached_prompt_tokens': uncached_tokens,
        'cached_prompt_tokens': cached_tokens,
    }


def log_openai_compatible_usage_event(
    *,
    model: str,
    inputs,
    payload: dict,
    api_base: str,
    status_code: int,
    ret_code,
    latency_s: float,
    response_json,
):
    log_file = os.environ.get('VLMEVAL_API_USAGE_LOG_FILE') or os.environ.get('TOKEN_USAGE_LOG_FILE')
    if not log_file:
        return
    usage = response_json.get('usage') if isinstance(response_json, dict) else None
    meta = _find_replay_meta(inputs)
    entry = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'model': model,
        'api_base': api_base,
        'status_code': status_code,
        'ret_code': ret_code,
        'latency_s': round(latency_s, 3),
        'usage_present': isinstance(usage, dict),
        'usage': usage if isinstance(usage, dict) else None,
        'cost': _estimate_cost_from_usage(model, usage if isinstance(usage, dict) else {}),
        'request_max_tokens': payload.get('max_tokens'),
        'temperature': payload.get('temperature'),
        'replay_mode': os.environ.get('REPLAY_MODE'),
        'replay_image_transform': os.environ.get('REPLAY_IMAGE_TRANSFORM'),
        'replay_prompt_template_name': os.environ.get('REPLAY_PROMPT_TEMPLATE_NAME'),
        'work_dir': os.environ.get('MMEVAL_ROOT'),
        'sample_index': meta.get('sample_index'),
        'dataset_name': meta.get('dataset_name'),
        'source_first_image_ref': meta.get('source_first_image_ref'),
    }
    log_path = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)

class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step-1v' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                # assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                # assert isinstance(key, str), (
                #     'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                # )
            else:
                env_key = resolve_openai_key_from_env()
                if key is None:
                    key = env_key
                # assert isinstance(key, str) and key.startswith('sk-'), (
                #     f'Illegal openai_key {key}. '
                #     'Please set the environment variable OPENAI_API_KEY to your openai key. '
                # )

        self.key = key
        # assert img_size > 0 or img_size == -1
        self.img_size = img_size
        # assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                elif 'OPENAI_API_BASE_JUDGE' in os.environ and os.environ['OPENAI_API_BASE_JUDGE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE_JUDGE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE_JUDGE']
                elif 'OPENAI_COMPATIBLE_API_BASE' in os.environ and os.environ['OPENAI_COMPATIBLE_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_COMPATIBLE_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_COMPATIBLE_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                sys.exit(-1)

        self.logger.info(f'Using API Base: {self.api_base}; API Key: <redacted>')
        print(f'Init finished', flush=True)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        # print(f'Input messages: {input_msgs}', flush=True)

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            self.logger.warning(
                'Less than 100 tokens left, '
                'may exceed the context window with some additional meta symbols. '
            )
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        chat_template_kwargs_raw = os.environ.get('VLMEVAL_OPENAI_CHAT_TEMPLATE_KWARGS', '').strip()
        if chat_template_kwargs_raw:
            try:
                chat_template_kwargs = json.loads(chat_template_kwargs_raw)
                if isinstance(chat_template_kwargs, dict):
                    payload['chat_template_kwargs'] = chat_template_kwargs
                else:
                    self.logger.warning(
                        'Ignoring VLMEVAL_OPENAI_CHAT_TEMPLATE_KWARGS because it is not a JSON object: '
                        f'{chat_template_kwargs_raw!r}'
                    )
            except Exception as e:
                self.logger.warning(
                    f'Ignoring invalid VLMEVAL_OPENAI_CHAT_TEMPLATE_KWARGS={chat_template_kwargs_raw!r}: {e}'
                )
        enable_thinking = os.environ.get('VLMEVAL_OPENAI_ENABLE_THINKING', '').strip().lower()
        if enable_thinking in {'0', 'false', 'no'}:
            payload.setdefault('chat_template_kwargs', {})['enable_thinking'] = False

        # # START ORIGIN ##
        # response = requests.post(
        #     self.api_base,
        #     headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)

        # END ORIGIN ##

        # Use native chat-completions payload to keep multimodal message support.
        api_base = getattr(self, 'api_base', None)
        if not api_base:
            # Compatibility fallback for branches where __init__ does not set self.api_base.
            env_base = (
                os.environ.get('OPENAI_API_BASE', '').strip()
                or os.environ.get('OPENAI_API_BASE_JUDGE', '').strip()
                or os.environ.get('OPENAI_COMPATIBLE_API_BASE', '').strip()
                or os.environ.get('MODELBEST_BASE_URL', '').strip()
                or os.environ.get('LLM_CENTER_BASE_URL', '').strip()
            )
            api_base = env_base if env_base else APIBASES['OFFICIAL']
            self.api_base = api_base
        if isinstance(api_base, str):
            api_base = normalize_openai_compatible_api_base(api_base)
        throttle_openai_compatible_request()
        request_started = time.time()
        response = requests.post(
            api_base,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout * 1.1,
        )
        latency_s = time.time() - request_started
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        resp_struct = None
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception:
            pass
        log_openai_compatible_usage_event(
            model=self.model,
            inputs=inputs,
            payload=payload,
            api_base=api_base,
            status_code=response.status_code,
            ret_code=ret_code,
            latency_s=latency_s,
            response_json=resp_struct,
        )
        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        """
        Estimate token length for inputs.
        Falls back to character-based estimation if tiktoken fails (e.g., no network).
        """
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except:
                enc = tiktoken.encoding_for_model('gpt-4')

            assert isinstance(inputs, list)
            tot = 0
            for item in inputs:
                if 'role' in item:
                    tot += self.get_token_len(item['content'])
                elif item['type'] == 'text':
                    tot += len(enc.encode(item['value']))
                elif item['type'] == 'image':
                    tot += self.get_image_token_len(item['value'], detail=self.img_detail)
            return tot
        except Exception as e:
            # Fallback: estimate ~4 characters per token (rough approximation)
            self.logger.warning(f"tiktoken failed ({type(e).__name__}), using character-based estimation")
            assert isinstance(inputs, list)
            tot = 0
            for item in inputs:
                if 'role' in item:
                    tot += self.get_token_len(item['content'])
                elif item['type'] == 'text':
                    tot += len(item['value']) // 4 + 1
                elif item['type'] == 'image':
                    tot += self.get_image_token_len(item['value'], detail=self.img_detail)
            return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
