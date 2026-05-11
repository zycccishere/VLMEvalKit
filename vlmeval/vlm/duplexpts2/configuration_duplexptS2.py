from transformers import Qwen2Config
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DuplexPTS2Config(Qwen2Config):
    model_type = "duplexpt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_cache=True,
        perceiver_name_or_path='/path/to/home/models/Qwen2.5-VL-3B-Instruct',
        thinker_name_or_path='/path/to/home/models/Qwen3-4B',
        t_tokenizer_name_or_path='/path/to/home/duplex_think/duplex_qwen3_tokenizer',
        p_tokenizer_name_or_path='/path/to/home/duplex_think/duplex_qwen25_processor',
        visual_bandwidth=1,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.perceiver_name_or_path = perceiver_name_or_path
        self.thinker_name_or_path = thinker_name_or_path
        self.t_tokenizer_name_or_path = t_tokenizer_name_or_path
        self.p_processor_name_or_path = p_tokenizer_name_or_path


        self.visual_bandwidth = visual_bandwidth

        super().__init__(**kwargs)
