import os
import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)

if os.environ.get('VLMEVAL_VLM_MINIMAL_IMPORT', '0').strip().lower() in {'1', 'true', 'yes', 'on'}:
    from .base import BaseModel
    from .gemma3_replay import Gemma3Replay
    from .gemma4_replay import Gemma4Replay
    from .minicpm_v_4_5_replay import MiniCPM_V_4_5, MiniCPM_V_4_5_Replay
    from .qwen3_vl import Qwen3VLChat
    from .qwen35_vl_replay import Qwen35VLChatReplay
else:
    import torch

    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    from .base import BaseModel
    from .gemma3_replay import Gemma3Replay
    from .gemma4_replay import Gemma4Replay
    from .cogvlm import CogVlm, GLM4v
    from .emu import Emu
    from .idefics import IDEFICS, IDEFICS2
    from .instructblip import InstructBLIP
    from .llava import LLaVA, LLaVA_Replay, LLaVA_HF, LLaVA_HF_Replay, LLaVA_Next, LLaVA_XTuner, LLaVA_Next2, LLaVA_OneVision
    from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6, MiniCPM_V_2_6_Replay
    from .minicpm_v_4_5_replay import MiniCPM_V_4_5, MiniCPM_V_4_5_Replay
    from .minigpt4 import MiniGPT4
    from .mmalaya import MMAlaya, MMAlaya2
    from .monkey import Monkey, MonkeyChat
    from .mplug_owl2 import mPLUG_Owl2
    from .omnilmm import OmniLMM12B
    from .open_flamingo import OpenFlamingo
    from .pandagpt import PandaGPT
    from .qwen_vl import QwenVL, QwenVLChat
    from .qwen2_vl import Qwen2VLChat, Qwen2VLChatReplay
    from .qwen3_vl import Qwen3VLChat
    from .qwen35_vl_replay import Qwen35VLChatReplay
    from .transcore_m import TransCoreM
    from .visualglm import VisualGLM
    from .xcomposer import ShareCaptioner, XComposer, XComposer2, XComposer2_4KHD, XComposer2d5
    from .yi_vl import Yi_VL
    from .internvl_chat import InternVLChat
    from .deepseek_vl import DeepSeekVL
    from .mgm import Mini_Gemini
    from .bunnyllama3 import BunnyLLama3
    from .vxverse import VXVERSE
    from .paligemma import PaliGemma
    from .qh_360vl import QH_360VL
    from .phi3_vision import Phi3Vision, Phi3_5Vision
    from .wemm import WeMM
    from .cambrian import Cambrian
    from .chameleon import Chameleon
    from .video_llm import VideoLLaVA, VideoLLaVA_HF, Chatunivi, VideoChatGPT, LLaMAVID, VideoChat2_HD, PLLaVA
    from .vila import VILA
    from .ovis import Ovis
    from .mantis import Mantis
    from .mixsense import LLama3Mixsense
    from .parrot import Parrot
    from .omchat import OmChat
    from .rbdash import RBDash
    from .xgen_mm import XGenMM
    from .slime import SliME
    from .mplug_owl3 import mPLUG_Owl3
    from .dual_thinker import DualThinker
    from .single_thinker_top import VLLMSingleThinkerTop
    from .dual_thinker_p2 import DualThinkerP2
    from .dual_thinker_p3 import DualThinkerP3
    from .dual_thinker_pout import DualThinkerPOut
    # from .duplexpts2outer import Duplexpts2outer
    from .duplex_pts2 import DuplexThinkerS2
    from .duplex_pts2_vllm import DuplexThinkerS2vLLM
    from .duplex_pts2_vllm_prefix import DuplexThinkerS2vLLMPrefix
    from .duplex_pts2_vllm_prefix_p3 import DuplexThinkerS2vLLMPrefixP3
    from .duplex_pts2_vllm_prefix_custom import DuplexThinkerS2vLLMPrefixCustom
    from .duplex_pts2_forward_vllm_prefix_custom import DuplexThinkerS2ForwardvLLMPrefixCustom

    from .duplex_pts31_uni import DuplexThinkerS31Uni
    from .qwen25vl_custom import Qwen25VLCustom
    from .qwen25vl_custom_prefix_custom import Qwen25VLCustomPrefixCustom
    from .qwen25vl_custom_vllm import Qwen25VLCustomvLLM
    from .qwen25vl_custom_replay_vllm import Qwen25VLCustomReplayvLLM
    from .qwen25vl_custom_prefix_custom_vllm import Qwen25VLCustomPrefixCustomvLLM
    from .qwen_gpt_vl import QwenGPTVL
    from .duplex_pts2_forward_vllm_prefix_custom_llava import DuplexThinkerS2ForwardvLLMPrefixCustomLLaVA
    from .duplex_pts2_forward_prefix_custom_llava import DuplexThinkerS2ForwardPrefixCustomLLaVA
    from .duplex_pts2_forward_vllm_prefix_custom_minicpmv import DuplexThinkerS2ForwardvLLMPrefixCustomMiniCPMV
