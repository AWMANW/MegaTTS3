# tts\infer_srt_cli.py (Cleaned Version - Library Focus)

import json
import os
import argparse
import librosa
import numpy as np
import torch
import logging
import io
import time

# --- 假设从项目根目录运行，使用标准导入 ---
try:
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    from langdetect import detect as classify_language
    from pydub import AudioSegment
    import pyloudnorm as pyln

    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
    from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
    from tts.utils.audio_utils.io import save_wav, to_wav_bytes, convert_to_wav_bytes, combine_audio_segments
    from tts.utils.commons.ckpt_utils import load_ckpt
    from tts.utils.commons.hparams import set_hparams, hparams
    from tts.utils.text_utils.text_encoder import TokenTextEncoder
    from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinesev2
except ImportError as e:
    print(f"Import Error: {e}. Ensure script is run from project root or PYTHONPATH is set.")
    raise

# --- 日志配置 (简化) ---
log = logging.getLogger(__name__)
# (日志级别和格式由调用者设置，例如 app.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Helper Functions (保持不变，但移除 print，使用 log) ---
def convert_to_wav(wav_path):
    # ... (使用 log.info, log.error) ...
    if not os.path.exists(wav_path): log.error(f"File not found: {wav_path}"); return
    if not wav_path.lower().endswith(".wav"):
        out_path = os.path.splitext(wav_path)[0] + ".wav"
        try:
            log.info(f"Converting '{wav_path}' to '{out_path}'")
            audio = AudioSegment.from_file(wav_path)
            audio.export(out_path, format="wav")
            log.info(f"Conversion successful.")
        except Exception as e: log.error(f"Error converting {wav_path} to WAV: {e}", exc_info=True)

def cut_wav(wav_path, max_len=28):
    # ... (使用 log.info, log.error) ...
     if not os.path.exists(wav_path): log.error(f"File not found for cutting: {wav_path}"); return
     try:
        log.info(f"Cutting '{wav_path}' to max_len={max_len}s")
        audio = AudioSegment.from_file(wav_path)
        audio = audio[:int(max_len * 1000)]
        audio.export(wav_path, format="wav") # Overwrites original
        log.info(f"Cutting successful.")
     except Exception as e: log.error(f"Error cutting {wav_path}: {e}", exc_info=True)

# --- MegaTTS3DiTInfer Class Definition (清理日志) ---
class MegaTTS3DiTInfer():
    def __init__(
            self,
            device=None,
            ckpt_root='./checkpoints',
            dit_exp_name='diffusion_transformer',
            frontend_exp_name='aligner_lm',
            wavvae_exp_name='wavvae',
            dur_ckpt_path='duration_lm',
            g2p_exp_name='g2p',
            precision=torch.float16,
            **kwargs
        ):
        # --- 使用 log.info 记录关键信息 ---
        log.info(f"Initializing MegaTTS3DiTInfer. CKPT Root: {ckpt_root}")
        self.sr = 24000
        self.fm = 8

        # --- 设备和精度处理 (保持方法一) ---
        if device is None: resolved_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif isinstance(device, torch.device): resolved_device_str = str(device).split(':')[0]
        elif isinstance(device, str): resolved_device_str = device.lower().split(':')[0]
        else: log.warning(f"Invalid device type: {type(device)}. Defaulting CPU."); resolved_device_str = 'cpu'
        self.device = torch.device(resolved_device_str)

        if isinstance(precision, str): # 处理精度字符串
             if precision.lower() in ['fp16', 'float16']: self.precision = torch.float16
             elif precision.lower() in ['bf16', 'bfloat16']: self.precision = torch.bfloat16
             else: self.precision = torch.float32
        else: self.precision = precision # 假设是 torch.dtype
        log.info(f"  Using Device: {self.device}, Precision: {self.precision}")

        # build models (使用绝对路径)
        abs_ckpt_root = os.path.abspath(ckpt_root)
        log.info(f"  Absolute Checkpoint Root: {abs_ckpt_root}")
        self.dit_exp_name = os.path.join(abs_ckpt_root, dit_exp_name)
        self.frontend_exp_name = os.path.join(abs_ckpt_root, frontend_exp_name)
        self.wavvae_exp_name = os.path.join(abs_ckpt_root, wavvae_exp_name)
        self.dur_exp_name = os.path.join(abs_ckpt_root, dur_ckpt_path)
        self.g2p_exp_name = os.path.join(abs_ckpt_root, g2p_exp_name)
        log.info("  Building model components...")
        self.build_model(self.device)
        log.info("  Model components built.")

        # init text normalizer & meter
        try:
            self.zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
            self.en_normalizer = EnNormalizer(overwrite_cache=False)
            self.loudness_meter = pyln.Meter(self.sr)
            log.info("  Normalizers and Loudness Meter initialized.")
        except Exception as e: log.error(f"Failed to initialize normalizers/meter: {e}", exc_info=True); raise
        log.info("MegaTTS3DiTInfer Initialized.")

    def build_model(self, device: torch.device):
        log.info(f"Setting hparams for DiT: {self.dit_exp_name}")
        try: hparams_dit = set_hparams(exp_name=self.dit_exp_name, print_hparams=False, global_hparams=False)
        except Exception as e: log.error(f"Failed to set hparams for DiT: {e}", exc_info=True); raise

        log.info("Loading ling dict...")
        try:
             # 假设 utils 在 tts 目录下
             current_script_dir = os.path.dirname(os.path.abspath(__file__))
             ling_dict_path = os.path.join(current_script_dir, "utils", "text_utils", "dict.json")
             log.info(f"  Loading dict from: {ling_dict_path}")
             with open(ling_dict_path, 'r', encoding='utf-8-sig') as f: ling_dict = json.load(f)
             self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
             self.token_encoder = self.ling_dict['phone']
             log.info(f"  Ling dict loaded. Phone vocab size: {len(self.token_encoder)}")
        except Exception as e: log.error(f"Error loading ling dict: {e}", exc_info=True); raise

        log.info(f"Loading Duration LM from: {self.dur_exp_name}")
        try:
            from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
            dur_config_path = os.path.join(self.dur_exp_name, 'config.yaml')
            hp_dur_model = self.hp_dur_model = set_hparams(dur_config_path, global_hparams=False)
            frames_multiple = hparams_dit.get('frames_multiple', hp_dur_model.get('frames_multiple', 1))
            hp_dur_model['frames_multiple'] = frames_multiple
            ph_dict_size = len(self.token_encoder)
            self.dur_model = ARDurPredictor(hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'], hp_dur_model['dur_model_layers'], ph_dict_size, hp_dur_model['dur_code_size'], use_rot_embed=hp_dur_model.get('use_rot_embed', False))
            self.length_regulator = LengthRegulator()
            log.info(f"  Loading Duration checkpoint...")
            load_ckpt(self.dur_model, self.dur_exp_name, 'dur_model') # 移除 verbose
            self.dur_model.eval().to(device)
        except Exception as e: log.error(f"Failed to load Duration LM: {e}", exc_info=True); raise

        log.info(f"Loading Diffusion Transformer from: {self.dit_exp_name}")
        try:
            from tts.modules.llm_dit.dit import Diffusion
            self.dit = Diffusion()
            log.info(f"  Loading DiT checkpoint...")
            load_ckpt(self.dit, self.dit_exp_name, 'dit', strict=False) # 移除 verbose
            self.dit.eval().to(device)
            self.cfg_mask_token_phone = hparams_dit.get('vocab_size', 302) - 1
            self.cfg_mask_token_tone = hparams_dit.get('tone_vocab_size', 32) - 1
        except Exception as e: log.error(f"Failed to load DiT: {e}", exc_info=True); raise

        log.info(f"Loading Aligner LM from: {self.frontend_exp_name}")
        try:
            from tts.modules.aligner.whisper_small import Whisper
            self.aligner_lm = Whisper()
            log.info(f"  Loading Aligner checkpoint...")
            load_ckpt(self.aligner_lm, self.frontend_exp_name, 'model') # 移除 verbose
            self.aligner_lm.eval().to(device)
            self.kv_cache = None; self.hooks = None
        except Exception as e: log.error(f"Failed to load Aligner LM: {e}", exc_info=True); raise

        log.info(f"Loading G2P LM from: {self.g2p_exp_name}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.g2p_tokenizer = AutoTokenizer.from_pretrained(self.g2p_exp_name, padding_side="right")
            self.g2p_tokenizer.padding_side = "right"
            self.g2p_model = AutoModelForCausalLM.from_pretrained(self.g2p_exp_name).eval().to(device)
            try:
                 encoded_token = self.g2p_tokenizer.encode('<Reserved_TTS_0>', add_special_tokens=False)
                 if not encoded_token: raise ValueError("Encoding <Reserved_TTS_0> failed.")
                 self.speech_start_idx = encoded_token[0]
                 log.info(f"  G2P speech_start_idx: {self.speech_start_idx}")
            except Exception as e_idx: log.error(f"Could not get G2P speech_start_idx: {e_idx}", exc_info=True); raise
        except Exception as e: log.error(f"Failed to load G2P Model/Tokenizer: {e}", exc_info=True); raise

        log.info(f"Loading WavVAE from: {self.wavvae_exp_name}")
        try:
            wavvae_config_path = os.path.join(self.wavvae_exp_name, 'config.yaml')
            self.hp_wavvae = hp_wavvae = set_hparams(wavvae_config_path, global_hparams=False)
            from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
            self.wavvae = WavVAE_V3(hparams=hp_wavvae)
            vae_encoder_ckpt = os.path.join(self.wavvae_exp_name, 'model_only_last.ckpt')
            vae_decoder_ckpt = os.path.join(self.wavvae_exp_name, 'decoder.ckpt')
            if os.path.exists(vae_encoder_ckpt):
                log.info(f"  Loading full VAE checkpoint: {vae_encoder_ckpt}")
                load_ckpt(self.wavvae, vae_encoder_ckpt, 'model_gen', strict=True) # 移除 verbose
                self.has_vae_encoder = True
            elif os.path.exists(vae_decoder_ckpt):
                log.info(f"  Loading decoder-only VAE checkpoint: {vae_decoder_ckpt}")
                load_ckpt(self.wavvae, vae_decoder_ckpt, 'model_gen', strict=False) # 移除 verbose
                self.has_vae_encoder = False
            else: raise FileNotFoundError(f"No WavVAE checkpoint found in {self.wavvae_exp_name}")
            self.wavvae.eval().to(device)
            self.vae_stride = hp_wavvae.get('vae_stride', 4)
            default_hop = self.sr // 100
            self.hop_size = hp_wavvae.get('hop_size', default_hop)
            log.info(f"  WavVAE loaded. Has Encoder: {self.has_vae_encoder}, Stride: {self.vae_stride}, Hop: {self.hop_size}")
        except Exception as e: log.error(f"Failed to load WavVAE: {e}", exc_info=True); raise
        log.info("build_model finished.")


    def preprocess(self, audio_bytes, latent_file=None, topk_dur=1, **kwargs):
        log.info(f"Preprocessing audio. Latent file hint: {latent_file}")
        try:
            wav_bytes_io = convert_to_wav_bytes(audio_bytes)
            wav, sr_read = librosa.core.load(wav_bytes_io, sr=self.sr)
            log.info(f"  Loaded wav, length={len(wav)}, sr={sr_read} -> {self.sr}")
        except Exception as e: log.error(f"Error loading/converting WAV: {e}", exc_info=True); raise

        try:
            hparams_current = hparams # Assuming global hparams is intended state here
            ws = hparams_current.get('win_size', 1024)
            if len(wav) % ws != 0:
                 padding_needed = ws - (len(wav) % ws)
                 if padding_needed < ws: wav = np.pad(wav, (0, padding_needed), mode='constant')
            extra_padding = 12000
            wav = np.pad(wav, (0, extra_padding), mode='constant').astype(np.float32)
            current_loudness_prompt = self.loudness_meter.integrated_loudness(wav.astype(float))
            log.info(f"  Padding complete. Prompt loudness: {current_loudness_prompt:.2f} LUFS")
        except Exception as e: log.error(f"Error padding/loudness: {e}", exc_info=True); raise

        log.info("  Running alignment...")
        try:
            ph_ref, tone_ref, mel2ph_ref = align(self, wav)
            log.info(f"  Alignment done. Shapes: ph={ph_ref.shape}, tone={tone_ref.shape}, mel2ph={mel2ph_ref.shape}")
        except Exception as e: log.error(f"Error during alignment: {e}", exc_info=True); raise

        with torch.inference_mode():
            vae_latent = None
            target_len = mel2ph_ref.size(1) // self.vae_stride
            log.debug(f"  Target latent length: {target_len}")
            try:
                if self.has_vae_encoder:
                    log.info("  Generating VAE latent...")
                    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(self.device)
                    vae_latent = self.wavvae.encode_latent(wav_tensor)
                    # Trim/Pad
                    if vae_latent.size(1) < target_len:
                         padding_size = target_len - vae_latent.size(1)
                         vae_latent = torch.nn.functional.pad(vae_latent, (0, 0, 0, padding_size))
                    elif vae_latent.size(1) > target_len: vae_latent = vae_latent[:, :target_len]
                    log.info(f"  VAE latent generated. Shape: {vae_latent.shape}")
                else:
                    log.info("  Loading VAE latent from file...")
                    if latent_file is None or not os.path.exists(latent_file):
                         raise ValueError(f"Latent file needed but not found/provided: {latent_file}")
                    vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                    if vae_latent.ndim == 2: vae_latent = vae_latent.unsqueeze(0)
                    # Trim/Pad
                    if vae_latent.size(1) < target_len:
                         padding_size = target_len - vae_latent.size(1)
                         vae_latent = torch.nn.functional.pad(vae_latent, (0, 0, 0, padding_size))
                    elif vae_latent.size(1) > target_len: vae_latent = vae_latent[:, :target_len]
                    log.info(f"  VAE latent loaded. Shape: {vae_latent.shape}")
            except Exception as e: log.error(f"Error processing VAE latent: {e}", exc_info=True); raise

            log.info("  Creating duration prompt...")
            try:
                topk_value = kwargs.get('topk_dur', 1)
                self.dur_model.hparams["infer_top_k"] = topk_value if topk_value > 1 else None
                incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref)
                log.info(f"  Duration prompt created. Tokens shape: {ctx_dur_tokens.shape}")
            except Exception as e: log.error(f"Error duration prompting: {e}", exc_info=True); raise

        resource_context = {
            'ph_ref': ph_ref, 'tone_ref': tone_ref, 'mel2ph_ref': mel2ph_ref,
            'vae_latent': vae_latent, 'incremental_state_dur_prompt': incremental_state_dur_prompt,
            'ctx_dur_tokens': ctx_dur_tokens, 'loudness_prompt': current_loudness_prompt # Use calculated loudness
        }
        log.info("Preprocessing finished.")
        return resource_context


    def forward(self, resource_context, input_text, time_step, p_w, t_w, dur_disturb=0.1, dur_alpha=1.0, **kwargs):
        # --- 清理了 forward 内不必要的日志，保留关键步骤信息 ---
        log.info(f"Forward call started. Text: '{input_text[:30]}...', ts={time_step}, pw={p_w}, tw={t_w}, dur_disturb={dur_disturb}")
        device = self.device

        # Validate context
        required_keys = ['ph_ref', 'tone_ref', 'mel2ph_ref', 'vae_latent', 'ctx_dur_tokens', 'incremental_state_dur_prompt', 'loudness_prompt']
        if not resource_context or not all(k in resource_context for k in required_keys):
             log.error(f"Invalid resource_context. Missing: {[k for k in required_keys if k not in resource_context]}")
             raise ValueError("Invalid resource_context in forward.")

        ph_ref = resource_context['ph_ref'].to(device)
        tone_ref = resource_context['tone_ref'].to(device)
        mel2ph_ref = resource_context['mel2ph_ref'].to(device)
        vae_latent = resource_context['vae_latent'].to(device)
        ctx_dur_tokens = resource_context['ctx_dur_tokens'].to(device)
        incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']
        loudness_prompt = resource_context['loudness_prompt']
        log.debug(f"Using loudness target: {loudness_prompt:.2f} LUFS")

        with torch.inference_mode():
            wav_pred_segments = []
            try: language_type = classify_language(input_text) if input_text and input_text.strip() else 'zh'
            except Exception: language_type = 'zh'; log.warning("Lang detect failed, using zh.")

            try:
                if language_type == 'en':
                    normalized_text = self.en_normalizer.normalize(input_text)
                    text_segs = chunk_text_english(normalized_text, max_chars=130)
                else:
                    normalized_text = self.zh_normalizer.normalize(input_text)
                    text_segs = chunk_text_chinesev2(normalized_text, limit=60)
                log.info(f"Text normalized and split into {len(text_segs)} segments.")
            except Exception as e: log.error(f"Error text processing: {e}", exc_info=True); raise

            valid_segs = [seg for seg in text_segs if seg and seg.strip()]
            if not valid_segs: log.warning("No valid segments found."); return to_wav_bytes(np.zeros(int(0.1*self.sr)), self.sr)

            for seg_i, text_seg in enumerate(valid_segs):
                log.info(f"Processing segment {seg_i+1}/{len(valid_segs)}...")
                try:
                    log.debug("Running G2P...")
                    ph_pred, tone_pred = g2p(self, text_seg)
                    log.debug("Running duration prediction...")
                    is_first, is_final = seg_i == 0, seg_i == len(valid_segs)-1
                    mel2ph_pred = dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first=is_first, is_final=is_final)
                    log.debug("Preparing DiT inputs...")
                    inputs = prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent)

                    log.debug("Running DiT inference...")
                    device_type_str = self.device.type # Get 'cuda' or 'cpu'
                    with torch.amp.autocast(device_type=device_type_str, dtype=self.precision, enabled=(self.precision != torch.float32)):
                        x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()

                    log.debug("Running WavVAE decode...")
                    prompt_latent_len = vae_latent.size(1)
                    if x.size(1) < prompt_latent_len: x[:, :x.size(1)] = vae_latent[:, :x.size(1)]
                    else: x[:, :prompt_latent_len] = vae_latent
                    with torch.amp.autocast(device_type=device_type_str, dtype=self.precision, enabled=(self.precision != torch.float32)):
                        wav_pred_seg = self.wavvae.decode(x)[0,0].to(torch.float32)

                    log.debug("Post-processing segment...")
                    actual_prompt_len = min(prompt_latent_len, x.size(1))
                    trim_point = actual_prompt_len * self.vae_stride * self.hop_size
                    wav_pred_trimmed = wav_pred_seg[trim_point:].cpu().numpy()

                    if np.any(wav_pred_trimmed):
                        try:
                            loudness_pred = self.loudness_meter.integrated_loudness(wav_pred_trimmed.astype(float))
                            if np.isfinite(loudness_pred) and np.isfinite(loudness_prompt):
                                wav_pred_norm = pyln.normalize.loudness(wav_pred_trimmed, loudness_pred, loudness_prompt)
                            else: wav_pred_norm = wav_pred_trimmed
                        except Exception: wav_pred_norm = wav_pred_trimmed
                    else: wav_pred_norm = wav_pred_trimmed

                    max_abs = np.abs(wav_pred_norm).max() if len(wav_pred_norm)>0 else 0
                    if max_abs >= 1.0: wav_pred_clipped = wav_pred_norm * (0.98 / (max_abs + 1e-6))
                    else: wav_pred_clipped = wav_pred_norm

                    wav_pred_segments.append(wav_pred_clipped)
                    log.info(f"Segment {seg_i+1} processed successfully.")

                except Exception as seg_e:
                    log.error(f"Error processing segment {seg_i+1}: {seg_e}", exc_info=True)
                    log.warning(f"Skipping segment {seg_i+1}.")
                    continue # Skip segment on error

            if not wav_pred_segments: log.error("No segments generated."); return to_wav_bytes(np.zeros(int(0.1*self.sr)), self.sr)

            log.info(f"Combining {len(wav_pred_segments)} segments...")
            final_wav_np = combine_audio_segments(wav_pred_segments, sr=self.sr).astype(np.float32)
            log.info(f"Final waveform length: {len(final_wav_np)}")
            final_wav_bytes = to_wav_bytes(final_wav_np, self.sr)
            log.info(f"Forward call finished. Returning {len(final_wav_bytes)} bytes.")
            return final_wav_bytes


# --- Main Execution Block (for basic CLI testing, less verbose) ---
if __name__ == '__main__':
    # Setup basic logging for direct CLI run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info("--- CLI Main Execution Start (Basic Test Mode) ---")

    parser = argparse.ArgumentParser(description="MegaTTS3-DiT CLI (Basic Test)")
    parser.add_argument('--input_wav', type=str, required=True)
    parser.add_argument('--input_npy', type=str, default=None)
    parser.add_argument('--input_text', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./output_cli_test")
    parser.add_argument('--time_step', type=int, default=32)
    parser.add_argument('--p_w', type=float, default=1.4) # Match known good defaults
    parser.add_argument('--t_w', type=float, default=3.0) # Match known good defaults
    parser.add_argument('--dur_disturb', type=float, default=0.0) # Deterministic default
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints') # Adjusted default assuming run from tts dir

    args = parser.parse_args()
    log.info(f"CLI Args: {args}")

    try:
        log.info("Initializing TTS Engine...")
        # Infer device and precision or get from args if added
        infer_ins = MegaTTS3DiTInfer(ckpt_root=args.ckpt_root) # device/precision auto-detected
        log.info("TTS Engine Initialized.")

        log.info("Reading prompt WAV...")
        if not os.path.exists(args.input_wav): raise FileNotFoundError(f"Input WAV not found: {args.input_wav}")
        with open(args.input_wav, 'rb') as file: prompt_content = file.read()

        log.info("Preprocessing prompt...")
        # Validate NPY if needed
        latent_file_arg = args.input_npy
        if not infer_ins.has_vae_encoder:
            if not latent_file_arg: raise ValueError("--input_npy required but not provided.")
            if not os.path.exists(latent_file_arg): raise FileNotFoundError(f"NPY file not found: {latent_file_arg}")
        elif latent_file_arg:
            log.warning(f"Ignoring --input_npy as model has encoder: {latent_file_arg}")
            latent_file_arg = None

        resource_context = infer_ins.preprocess(prompt_content, latent_file=latent_file_arg)
        log.info("Preprocessing complete.")

        # --- Optional: Print simplified context for basic CLI check ---
        log.info("--- RESOURCE_CONTEXT (CLI Basic Check) ---")
        for key, value in resource_context.items():
            if torch.is_tensor(value): log.info(f"  '{key}': shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict): log.info(f"  '{key}': type=dict")
            else: log.info(f"  '{key}': type={type(value)}")
        log.info("-----------------------------------------")
        # --- Check End ---

        log.info("Running forward pass...")
        wav_bytes = infer_ins.forward(
            resource_context, args.input_text,
            time_step=args.time_step, p_w=args.p_w, t_w=args.t_w,
            dur_disturb=args.dur_disturb
        )
        log.info("Forward pass complete.")

        os.makedirs(args.output_dir, exist_ok=True)
        sanitized_text = "".join(c if c.isalnum() or c in [' ', '-'] else '_' for c in args.input_text[:20]).rstrip()
        output_filename = f"[CLI_P{args.p_w}_T{args.t_w}]_{sanitized_text.replace(' ', '_')}.wav"
        output_filepath = os.path.join(args.output_dir, output_filename)
        log.info(f"Saving result to: {output_filepath}")
        save_wav(wav_bytes, output_filepath)
        log.info("Result saved.")

    except Exception as e:
        log.error("Error during CLI execution.", exc_info=True)

    log.info("--- CLI Main Execution End ---")