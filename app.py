# optimized_app_v1.py (Complete code with run_text_tts_ui moved)

import gradio as gr
import os
import torch
import srt
import tempfile
import traceback
import time
from datetime import timedelta
import logging
import random
import numpy as np
import sys
import shutil
from tqdm.auto import tqdm
import concurrent.futures
import gc
import json
import glob
from pydub import AudioSegment

# --- 设置随机种子 ---
seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --- 配置日志 ---
log_file = os.path.join(os.path.dirname(__file__), "app.log")
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)
logging.info("--- Gradio App Start ---")

# --- 导入必要的模块 ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path: sys.path.insert(0, project_root); logging.debug(f"Added project root to sys.path: {project_root}")
    from tts.infer_srt_cli import MegaTTS3DiTInfer
    from audio_stitcher import stitch_audio_clips
    from tts.utils.audio_utils.io import save_wav
except ImportError as e: logging.error(f"Import Error: {e}. Ensure structure correct.", exc_info=True); print(f"Import Error: {e}"); exit(1)
except Exception as e_gen: logging.error(f"Unknown import error: {e_gen}", exc_info=True); raise

# --- 全局变量和模型加载 ---
infer_engine = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION = torch.float16
DEFAULT_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "app_params.json")

# --- CUDA 内存管理函数 ---
def clear_cuda_cache():
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect(); logging.info("CUDA cache cleared")

# --- 模型加载函数 ---
def load_tts_model_ui(ckpt_root, dit_exp, progress=gr.Progress()):
    global infer_engine
    if infer_engine is not None:
        engine_status = "TTS Engine already loaded";
        if not infer_engine.has_vae_encoder: engine_status += " (Requires .npy)"
        logging.info(engine_status); return engine_status
    logging.info(f"Loading TTS model..."); progress(0, desc="Initializing TTS...")
    try:
        clear_cuda_cache(); abs_ckpt_root = os.path.abspath(ckpt_root)
        infer_engine = MegaTTS3DiTInfer(device=DEVICE, precision=PRECISION, ckpt_root=abs_ckpt_root, dit_exp_name=dit_exp)
        progress(1, desc="TTS Engine Loaded."); logging.info("TTS Engine Init complete.")
        if not infer_engine.has_vae_encoder: return "TTS Engine Loaded (Requires .npy)"
        else: return "TTS Engine Loaded Successfully."
    except Exception as e:
        logging.error("Error loading TTS Model", exc_info=True); infer_engine = None; clear_cuda_cache()
        progress(1.0, desc="Model Load Failed!"); return f"Error loading TTS Model: {e}"

# --- 获取音频文件列表函数 ---
def get_audio_files(directory):
    wav_files = []
    if directory and os.path.isdir(directory):
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(".wav"): wav_files.append(os.path.splitext(filename)[0])
            logging.info(f"Found {len(wav_files)} WAV files in '{directory}'")
        except Exception as e: logging.error(f"Error scanning dir '{directory}': {e}", exc_info=True); return []
    else: return []
    return sorted(wav_files)

# --- 并行处理音频片段生成函数 ---
def generate_audio_clip(args_tuple):
    sub, resource_context, clips_dir, time_step, p_w, t_w, dur_disturb, infer_engine_local = args_tuple
    try:
        subtitle_text = sub.content.strip();
        if not subtitle_text: return sub.index, None, False
        wav_bytes = infer_engine_local.forward(resource_context=resource_context, input_text=subtitle_text, time_step=time_step, p_w=p_w, t_w=t_w, dur_disturb=dur_disturb)
        clip_filename = os.path.join(clips_dir, f"clip_{sub.index}.wav"); save_wav(wav_bytes, clip_filename)
        return sub.index, clip_filename, True
    except Exception as e: logging.error(f"ERROR generating clip for sub {sub.index}", exc_info=True); return sub.index, None, False

def batch_generate_audio_clips(subtitles, resource_context, clips_dir, time_step, p_w, t_w, dur_disturb, infer_engine_ref, max_workers=2, batch_size=4, progress_fn=None):
    clip_path_map = {}; generation_errors = 0; tasks = []
    for sub in subtitles:
        if sub.content.strip(): tasks.append((sub, resource_context, clips_dir, time_step, p_w, t_w, dur_disturb, infer_engine_ref))
    total_tasks = len(tasks); processed_tasks = 0
    if total_tasks == 0: return {}, 0
    for i in range(0, total_tasks, batch_size):
        batch_tasks = tasks[i:i+batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_audio_clip, task) for task in batch_tasks]
            for future in concurrent.futures.as_completed(futures):
                processed_tasks += 1
                if progress_fn: progress_fn(processed_tasks / total_tasks, desc=f"Generating clip {processed_tasks}/{total_tasks}")
                try:
                    sub_index, clip_filename, success = future.result()
                    if success and clip_filename: clip_path_map[sub_index] = clip_filename
                    else: generation_errors += 1
                except Exception as e: logging.error(f"Error processing future result: {e}", exc_info=True); generation_errors += 1
        clear_cuda_cache()
    return clip_path_map, generation_errors

# --- 新增：文本转语音处理函数 ---
def run_text_tts_ui(
    input_text_tts, # Input from the new Textbox
    selected_prompt_name, # Input from the main tab's dropdown
    prompt_audio_dir, # Input from the main tab's directory textbox
    ckpt_root_dir, # Model params from main tab
    dit_exp_name,
    time_step,
    p_w,
    t_w,
    dur_disturb,
    progress=gr.Progress()
):
    global infer_engine
    request_id = f"TextTTS_{int(time.time())}"; logging.info(f"--- [{request_id}] Text TTS Started ---")
    status_updates = [f"--- [{request_id}] Text-to-Speech Process Started ---"]; yield "\n".join(status_updates), None

    final_output_path = None

    try:
        # --- 1. 加载模型 (与主函数逻辑类似) ---
        load_status = load_tts_model_ui(ckpt_root_dir, dit_exp_name, progress)
        status_updates.append(load_status); logging.info(f"[{request_id}] Model load: {load_status}")
        yield "\n".join(status_updates), None
        if infer_engine is None: err_msg = f"FATAL: Model load failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Failed!"); yield "\n".join(status_updates), None; return

        # --- 输入验证 ---
        if not input_text_tts or not input_text_tts.strip():
            err_msg = "ERROR: Input text cannot be empty."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return
        if not selected_prompt_name or not prompt_audio_dir:
            err_msg = "ERROR: Prompt selection and Directory required (from main tab)."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return
        if not os.path.isdir(prompt_audio_dir):
            err_msg = f"ERROR: Prompt Dir invalid: {prompt_audio_dir}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return

        # --- 构造样本音和 NPY 路径 (与主函数逻辑类似) ---
        progress(0.1, "Preparing prompt...")
        abs_prompt_audio_dir = os.path.abspath(prompt_audio_dir); abs_input_wav = os.path.join(abs_prompt_audio_dir, selected_prompt_name + ".wav"); latent_file_arg = None
        if not os.path.exists(abs_input_wav):
            err_msg = f"ERROR: Selected WAV not found: {abs_input_wav}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return

        if not infer_engine.has_vae_encoder:
            status_updates.append("Model requires .npy (searching)..."); abs_input_npy = os.path.join(abs_prompt_audio_dir, selected_prompt_name + ".npy")
            logging.info(f"[{request_id}] Expecting NPY at: {abs_input_npy}")
            if not os.path.exists(abs_input_npy):
                err_msg = f"ERROR: Required .npy not found: {abs_input_npy}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="NPY Error!"); yield "\n".join(status_updates), None; return
            else:
                latent_file_arg = abs_input_npy; status_updates.append("Found required NPY."); logging.info(f"[{request_id}] Using NPY path: {latent_file_arg}")
        else:
            status_updates.append("Model has encoder, .npy not needed."); logging.info(f"[{request_id}] Model has encoder.")
        yield "\n".join(status_updates), None

        # --- 2. 预处理样本音 (与主函数逻辑类似) ---
        resource_context = None
        try:
            status_updates.append("Preprocessing prompt audio..."); yield "\n".join(status_updates), None; progress(0.2, "Preprocessing prompt...")
            with open(abs_input_wav, 'rb') as f_prompt: prompt_content = f_prompt.read()
            logging.info(f"[{request_id}] Calling preprocess with latent_file='{latent_file_arg}'")
            resource_context = infer_engine.preprocess(prompt_content, latent_file=latent_file_arg)
            status_updates.append("Prompt preprocessing complete."); yield "\n".join(status_updates), None; logging.info(f"[{request_id}] Preprocess successful.")
        except Exception as e:
            status_updates.append(f"ERROR during prompt preprocessing: {e}"); logging.error(f"[{request_id}] ERROR prompt preprocess", exc_info=True); progress(1.0, desc="Prep Failed!"); yield "\n".join(status_updates), None; return
        if not resource_context:
            err_msg = "ERROR: Prompt preprocessing failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Prep Failed!"); yield "\n".join(status_updates), None; return

        # --- 3. 生成语音 --- 
        status_updates.append("Generating speech from text..."); yield "\n".join(status_updates), None; progress(0.5, "Generating speech...")
        start_time = time.time()
        wav_bytes = None
        try:
            wav_bytes = infer_engine.forward(
                resource_context=resource_context,
                input_text=input_text_tts.strip(),
                time_step=int(time_step),
                p_w=float(p_w),
                t_w=float(t_w),
                dur_disturb=float(dur_disturb)
            )
            gen_duration = time.time() - start_time
            log_msg = f"Speech generation done ({gen_duration:.2f}s). Output size: {len(wav_bytes) if wav_bytes else 0} bytes."
            status_updates.append(log_msg); logging.info(f"[{request_id}] {log_msg}"); yield "\n".join(status_updates), None
        except Exception as e:
            status_updates.append(f"ERROR during speech generation: {e}"); logging.error(f"[{request_id}] ERROR forward pass", exc_info=True); progress(1.0, desc="Generation Failed!"); yield "\n".join(status_updates), None; return

        if not wav_bytes:
            err_msg = "ERROR: Generation resulted in empty audio."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Generation Failed!"); yield "\n".join(status_updates), None; return

        # --- 4. 导出最终文件 --- 
        progress(0.9, "Exporting audio...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"text_tts_{request_id}_") as temp_output_file:
            final_output_path = temp_output_file.name
            temp_output_file.write(wav_bytes) # Write bytes directly
            logging.info(f"Audio bytes written to temp file: {final_output_path}")

        status_updates.append(f"Exporting final audio to temp file: {final_output_path}"); logging.info(f"[{request_id}] Exporting to: {final_output_path}"); yield "\n".join(status_updates), None

        progress(1.0, "Done!")
        status_updates.append(f"--- [{request_id}] Text-to-Speech Completed Successfully! --- "); logging.info(f"--- [{request_id}] Text TTS Completed Successfully! ---")
        yield "\n".join(status_updates), final_output_path # Return log and audio path

    except Exception as e: # General error catch
        logging.error(f"[{request_id}] Error during text TTS processing loop", exc_info=True)
        status_updates.append(f"ERROR during processing: {e}")
        progress(1.0, desc="Processing Failed!")
        yield "\n".join(status_updates), None
        return

    finally:
        clear_cuda_cache() # Clean GPU at the end

# --- 核心处理函数 ---
def run_srt_dubbing_ui(
    selected_prompt_name, prompt_audio_dir, srt_file,
    clips_save_dir_ui, ckpt_root_dir, dit_exp_name, time_step, p_w, t_w, dur_disturb,
    initial_speed, enable_auto_adjust, max_speed, max_workers, batch_size,
    progress=gr.Progress(track_tqdm=True)
):
    global infer_engine
    request_id = f"Gradio_{int(time.time())}"; logging.info(f"--- [{request_id}] UI Dubbing Started ---")
    status_updates = [f"--- [{request_id}] Dubbing Process Started ---"]; yield "\n".join(status_updates), None

    # --- 保存参数 ---
    current_params = {
        "selected_prompt_name": selected_prompt_name,
        "prompt_audio_dir": prompt_audio_dir,
        "clips_save_dir_ui": clips_save_dir_ui,
        "ckpt_root_dir": ckpt_root_dir,
        "dit_exp_name": dit_exp_name,
        "time_step": time_step,
        "p_w": p_w,
        "t_w": t_w,
        "dur_disturb": dur_disturb,
        "initial_speed": initial_speed,
        "enable_auto_adjust": enable_auto_adjust,
        "max_speed": max_speed,
        "max_workers": max_workers,
        "batch_size": batch_size
    }
    save_params_status = save_params(current_params, DEFAULT_PARAMS_FILE); status_updates.append(save_params_status); yield "\n".join(status_updates), None

    # --- 初始化 ---
    clip_directory_context = None; output_clips_dir_abs = None; final_output_path = None

    try:
        # --- 1. 加载模型 ---
        load_status = load_tts_model_ui(ckpt_root_dir, dit_exp_name, progress)
        status_updates.append(load_status); logging.info(f"[{request_id}] Model load: {load_status}")
        yield "\n".join(status_updates), None
        if infer_engine is None: err_msg = f"FATAL: Model load failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Failed!"); yield "\n".join(status_updates), None; return

        # --- 输入验证 ---
        if not selected_prompt_name or not prompt_audio_dir or not srt_file: err_msg = "ERROR: Prompt selection, Directory, and SRT required."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return
        if not os.path.isdir(prompt_audio_dir): err_msg = f"ERROR: Prompt Dir invalid: {prompt_audio_dir}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return
        abs_input_srt = srt_file.name
        if not os.path.exists(abs_input_srt): err_msg = f"ERROR: SRT not found: {abs_input_srt}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return

        # --- 构造样本音和 NPY 路径 ---
        abs_prompt_audio_dir = os.path.abspath(prompt_audio_dir); abs_input_wav = os.path.join(abs_prompt_audio_dir, selected_prompt_name + ".wav"); latent_file_arg = None
        if not os.path.exists(abs_input_wav): err_msg = f"ERROR: Selected WAV not found: {abs_input_wav}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Input Error!"); yield "\n".join(status_updates), None; return

        if not infer_engine.has_vae_encoder:
            status_updates.append("Model requires .npy (searching)..."); abs_input_npy = os.path.join(abs_prompt_audio_dir, selected_prompt_name + ".npy")
            logging.info(f"[{request_id}] Expecting NPY at: {abs_input_npy}")
            if not os.path.exists(abs_input_npy): err_msg = f"ERROR: Required .npy not found: {abs_input_npy}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="NPY Error!"); yield "\n".join(status_updates), None; return
            else:
                latent_file_arg = abs_input_npy; status_updates.append("Found required NPY."); logging.info(f"[{request_id}] Using NPY path: {latent_file_arg}")
        else:
            status_updates.append("Model has encoder, .npy not needed."); logging.info(f"[{request_id}] Model has encoder.")
        yield "\n".join(status_updates), None

        # --- 处理片段保存目录 ---
        use_temp_clips_dir = not clips_save_dir_ui or not clips_save_dir_ui.strip()
        if use_temp_clips_dir:
            status_updates.append("Using system temp dir for clips."); clip_directory_context = tempfile.TemporaryDirectory(prefix=f"gradio_clips_{request_id}_")
            try: output_clips_dir_abs = clip_directory_context.__enter__()
            except Exception as temp_e: err_msg = f"ERROR creating temp dir: {temp_e}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}", exc_info=True); progress(1.0, desc="Error!"); yield "\n".join(status_updates), None; return
        else:
            output_clips_dir_abs = os.path.abspath(clips_save_dir_ui)
            try: os.makedirs(output_clips_dir_abs, exist_ok=True); status_updates.append(f"Saving clips to: {output_clips_dir_abs} (Manual cleanup).")
            except OSError as e: err_msg = f"ERROR creating clips dir '{output_clips_dir_abs}': {e}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}", exc_info=True); progress(1.0, desc="Error!"); yield "\n".join(status_updates), None; return
        yield "\n".join(status_updates), None; logging.info(f"[{request_id}] Clips target dir: {output_clips_dir_abs}")

        # --- 2. 预处理 ---
        resource_context = None
        try:
            status_updates.append("Preprocessing prompt..."); yield "\n".join(status_updates), None; progress(0.1, "Preprocessing...")
            with open(abs_input_wav, 'rb') as f_prompt: prompt_content = f_prompt.read()
            logging.info(f"[{request_id}] Calling preprocess with latent_file='{latent_file_arg}'")
            resource_context = infer_engine.preprocess(prompt_content, latent_file=latent_file_arg)
            status_updates.append("Preprocessing complete."); yield "\n".join(status_updates), None; logging.info(f"[{request_id}] Preprocess successful.")
        except Exception as e: status_updates.append(f"ERROR preprocess: {e}"); logging.error(f"[{request_id}] ERROR preprocess", exc_info=True); progress(1.0, desc="Prep Failed!"); yield "\n".join(status_updates), None; return # finally 会清理
        if not resource_context: err_msg = "ERROR: Preprocessing failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Prep Failed!"); yield "\n".join(status_updates), None; return # finally 会清理

        # --- 3. 解析 SRT ---
        try:
            status_updates.append("Parsing SRT..."); yield "\n".join(status_updates), None; progress(0.2, "Parsing SRT...")
            with open(abs_input_srt, 'r', encoding='utf-8-sig') as f_srt: srt_content = f_srt.read()
            subtitles = list(srt.parse(srt_content))
            if not subtitles: # 修正后的检查
                err_msg = f"ERROR: SRT empty/invalid: {abs_input_srt}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}")
                progress(1.0, desc="SRT Error!"); yield "\n".join(status_updates), None; return # finally 会清理
            srt_total_duration_ms = int(subtitles[-1].end.total_seconds() * 1000); status_updates.append(f"SRT parsed: {len(subtitles)} subtitles."); logging.info(f"[{request_id}] SRT parsed: {len(subtitles)} subs, Duration(ms): {srt_total_duration_ms}"); yield "\n".join(status_updates), None
        except Exception as e: status_updates.append(f"ERROR parsing SRT: {e}"); logging.error(f"[{request_id}] ERROR parsing SRT", exc_info=True); progress(1.0, desc="SRT Error!"); yield "\n".join(status_updates), None; return # finally 会清理

        # --- 4. 生成片段 (并行) ---
        clip_path_map = {}; generation_errors = 0; start_time = time.time()
        logging.info(f"[{request_id}] Starting parallel clip generation..."); status_updates.append(f"Generating clips (workers={max_workers}, batch={batch_size})..."); yield "\n".join(status_updates), None
        def progress_callback(current_prog, desc=""): progress(current_prog * 0.6 + 0.3, desc=desc)

        clip_path_map, generation_errors = batch_generate_audio_clips(subtitles=subtitles, resource_context=resource_context, clips_dir=output_clips_dir_abs, time_step=int(time_step), p_w=float(p_w), t_w=float(t_w), dur_disturb=float(dur_disturb), infer_engine_ref=infer_engine, max_workers=int(max_workers), batch_size=int(batch_size), progress_fn=progress_callback)
        gen_duration = time.time() - start_time; log_msg = f"Clip generation done ({gen_duration:.2f}s). Success: {len(clip_path_map)}, Failed: {generation_errors}."
        status_updates.append(log_msg); logging.info(f"[{request_id}] {log_msg}"); yield "\n".join(status_updates), None

        # --- 修正后的空片段检查 ---
        if not clip_path_map:
            err_msg = "ERROR: No clips generated successfully."
            status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}")
            progress(1.0, desc="Generation Failed!")
            yield "\n".join(status_updates), None
            return # finally 会清理
        # --- 修正结束 ---

        # --- 5. 拼接 ---
        status_updates.append("Starting stitching..."); yield "\n".join(status_updates), None
        logging.info(f"[{request_id}] Starting stitching from {output_clips_dir_abs}"); progress(0.95, "Stitching...")
        stitch_start_time = time.time(); stitcher_status_list = []; final_output_path = None

        final_audio, final_speed, final_gap, stitch_errors = stitch_audio_clips(subtitles=subtitles, clip_path_map=clip_path_map, srt_total_duration_ms=srt_total_duration_ms, initial_speed_factor=float(initial_speed), enable_auto_adjust=bool(enable_auto_adjust), max_speed_factor=float(max_speed), status_updates_list=stitcher_status_list)
        for msg in stitcher_status_list: logging.info(f"[{request_id}] Stitcher: {msg}")
        status_updates.extend(stitcher_status_list)
        if final_audio is None: err_msg = "ERROR: Stitching failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); progress(1.0, desc="Stitch Failed!"); yield "\n".join(status_updates), None; return # finally 会清理

        stitch_duration = time.time() - stitch_start_time; log_msg = f"Stitching done ({stitch_duration:.2f}s). Errors: {stitch_errors}. Speed={final_speed:.3f}, Gap={final_gap}ms. Duration: {len(final_audio)/1000.0:.2f}s"
        status_updates.append(f"Stitching done ({stitch_duration:.2f}s).");
        if stitch_errors > 0: status_updates.append(f"WARNING: {stitch_errors} stitch errors.")
        status_updates.append(f"Final Params: Speed={final_speed:.3f}, Gap={final_gap}ms"); status_updates.append(f"Final Duration: {len(final_audio)/1000.0:.2f}s"); logging.info(f"[{request_id}] {log_msg}"); yield "\n".join(status_updates), None

        # --- 6. 导出最终文件 ---
        progress(0.98, "Exporting...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"final_{request_id}_") as temp_output_file: final_output_path = temp_output_file.name
        status_updates.append("Exporting final audio..."); logging.info(f"[{request_id}] Exporting to: {final_output_path}"); yield "\n".join(status_updates), None
        final_audio.export(final_output_path, format="wav")

        progress(1.0, "Done!")
        status_updates.append(f"--- [{request_id}] Dubbing Completed Successfully! ---"); logging.info(f"--- [{request_id}] Dubbing Completed Successfully! ---")
        yield "\n".join(status_updates), final_output_path # 返回最终结果

    except Exception as e: # 捕获所有处理阶段的错误
        logging.error(f"[{request_id}] Error during main processing loop", exc_info=True)
        status_updates.append(f"ERROR during processing: {e}")
        progress(1.0, desc="Processing Failed!")
        yield "\n".join(status_updates), None
        # finally 会执行清理
        return # 明确返回

    finally: # <<<--- 确保清理逻辑在此处 ---
        if clip_directory_context:
            try: clip_directory_context.__exit__(None, None, None); logging.info(f"[{request_id}] Temp clip dir cleaned.")
            except Exception as e_clean: logging.error(f"Error closing temp dir: {e_clean}")
        elif output_clips_dir_abs:
             logging.warning(f"[{request_id}] Remember to manually clean clips in: {output_clips_dir_abs}")
        clear_cuda_cache() # 最终清理 GPU

# --- 参数保存/加载函数 ---
def save_params(params_dict, file_path):
    try: # ...(保存逻辑不变, 确保包含 prompt_audio_dir)...
        serializable_params = {}
        for k, v in params_dict.items():
             if isinstance(v, (str, int, float, bool, list, dict)) or v is None: serializable_params[k] = v
             elif isinstance(v, np.bool_): serializable_params[k] = bool(v)
             else: serializable_params[k] = str(v); logging.warning(f"Param '{k}' type {type(v)} saved as string.")
        with open(file_path, 'w', encoding='utf-8') as f: json.dump(serializable_params, f, indent=4, ensure_ascii=False)
        logging.info(f"Parameters saved to {file_path}"); return f"参数已自动保存到 {os.path.basename(file_path)}"
    except Exception as e: logging.error(f"保存参数失败: {e}", exc_info=True); return f"自动保存参数失败: {e}"

def load_default_params():
    default_values = { # 添加 prompt_audio_dir 和 selected_prompt_name
        "prompt_audio_dir": "",
        "selected_prompt_name": None, # <-- 添加默认值
        "clips_save_dir_ui": "", "ckpt_root_dir": "./checkpoints", "dit_exp_name": "diffusion_transformer",
        "time_step": 32, "p_w": 1.4, "t_w": 3.0, "dur_disturb": 0.0, "initial_speed": 1.0,
        "enable_auto_adjust": True, "max_speed": 1.5, "max_workers": 2, "batch_size": 4
    }
    if os.path.exists(DEFAULT_PARAMS_FILE):
        logging.info(f"Loading default parameters from {DEFAULT_PARAMS_FILE}")
        try:
            with open(DEFAULT_PARAMS_FILE, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
            for key in default_values:
                if key in loaded_params:
                    default_values[key] = loaded_params[key]
            logging.info("Successfully loaded parameters from file.")
            return default_values
        except Exception as e:
            logging.warning(f"Failed load {DEFAULT_PARAMS_FILE}: {e}. Using defaults.", exc_info=True)
            return default_values
    else:
        logging.info(f"Default params file not found. Using defaults.")
        return default_values

# --- 二次拼接函数 ---
def restitch_audio_clips(subtitles, clips_dir, srt_total_duration_ms, initial_speed, enable_auto_adjust, max_speed):
    request_id = f"Restitch_{int(time.time())}"; status_updates = [f"--- [{request_id}] Starting Restitch ---"]
    logging.info(f"--- [{request_id}] Starting Restitch ---"); final_output_path = None
    try:
        if not os.path.isdir(clips_dir): err_msg = f"ERROR: Clips dir not found: {clips_dir}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); return "\n".join(status_updates), None
        clip_path_map = {}; clip_files = glob.glob(os.path.join(clips_dir, "clip_*.wav"))
        if not clip_files: err_msg = f"ERROR: No clips found in {clips_dir}"; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); return "\n".join(status_updates), None
        for clip_file in clip_files:
            try:
                file_name = os.path.basename(clip_file)
                if file_name.startswith("clip_") and file_name.endswith(".wav"):
                    index_str = file_name[5:-4]
                    if index_str.isdigit():
                        clip_path_map[int(index_str)] = clip_file
            except Exception as e: logging.error(f"[{request_id}] Error processing clip file {clip_file}: {e}")
        if not clip_path_map: err_msg = "ERROR: Could not map clips."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); return "\n".join(status_updates), None
        status_updates.append(f"Found {len(clip_path_map)} clips."); logging.info(f"[{request_id}] Found {len(clip_path_map)} clips.")
        status_updates.append("Starting audio stitching..."); logging.info(f"[{request_id}] Starting audio stitching.")
        stitch_start_time = time.time(); stitcher_status_list = []
        final_audio, final_speed, final_gap, stitch_errors = stitch_audio_clips(subtitles=subtitles, clip_path_map=clip_path_map, srt_total_duration_ms=srt_total_duration_ms, initial_speed_factor=float(initial_speed), enable_auto_adjust=bool(enable_auto_adjust), max_speed_factor=float(max_speed), status_updates_list=stitcher_status_list)
        for msg in stitcher_status_list: logging.info(f"[{request_id}] Stitcher: {msg}")
        status_updates.extend(stitcher_status_list)
        if final_audio is None: err_msg = "ERROR: Restitching failed."; status_updates.append(err_msg); logging.error(f"[{request_id}] {err_msg}"); return "\n".join(status_updates), None
        stitch_duration = time.time() - stitch_start_time; log_msg = f"Restitching done ({stitch_duration:.2f}s). Errors: {stitch_errors}. Speed={final_speed:.3f}, Gap={final_gap}ms. Duration: {len(final_audio)/1000.0:.2f}s"
        status_updates.append(f"Restitching done ({stitch_duration:.2f}s).");
        if stitch_errors > 0: status_updates.append(f"WARNING: {stitch_errors} stitch errors.")
        status_updates.append(f"Final Params: Speed={final_speed:.3f}, Gap={final_gap}ms"); status_updates.append(f"Final Duration: {len(final_audio)/1000.0:.2f}s"); logging.info(f"[{request_id}] {log_msg}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"restitch_{request_id}_") as temp_output_file: final_output_path = temp_output_file.name
        status_updates.append(f"Exporting final audio..."); logging.info(f"[{request_id}] Exporting final audio to: {final_output_path}")
        final_audio.export(final_output_path, format="wav")
        status_updates.append(f"--- [{request_id}] Restitching Completed Successfully! ---"); logging.info(f"--- [{request_id}] Restitching Completed Successfully! ---")
        return "\n".join(status_updates), final_output_path
    except Exception as e: status_updates.append(f"ERROR during restitching: {e}"); logging.error(f"[{request_id}] ERROR restitching", exc_info=True); return "\n".join(status_updates), None

# --- 清理临时文件函数 ---
def clean_temp_files():
    try:
        temp_dir = tempfile.gettempdir(); patterns = [ os.path.join(temp_dir, p) for p in ["gradio_clips_*", "final_Gradio_*", "restitch_*", "*.tmp"] ]
        total_removed_files, total_removed_dirs = 0, 0
        for pattern in patterns:
            items = glob.glob(pattern)
            for item in items:
                try:
                    if os.path.isdir(item): shutil.rmtree(item); total_removed_dirs += 1
                    elif os.path.isfile(item): os.remove(item); total_removed_files += 1
                except Exception as e: logging.error(f"Error cleaning {item}: {e}")
        msg = f"Cleaned {total_removed_files} temp files and {total_removed_dirs} temp dirs."
        logging.info(msg); return msg
    except Exception as e: logging.error(f"Error cleaning temp files: {e}", exc_info=True); return f"Error cleaning temp files: {e}"

# --- 构建 Gradio 界面 ---
default_params = load_default_params()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SRT Dubbing with MegaTTS3-DiT (Select Audio from Dir)")
    gr.Markdown("Specify directory, select prompt, upload SRT, configure and run.")

    with gr.Tabs() as tabs:
        with gr.TabItem("主要处理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Inputs")
                    input_prompt_dir = gr.Textbox(label="样本音频/NPY 目录", placeholder="e.g., D:/Programes/MegaTTS3/Myvoice", value=default_params.get("prompt_audio_dir", ""))
                    with gr.Row():
                        prompt_audio_dropdown = gr.Dropdown(label="选择样本音频", choices=[], interactive=True, scale=3)
                        refresh_prompts_button = gr.Button("刷新", scale=1)
                    prompt_audio_preview = gr.Audio(label="样本音预览", type="filepath", interactive=False)
                    input_srt_file = gr.File(label="字幕文件 (.srt)", file_types=[".srt"], type="filepath")

                    # --- 使用 Accordion 折叠高级设置 ---
                    with gr.Accordion("高级设置 (点击展开)", open=False):
                        input_clips_dir = gr.Textbox(label="保存中间片段到 (可选)", placeholder="留空使用临时目录", value=default_params.get("clips_save_dir_ui", ""))
                        gr.Markdown("### Model Configuration")
                        input_ckpt_root = gr.Textbox(label="检查点根目录", value=default_params.get("ckpt_root_dir", "./checkpoints"))
                        input_dit_exp_name = gr.Textbox(label="DiT实验名称", value=default_params.get("dit_exp_name", "diffusion_transformer"))
                        gr.Markdown("### TTS Generation Settings")
                        input_time_step = gr.Slider(label="扩散步数", minimum=4, maximum=100, value=default_params.get("time_step", 32), step=1)
                        input_p_w = gr.Slider(label="清晰度(p_w)", minimum=0.0, maximum=5.0, value=default_params.get("p_w", 1.4), step=0.1)
                        input_t_w = gr.Slider(label="相似度(t_w)", minimum=0.0, maximum=5.0, value=default_params.get("t_w", 3.0), step=0.1)
                        input_dur_disturb = gr.Slider(label="时长扰动(0=无)", minimum=0.0, maximum=0.5, value=default_params.get("dur_disturb", 0.0), step=0.01)
                        gr.Markdown("### Audio Stitching Settings")
                        input_initial_speed = gr.Slider(label="初始速度", minimum=0.5, maximum=2.0, value=default_params.get("initial_speed", 1.0), step=0.05)
                        input_enable_auto_adjust = gr.Checkbox(label="自动调整", value=default_params.get("enable_auto_adjust", True))
                        input_max_speed = gr.Slider(label="最大速度", minimum=1.0, maximum=3.0, value=default_params.get("max_speed", 1.5), step=0.1)
                        gr.Markdown("### Parallel Processing")
                        input_max_workers = gr.Slider(label="最大工作线程", minimum=1, maximum=os.cpu_count() or 1, value=default_params.get("max_workers", 2), step=1)
                        input_batch_size = gr.Slider(label="批处理大小", minimum=1, maximum=16, value=default_params.get("batch_size", 4), step=1)
                    # --- Accordion 结束 ---

                    start_button = gr.Button("开始配音处理", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Output")
                    dub_progress = gr.Progress(track_tqdm=True)
                    output_log = gr.Textbox(label="Status Log", lines=20, interactive=False, autoscroll=True)
                    output_audio = gr.Audio(label="Dubbed Audio Output", type="filepath")

        # --- 新增：文本配音标签页 ---
        with gr.TabItem("文本配音"):
            gr.Markdown("### 直接从文本生成语音")
            gr.Markdown("在此处输入文本，使用'主要处理'标签页中选定的样本音频和设置进行配音。")
            with gr.Row():
                with gr.Column(scale=2):
                    text_input_tts = gr.Textbox(label="输入文本", lines=10, placeholder="在此输入要合成语音的文本...")
                    generate_text_tts_button = gr.Button("生成配音", variant="primary")
                with gr.Column(scale=1):
                    status_text_tts = gr.Textbox(label="状态日志", lines=10, interactive=False, autoscroll=True)
                    output_audio_text_tts = gr.Audio(label="配音输出", type="filepath")
            text_tts_progress = gr.Progress(track_tqdm=True) # Progress bar for this tab

        # ...(二次处理和工具 Tab 保持不变)... 
        with gr.TabItem("二次处理"):
             with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 二次拼接设置")
                    use_main_inputs_button = gr.Button("使用主页面设置填充", scale=1)
                    restitch_clips_dir = gr.Textbox(label="音频片段目录")
                    restitch_srt_file = gr.File(label="字幕文件 (.srt)", file_types=[".srt"], type="filepath")
                    restitch_initial_speed = gr.Slider(label="初始速度因子", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                    restitch_enable_auto_adjust = gr.Checkbox(label="启用自动调整", value=True)
                    restitch_max_speed = gr.Slider(label="最大速度因子", minimum=1.0, maximum=3.0, value=1.5, step=0.1)
                    restitch_button = gr.Button("开始重新拼接", variant="primary")
                with gr.Column(scale=2):
                    restitch_progress = gr.Progress(track_tqdm=True)
                    restitch_log = gr.Textbox(label="状态日志", lines=25, interactive=False, autoscroll=True)
                    restitch_audio = gr.Audio(label="重新拼接输出", type="filepath")
        with gr.TabItem("工具"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 系统维护工具")
                    clean_temp_button = gr.Button("清理系统临时文件", variant="secondary")
                    clean_temp_status = gr.Textbox(label="清理状态", interactive=False)


    # --- 收集所有可调整参数的辅助函数 ---
    all_ui_inputs_for_save_load = [
        input_prompt_dir, input_clips_dir, input_ckpt_root, input_dit_exp_name,
        input_time_step, input_p_w, input_t_w, input_dur_disturb,
        input_initial_speed, input_enable_auto_adjust, input_max_speed,
        input_max_workers, input_batch_size
    ]
    all_ui_outputs_on_load = [
        input_prompt_dir, prompt_audio_dropdown, prompt_audio_preview,
        input_clips_dir, input_ckpt_root, input_dit_exp_name,
        input_time_step, input_p_w, input_t_w, input_dur_disturb,
        input_initial_speed, input_enable_auto_adjust, input_max_speed,
        input_max_workers, input_batch_size
    ]

    # --- 事件处理 ---
    def refresh_prompt_list_and_update(directory):
        choices = get_audio_files(directory)
        return gr.update(choices=choices, value=None)
    refresh_prompts_button.click(fn=refresh_prompt_list_and_update, inputs=[input_prompt_dir], outputs=[prompt_audio_dropdown])

    def update_preview(selected_name, directory):
        wav_path = None
        if selected_name and directory and os.path.isdir(directory):
            potential_wav_path = os.path.join(directory, selected_name + ".wav")
            if os.path.exists(potential_wav_path):
                wav_path = potential_wav_path
                logging.info(f"Previewing audio: {wav_path}")
            else:
                logging.warning(f"Preview WAV not found: {potential_wav_path}")
        return wav_path
    prompt_audio_dropdown.change(
        fn=update_preview,
        inputs=[prompt_audio_dropdown, input_prompt_dir],
        outputs=[prompt_audio_preview]
    )

    def sync_inputs_to_restitch(main_clips_dir_value, main_srt_file_obj):
        srt_file_path = None
        if main_srt_file_obj and hasattr(main_srt_file_obj, 'name'):
            srt_file_path = main_srt_file_obj.name
            logging.info(f"Syncing SRT path: {srt_file_path}")
        else:
            logging.info("No SRT file object found in main tab to sync.")
        logging.info(f"Syncing clips dir: {main_clips_dir_value}")
        return gr.update(value=main_clips_dir_value), gr.update(value=srt_file_path)
    use_main_inputs_button.click(
        fn=sync_inputs_to_restitch,
        inputs=[input_clips_dir, input_srt_file],
        outputs=[restitch_clips_dir, restitch_srt_file]
    )

    start_button.click(
        fn=run_srt_dubbing_ui,
        inputs=[
            prompt_audio_dropdown, input_prompt_dir, input_srt_file,
            input_clips_dir, input_ckpt_root, input_dit_exp_name,
            input_time_step, input_p_w, input_t_w, input_dur_disturb,
            input_initial_speed, input_enable_auto_adjust, input_max_speed,
            input_max_workers, input_batch_size
        ],
        outputs=[output_log, output_audio]
    )

    def wrapped_process_restitch(clips_dir, srt_file_obj, *args, progress=gr.Progress()):
        progress(0, desc="准备中...")
        if not clips_dir or not srt_file_obj: yield "错误: 需提供目录和SRT", None; progress(1.0, desc="失败"); return
        try:
            with open(srt_file_obj.name, 'r', encoding='utf-8-sig') as f_srt: srt_content = f_srt.read()
            subtitles = list(srt.parse(srt_content))
            if not subtitles: yield "错误: SRT文件为空", None; progress(1.0, desc="失败"); return
            srt_total_duration_ms = int(subtitles[-1].end.total_seconds() * 1000)
            progress(0.5, desc="正在拼接...")
            status, audio_path = restitch_audio_clips(subtitles, clips_dir, srt_total_duration_ms, *args)
            yield status, audio_path # 只返回 log 和 audio
            progress(1.0, desc="完成")
        except Exception as e: logging.error(f"二次拼接错误", exc_info=True); yield f"处理错误: {e}", None; progress(1.0, desc="失败")
    restitch_button.click(
        fn=wrapped_process_restitch,
        inputs=[restitch_clips_dir, restitch_srt_file, restitch_initial_speed, restitch_enable_auto_adjust, restitch_max_speed],
        outputs=[restitch_log, restitch_audio]
    )

    clean_temp_button.click(fn=clean_temp_files, outputs=[clean_temp_status])

    # --- 新增：文本配音按钮的点击事件 ---
    tts_param_inputs = [
        input_ckpt_root, input_dit_exp_name, # Model params
        input_time_step, input_p_w, input_t_w, input_dur_disturb # TTS params
    ]
    generate_text_tts_button.click(
        fn=run_text_tts_ui,
        inputs=[
            text_input_tts,           # From new Textbox
            prompt_audio_dropdown,    # From Main Tab
            input_prompt_dir,         # From Main Tab
            *tts_param_inputs         # Unpack the parameter components
        ],
        outputs=[status_text_tts, output_audio_text_tts] # To new outputs
    )
    # Note: Progress bar for text_tts_tab is linked via default argument in run_text_tts_ui

    # --- 应用启动时加载逻辑 --- 
    def initial_load_behavior():
        """Loads default params and sets initial UI state."""
        logging.info("--- Running Initial Load Behavior ---")
        defaults = default_params # Use preloaded defaults
        prompt_dir = defaults.get("prompt_audio_dir", "")
        selected_prompt = defaults.get("selected_prompt_name", None)

        audio_files = get_audio_files(prompt_dir)

        final_selection = selected_prompt if selected_prompt in audio_files else None

        preview_path = None
        if final_selection and prompt_dir and os.path.isdir(prompt_dir):
            potential_wav_path = os.path.join(prompt_dir, final_selection + ".wav")
            if os.path.exists(potential_wav_path):
                preview_path = potential_wav_path
                logging.info(f"Initial preview set to: {preview_path}")

        # Return updates for components in the updated all_ui_outputs_on_load list (15 items)
        updates = [
            gr.update(value=prompt_dir),                          # 1
            gr.update(choices=audio_files, value=final_selection),# 2
            gr.update(value=preview_path),                        # 3
            gr.update(value=defaults.get("clips_save_dir_ui", "")),# 4
            gr.update(value=defaults.get("ckpt_root_dir", "./checkpoints")), # 5
            gr.update(value=defaults.get("dit_exp_name", "diffusion_transformer")), # 6
            gr.update(value=defaults.get("time_step", 32)),         # 7
            gr.update(value=defaults.get("p_w", 1.4)),             # 8
            gr.update(value=defaults.get("t_w", 3.0)),             # 9
            gr.update(value=defaults.get("dur_disturb", 0.0)),       # 10
            gr.update(value=defaults.get("initial_speed", 1.0)),     # 11
            gr.update(value=defaults.get("enable_auto_adjust", True)),# 12
            gr.update(value=defaults.get("max_speed", 1.5)),        # 13
            gr.update(value=defaults.get("max_workers", 2)),        # 14
            gr.update(value=defaults.get("batch_size", 4)),         # 15
        ]
        # Ensure the length matches the output list length (15)
        if len(updates) != len(all_ui_outputs_on_load):
             logging.error(f"Mismatch in initial load updates ({len(updates)}) vs expected outputs ({len(all_ui_outputs_on_load)})")
             updates.extend([gr.update()] * (len(all_ui_outputs_on_load) - len(updates)))

        return updates # Return the list directly
    demo.load(initial_load_behavior, inputs=None, outputs=all_ui_outputs_on_load)

# --- 启动 Gradio 应用 --- 
if __name__ == "__main__":
    logging.info("--- Gradio Application Main Entry Point ---")
    print(f"Initializing App with Audio Selection from Directory...")
    print(f"Using Device: {DEVICE}")
    print(f"Logging info (and errors) to: {log_file}")
    print(f"Default parameters loaded from/saved to: {DEFAULT_PARAMS_FILE}")
    if torch.cuda.is_available(): print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("Attempting to launch Gradio interface...")
    demo.queue().launch(server_name="0.0.0.0")
    print("Gradio server stopped.")
    logging.info("--- Application Exit ---")