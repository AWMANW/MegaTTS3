# audio_stitcher.py

import os
from pydub import AudioSegment
from pydub.effects import speedup
import traceback
from tqdm.auto import tqdm # 使用标准 tqdm

# --- 拼接模块的全局配置 ---
MAX_ITERATIONS = 15
BASE_MINIMUM_GAP_MS = 300
MIN_ALLOWABLE_GAP_MS = 160
GAP_REDUCTION_STEP_MS = 20
SPEED_INCREASE_RATE = 1.02

def _build_audio_pass(
    subtitles,
    clip_path_map,
    speed_factor,
    minimum_gap_ms,
    progress_desc="构建音频"
):
    """
    执行单次顺序音频构建的辅助函数。(内部使用)
    """
    print(f"--- Building pass: Speed={speed_factor:.3f}x, MinGap={minimum_gap_ms}ms ---")
    built_audio = AudioSegment.empty()
    current_timeline_position_ms = 0
    merge_errors_count = 0

    for i, sub in enumerate(tqdm(subtitles, desc=progress_desc, unit="片段", leave=False)):
        subtitle_index = sub.index
        original_srt_start_ms = int(sub.start.total_seconds() * 1000)

        if subtitle_index not in clip_path_map:
            continue

        clip_path = clip_path_map[subtitle_index]

        try:
            clip_audio = AudioSegment.from_file(clip_path)
            original_duration = len(clip_audio)

            if speed_factor != 1.0 and speed_factor > 0:
                adjusted_clip = speedup(clip_audio, playback_speed=speed_factor)
                adjusted_duration = len(adjusted_clip)
            else:
                adjusted_clip = clip_audio
                adjusted_duration = original_duration

            if i == 0: effective_start_ms = max(0, original_srt_start_ms)
            else:
                min_start_due_to_previous = current_timeline_position_ms + minimum_gap_ms
                effective_start_ms = max(original_srt_start_ms, min_start_due_to_previous)

            silence_duration = effective_start_ms - current_timeline_position_ms
            if silence_duration > 0: built_audio += AudioSegment.silent(duration=silence_duration)

            built_audio += adjusted_clip
            current_timeline_position_ms = effective_start_ms + adjusted_duration

        except Exception as e:
            merge_errors_count += 1
            print(f"ERROR build pass: Clip {os.path.basename(clip_path)} Idx {subtitle_index} failed: {e}")
            continue

    print(f"--- Build pass finished: Duration={len(built_audio)}ms, Errors={merge_errors_count} ---")
    return built_audio, merge_errors_count


def stitch_audio_clips(
    subtitles,
    clip_path_map,
    srt_total_duration_ms,
    initial_speed_factor,
    enable_auto_adjust,
    max_speed_factor,
    status_updates_list,
):
    """
    核心音频拼接模块。
    """
    final_audio = None
    build_errors = 0
    final_speed_factor = float(initial_speed_factor)
    final_minimum_gap_ms = BASE_MINIMUM_GAP_MS

    if not enable_auto_adjust:
        # --- 单次构建 ---
        status_updates_list.append("\n开始顺序构建音频（禁用自动调整）...")
        print("Starting single pass sequential audio construction...")
        final_audio, build_errors = _build_audio_pass(
            subtitles, clip_path_map, final_speed_factor, final_minimum_gap_ms,
            progress_desc=f"构建音频 (速度 {final_speed_factor:.2f}x)"
        )
        status_updates_list.append(f"单次构建完成。时长: {len(final_audio)/1000.0:.2f}s, 错误: {build_errors}")
        print(f"Single pass finished. Duration: {len(final_audio)}ms, Errors: {build_errors}")
        if len(final_audio) > srt_total_duration_ms: status_updates_list.append("警告: 时长超标！"); print("Warning: Duration exceeds target.")

    else:
        # --- 迭代构建 ---
        status_updates_list.append("\n开始迭代构建音频（启用自动调整）...")
        print(f"Starting iterative construction (Target: {srt_total_duration_ms} ms)")
        current_speed_factor = float(initial_speed_factor)
        current_minimum_gap_ms = BASE_MINIMUM_GAP_MS
        iteration_count = 0; stage = 1

        while iteration_count < MAX_ITERATIONS:
            iteration_count += 1
            progress_desc = f"迭代 {iteration_count} (阶段 {stage}, 速度 {current_speed_factor:.2f}x, 间隔 {current_minimum_gap_ms}ms)"
            status_updates_list.append(f"\n--- {progress_desc} ---")
            print(f"\n--- Iteration #{iteration_count} (Stage {stage}, Speed {current_speed_factor:.3f}, Gap {current_minimum_gap_ms}) ---")

            current_pass_audio, merge_errors_this_pass = _build_audio_pass(
                subtitles, clip_path_map, current_speed_factor, current_minimum_gap_ms,
                progress_desc=progress_desc
            )

            build_errors = merge_errors_this_pass; final_audio = current_pass_audio
            if len(final_audio) == 0: status_updates_list.append("错误：迭代中无有效音频。"); print("Error: No audio in iteration."); break
            current_pass_duration_ms = len(final_audio)
            status_updates_list.append(f"迭代 #{iteration_count} 完成。时长: {current_pass_duration_ms/1000.0:.2f}s (目标 <= {srt_total_duration_ms/1000.0:.2f}s)")
            print(f"Iteration #{iteration_count} finished. Duration: {current_pass_duration_ms} ms")
            if merge_errors_this_pass > 0: status_updates_list.append(f"警告: 迭代 #{iteration_count} 中有 {merge_errors_this_pass} 个合并错误。"); print(f"Warning: {merge_errors_this_pass} merge errors.")
            if current_pass_duration_ms <= srt_total_duration_ms: status_updates_list.append("目标时长已满足！"); print("Duration goal met."); break

            # --- 决定下一步 (检查并修正缩进) ---
            if stage == 1: # 尝试加速
                if current_speed_factor < max_speed_factor:
                    next_speed_factor = current_speed_factor * SPEED_INCREASE_RATE
                    current_speed_factor = min(next_speed_factor, max_speed_factor)
                    status_updates_list.append(f"时长超标，尝试加速至 {current_speed_factor:.3f}x..."); print(f"Increasing speed to {current_speed_factor:.3f}x...")
                    # 注意：这里不需要 continue，循环会自动继续到下一次迭代
                else: # 达到最大速度，进入下一阶段
                    status_updates_list.append(f"已达最大速度 ({max_speed_factor:.2f}x)，将开始缩短间隔。"); print(f"Reached max speed. Starting gap reduction.")
                    stage = 2
                    # 检查是否还能缩短间隔 (确保此 if 块的缩进正确)
                    if current_minimum_gap_ms <= MIN_ALLOWABLE_GAP_MS:
                        status_updates_list.append("无法再缩短间隔。"); print("Cannot reduce gap further.")
                        break # 退出 while 循环
                    # 如果还能缩短，则在下一次迭代开始时执行缩短 (stage == 2 分支)

            elif stage == 2: # 尝试缩短间隔
                if current_minimum_gap_ms > MIN_ALLOWABLE_GAP_MS:
                    current_minimum_gap_ms = max(MIN_ALLOWABLE_GAP_MS, current_minimum_gap_ms - GAP_REDUCTION_STEP_MS)
                    status_updates_list.append(f"时长仍超标，尝试缩短间隔至 {current_minimum_gap_ms}ms..."); print(f"Reducing gap to {current_minimum_gap_ms}ms...")
                    # 注意：这里不需要 continue
                else: # 达到最小间隔
                    status_updates_list.append(f"已达最小间隔 ({MIN_ALLOWABLE_GAP_MS}ms)，无法再缩短。"); print(f"Reached minimum gap. Cannot reduce further.");
                    break # 退出 while 循环
            # --- 缩进修正结束 ---

        # --- 循环结束处理 ---
        final_speed_factor = current_speed_factor; final_minimum_gap_ms = current_minimum_gap_ms
        if iteration_count >= MAX_ITERATIONS and (final_audio is None or len(final_audio) > srt_total_duration_ms): status_updates_list.append(f"\n警告：达到最大迭代次数({MAX_ITERATIONS})，仍超长。"); print(f"Warning: Reached max iterations({MAX_ITERATIONS}).")
        status_updates_list.append(f"\n最终参数 -> 速度: {final_speed_factor:.3f}x, 间隔: {final_minimum_gap_ms}ms"); print(f"Final params -> Speed: {final_speed_factor:.3f}x, Min Gap: {final_minimum_gap_ms}ms")

    return final_audio, final_speed_factor, final_minimum_gap_ms, build_errors
# --- [模块结束] ---