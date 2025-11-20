import argparse
import os
import time

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
import torch
from awq.quantize import fake_quant
from tinychat.models.nvila_qwen2 import NVILAQwen2
from transformers import AutoConfig
from tinychat.models.qwen2 import Qwen2ForCausalLM
from tinychat.utils.load_quant import load_non_quantized_model
from tinychat.modules import (
    make_quant_norm,
    make_quant_attn,
    make_fused_mlp,
    make_fused_vision_attn,
)
from tinychat.utils.llava_image_processing import (
    load_images,
    vis_images,
)
import ultralytics
import cv2
import tempfile


def skip(*args, **kwargs):
    pass


def detect_people(image_input, yolo_model, output_dir=None):
    """
    Annotate the input image with person boxes and return the annotated image (numpy array)
    and the detection time. This function no longer saves files to disk.
    """
    t_detect_start = time.perf_counter()
    input_is_path = isinstance(image_input, str)
    if input_is_path:
        image = cv2.imread(image_input)
    else:
        image = image_input

    tracking_id = 0
    results = yolo_model(image)
    boxes = results[0].boxes

    for box in boxes:
        if int(box.cls[0]) == 0:  # person
            tracking_id += 1
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                str(tracking_id),
                (x1 + 5, y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    t_detect_end = time.perf_counter()
    detect_time = t_detect_end - t_detect_start
    return image, detect_time


def _output_dir_for_input_folder(folder):
    if 'images_without_boxes' in folder:
        return folder.replace('images_without_boxes', 'images_with_boxes')
    parent = os.path.dirname(folder.rstrip('/'))
    return os.path.join(parent, 'images_with_boxes')


def main(args):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(args.model_path)
    config.resume_path = args.model_path
    model = NVILAQwen2(config, False).half()

    if args.smooth_VT or args.all:
        from awq.quantize import smooth_lm

        act_scales = torch.load(args.act_scale_path)
        smooth_lm(model.vision_tower, act_scales, 0.3)

    if args.quant_llm or args.all:
        model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
        make_quant_attn(model.llm, args.device, True)
        make_quant_norm(model.llm)
        model.llm.cpu()
        model.llm.resize_token_embeddings(len(model.tokenizer))

    if args.quant_VT or args.all:
        from tinychat.modules import QuantSiglipEncoder

        if args.fakequant_VT:
            fake_quant(model.vision_tower.vision_tower.vision_model.encoder)
        else:
            model.vision_tower.vision_tower.vision_model.encoder = QuantSiglipEncoder(
                model.vision_tower.vision_tower.vision_model.encoder
            )

    model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
    model.llm = load_non_quantized_model(model.llm, args.llm_checkpoint, args.device)
    model.llm.cpu()
    model.llm.resize_token_embeddings(len(model.tokenizer))
    model = model.cuda().eval()
    tune_llava_patch_embedding = None
    try:
        from tinychat.utils.tune import tune_llava_patch_embedding

        tune_llava_patch_embedding(model.vision_tower, device=args.device)
    except Exception:
        pass

    yolo_model = ultralytics.YOLO('yolo11x.pt')

    detect_times = []
    model_times = []
    pipeline_times = []

    input_indicator = "USER: "
    output_indicator = "ASSISTANT: "
    from tinychat.utils.prompt_templates import (
        get_prompter,
        get_stop_token_ids,
        get_image_token,
    )

    image_token = get_image_token(args.model_type, args.model_path)
    short_prompt = args.max_seq_len <= 1024
    base_prompt = (
        "You are a robot in a room with multiple people. Your task is to identify individuals who are available for interaction. "
        "Use the following criteria to determine the most suitable candidates for interaction:\n"
        "1. Group Dynamics: Prefer isolated individuals over people in small groups. Two people are likely engaged in a conversation if they are facing each other, maintaining close proximity, "
        "and showing body language cues such as hand gestures or mutual gaze.\n"
        "2. Availability Cues: Focus on individuals who are not engaged in conversations, using a phone or laptop, wearing headphones or reading a book. "
        "Cell phones, books and headphones are located close to their hand, face or chest area.\n"
        "3. Orientation: Prioritize people whose head are directed towards the robot.\n"
        "4. Proximity: If multiple people meet the above criteria, prioritize those closest to the robot.\n"
        "Available individuals should come first, sorted by priority based on the established criteria.\n"
        "If an individual is heavily occluded or not completely visible in the image, they should be considered not available and assigned the lowest priority.\n"
        "Non-available individuals should follow, prioritized as follows:\n"
        "1. Prioritize individuals who are oriented toward the robot (those facing it with both head and body) over those who are partially or fully turned away. When individuals are at the same distance, prefer the one with a more direct orientation toward the robot.\n"
        "2. Among individuals with similar orientation (head and body), rank them based on proximity, giving higher priority to those closer to the robot.\n"
        "In the images, people are identified with bounding boxes and numeric identifiers. The numeric identifier is drawn at the bottom left of each person.\n"
        f"Now, process this image, considering that it originates from a camera located up to you: {image_token}\n"
        "Output format. Return a JSON object with two lists:\n"
        "1. Priority: a list of person numeric identifiers, sorted from most to least suitable for interaction, according to the aforementioned criteria.\n"
        "2. Availability: a list of binary values (1 = available, 0 = not available). Ensure that individuals with the highest priority appearing first in the list.\n"
        "If two or more individuals have equal priority, sort them by their numeric identifiers in ascending order.\n"
        "Explain me the reasoning behind your choices."
    )

    # If user passed an explicit video_output_dir use it, otherwise use ./video_output
    output_dir_base = args.video_output_dir if getattr(args, "video_output_dir", None) else os.path.join(os.getcwd(), "video_output")
    model_prompter = get_prompter(
        args.model_type, args.model_path, short_prompt, args.empty_prompt
    )
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    input_prompt = input_indicator + base_prompt
    model_prompter.insert_prompt(input_prompt)
    from tinychat.stream_generators.NVILA_stream_gen import NVILAStreamGenerator
    from tinychat.utils.conversation_utils import gen_params, stream_output, TimeStats

    # choose video path from --media or explicit --video-path
    video_path = None
    if args.media and len(args.media) > 0:
        video_path = args.media[0]
    if args.video_path:
        video_path = args.video_path
    if not video_path:
        print("No video path provided. Use --media or --video-path.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: unable to open video {video_path}")
        return

    frame_count = 0
    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or unable to read frame.")
                break

            frame_count += 1
            if args.frame_skip > 1 and (frame_count - 1) % args.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))

            t_pipeline_start = time.perf_counter()
            annotated_image, detect_time = detect_people(frame, yolo_model, output_dir=output_dir_base)

            # write annotated image to a temporary file for the model input,
            # then move it to the visible output folder only after model inference
            tmp_dir = os.path.join(output_dir_base, ".tmp_frames")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_f = tempfile.NamedTemporaryFile(suffix=".jpg", dir=tmp_dir, delete=False)
            tmp_path = tmp_f.name
            tmp_f.close()
            cv2.imwrite(tmp_path, annotated_image)
            print(f"Prepared temp image for inference: {tmp_path}")

            prompt = []
            media = Image(tmp_path)
            prompt.append(media)
            conversation = [{"from": "human", "value": prompt}]
            media, media_cfg = model.prepare_media(conversation)

            model.eval()
            time_stats = TimeStats()
            time_stats.show()
            start_pos = 0

            t_model_start = time.perf_counter()
            output_stream = NVILAStreamGenerator(
                model,
                gen_params,
                model_prompter.model_input,
                media,
                media_cfg,
                start_pos,
                device=args.device,
                stop_token_ids=stop_token_ids,
                chunk_prefilling=args.chunk_prefilling,
                quant_llm=args.quant_llm or args.all,
            )
            print(output_indicator, end="", flush=True)
            outputs, total_tokens = stream_output(output_stream, time_stats)
            t_model_end = time.perf_counter()
            model_time = t_model_end - t_model_start
            pipeline_time = t_model_end - t_pipeline_start
            detect_times.append(detect_time)
            model_times.append(model_time)
            pipeline_times.append(pipeline_time)

            # move the temp file to the final output folder now that inference is done
            out_base = output_dir_base if output_dir_base is not None else os.path.join(os.getcwd(), "video_output")
            os.makedirs(out_base, exist_ok=True)
            timestamp = int(time.time() * 1000)
            final_name = f"frame_{frame_count}_{timestamp}.jpg"
            final_path = os.path.join(out_base, final_name)
            try:
                os.replace(tmp_path, final_path)
                print(f"Saved final annotated frame: {final_path}")
            except Exception:
                # fallback to copy if replace fails
                import shutil

                shutil.copy(tmp_path, final_path)
                os.remove(tmp_path)
                print(f"Saved final annotated frame (copied): {final_path}")

            processed_frames += 1
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                print(f"Reached max frames ({args.max_frames}), stopping.")
                break

    finally:
        cap.release()

    try:
        n = len(pipeline_times)
        if n > 0:
            print("\n=== Timing Summary ===")
            print(f"Frames processed: {n}")
            print(f"Detection times: total={sum(detect_times):.3f}s avg={sum(detect_times)/n:.3f}s min={min(detect_times):.3f}s max={max(detect_times):.3f}s")
            print(f"Model inference times: total={sum(model_times):.3f}s avg={sum(model_times)/n:.3f}s min={min(model_times):.3f}s max={max(model_times):.3f}s")
            print(f"Full pipeline times: total={sum(pipeline_times):.3f}s avg={sum(pipeline_times)/n:.3f}s min={min(pipeline_times):.3f}s max={max(pipeline_times):.3f}s")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="LLaMa", help="type of the model")
    parser.add_argument("--model-path", dest="model_path", type=str, default="/data/llm/checkpoints/llava/llava-v1.5-7b")
    parser.add_argument("--video-path", dest="video_path", type=str, help="single video path to process")
    parser.add_argument("--media", type=str, nargs="+", help="Multi-modal input (Video or image path)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--frame-skip", dest="frame_skip", type=int, default=210, help="Process every Nth frame")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0, help="Max number of frames to process (0 = all)")
    parser.add_argument("--empty-prompt", action="store_true", help="whether to use empty prompt template")
    parser.add_argument("--chunk_prefilling", action="store_true", help="use chunk prefilling optimization")
    parser.add_argument("--quant_llm", action="store_true")
    parser.add_argument("--quant_VT", action="store_true")
    parser.add_argument("--smooth_VT", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fakequant_VT", action="store_true", help="Use fake quant or real quant for VisionTower")
    parser.add_argument("--llm-checkpoint", dest="llm_checkpoint", type=str, default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt", help="Path to non-quantized LLM checkpoint")
    parser.add_argument("--act_scale_path", type=str, default="/PATH/TO/SCALE")
    parser.add_argument("--video_output_dir", type=str, default=None, help="Where to save frames with boxes (defaults to ./video_output)")
    args = parser.parse_args()
    main(args)
