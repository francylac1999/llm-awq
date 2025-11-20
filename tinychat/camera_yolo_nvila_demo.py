import argparse

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
import torch
import time
from awq.quantize import fake_quant
from tinychat.models.nvila_qwen2 import NVILAQwen2
from transformers import AutoConfig
from tinychat.models.qwen2 import Qwen2ForCausalLM
from tinychat.utils.load_quant import load_non_quantized_model #, load_awq_model
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


def skip(*args, **kwargs):
    pass


from tinychat.utils.tune import (
    device_warmup,
    tune_all_wqlinears,
    tune_llava_patch_embedding,
)
from tinychat.utils.prompt_templates import (
    get_prompter,
    get_stop_token_ids,
    get_image_token,
)
from llava.utils.media import extract_media
import tinychat.utils.constants
from tinychat.stream_generators.NVILA_stream_gen import NVILAStreamGenerator
from tinychat.utils.conversation_utils import gen_params, stream_output, TimeStats

import os
import cv2
import numpy as np
import pyrealsense2 as rs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def detect_people(image_input, yolo_model, output_dir=None):
    """
    Accepts either a file path (str) or an ndarray `image_input` (from cv2).
    Runs YOLO on the image, draws person boxes and numeric ids, saves the
    resulting image to `output_dir` (or a default camera output directory)
    and returns the saved path and the detection time.
    """
    t_detect_start = time.perf_counter()
    input_is_path = isinstance(image_input, str)
    if input_is_path:
        image = cv2.imread(image_input)
    else:
        image = image_input

    tracking_id = 0
    # ultralytics YOLO accepts ndarray inputs
    results = yolo_model(image)
    boxes = results[0].boxes
    output_path = None

    for box in boxes:
        if int(box.cls[0]) == 0:  # Class 0 corresponds to 'person' in COCO dataset
            tracking_id += 1
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])

            # Draw the bounding box and id
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

    # Prepare output path
    if input_is_path:
        dir_name = os.path.dirname(image_input)
        out_base = output_dir if output_dir is not None else dir_name
        output_dir_final = os.path.join(out_base, os.path.basename(dir_name))
        os.makedirs(output_dir_final, exist_ok=True)
        output_path = os.path.join(output_dir_final, os.path.basename(image_input))
    else:
        out_base = output_dir if output_dir is not None else os.path.join(os.getcwd(), "camera_output")
        os.makedirs(out_base, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(out_base, f"camera_{timestamp}.jpg")

    print(f"Saving image in {output_path}")
    cv2.imwrite(output_path, image)

    t_detect_end = time.perf_counter()
    detect_time = t_detect_end - t_detect_start
    return output_path, detect_time

def _output_dir_for_input_folder(folder):
    # Map a folder containing 'images_without_boxes' to the corresponding
    # 'images_with_boxes' path in the same dataset directory.
    if 'images_without_boxes' in folder:
        return folder.replace('images_without_boxes', 'images_with_boxes')
    # Fallback: create an images_with_boxes sibling under the parent
    parent = os.path.dirname(folder.rstrip('/'))
    return os.path.join(parent, 'images_with_boxes')

def main(args):
    # Accelerate model initialization
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Prepare model
    config = AutoConfig.from_pretrained(args.model_path)
    config.resume_path = args.model_path
    if args.quant_llm or args.all:
        model = NVILAQwen2(config, False).half()
    else:
        model = NVILAQwen2(config, False).half()
    if args.smooth_VT or args.all:
        from awq.quantize import smooth_lm

        act_scales = torch.load(args.act_scale_path)
        smooth_lm(model.vision_tower, act_scales, 0.3)
    if args.quant_llm or args.all:
        model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
        #model.llm = load_awq_model(model.llm, args.quant_path, 4, 128, args.device)
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
    device_warmup(args.device)
    tune_llava_patch_embedding(model.vision_tower, device=args.device)
    yolo_model = ultralytics.YOLO('yolo11x.pt')  # load a pretrained YOLOv11x model

    # Pre-prepare media
    # Timing lists
    detect_times = []
    model_times = []
    pipeline_times = []

    # Prompt setup and short_prompt flag
    input_indicator = "USER: "
    output_indicator = "ASSISTANT: "
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

    # --- Camera capture loop (replace folder-based processing) ---
    output_dir_base = args.camera_output_dir if hasattr(args, "camera_output_dir") else os.path.join(os.getcwd(), "camera_output")
    model_prompter = get_prompter(
        args.model_type, args.model_path, short_prompt, args.empty_prompt
    )
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    # Insert prompt into the prompter
    input_prompt = input_indicator + base_prompt
    model_prompter.insert_prompt(input_prompt)
    stream_generator = NVILAStreamGenerator

    cap = None
    rs_pipeline = None

    if args.camera_type == "realsense":
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            rs_pipeline.start(rs_config)
        except Exception as err:
            print(f"Error: unable to start RealSense pipeline ({err})")
            return
    else:
        cam_index = args.camera_index if hasattr(args, "camera_index") else 0
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Error: unable to open camera index {cam_index}")
            return

    frame_count = 0
    try:
        while True:
            if args.camera_type == "realsense":
                try:
                    frames = rs_pipeline.wait_for_frames(timeout_ms=5000)
                except Exception as err:
                    print(f"RealSense frame timeout: {err}")
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("No color frame captured from RealSense, skipping.")
                    continue
                frame = np.asarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    print("No frame captured from camera, stopping.")
                    break

            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            t_pipeline_start = time.perf_counter()
            output_path, detect_time = detect_people(frame, yolo_model, output_dir=output_dir_base)
            print(f"Processing image: {output_path}")

            prompt = []
            media = Image(output_path)
            prompt.append(media)
            conversation = [{"from": "human", "value": prompt}]
            media, media_cfg = model.prepare_media(conversation)

            model.eval()
            time_stats = TimeStats()
            time_stats.show()
            start_pos = 0

            # model inference timing
            t_model_start = time.perf_counter()
            output_stream = stream_generator(
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

            frame_count += 1
            if hasattr(args, "max_frames") and args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames ({args.max_frames}), stopping.")
                break

    finally:
        if cap is not None:
            cap.release()
        if rs_pipeline is not None:
            rs_pipeline.stop()

    # Print timing summary
    try:
        n = len(pipeline_times)
        if n > 0:
            print("\n=== Timing Summary ===")
            print(f"Images processed: {n}")
            print(f"Detection times: total={sum(detect_times):.3f}s avg={sum(detect_times)/n:.3f}s min={min(detect_times):.3f}s max={max(detect_times):.3f}s")
            print(f"Model inference times: total={sum(model_times):.3f}s avg={sum(model_times)/n:.3f}s min={min(model_times):.3f}s max={max(model_times):.3f}s")
            print(f"Full pipeline times: total={sum(pipeline_times):.3f}s avg={sum(pipeline_times)/n:.3f}s min={min(pipeline_times):.3f}s max={max(pipeline_times):.3f}s")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model-path", type=str, default="/data/llm/checkpoints/llava/llava-v1.5-7b"
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt",
    )
    parser.add_argument(
        "--llm-checkpoint",
        dest="llm_checkpoint",
        type=str,
        default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt",
        help="Path to non-quantized LLM checkpoint (accepts --llm-checkpoint or --llm_checkpoint)",
    )
    parser.add_argument(
        "--act_scale_path",
        type=str,
        default="/PATH/TO/SCALE",
    )
    parser.add_argument(
        "--media", type=str, nargs="+", help="Multi-modal input (Video or image path)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--single_round",
        action="store_true",
        help="whether to memorize previous conversations",
    )
    parser.add_argument(
        "--vis-image",
        action="store_true",
        help="whether to visualize the image while chatting",
    )
    parser.add_argument(
        "--empty-prompt",
        action="store_true",
        help="whether to use empty prompt template",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="whether to use flash attention",
    )
    parser.add_argument(
        "--chunk_prefilling",
        action="store_true",
        help="If used, in context stage, the history tokens will not be recalculated, greatly speeding up the calculation",
    )
    parser.add_argument(
        "--test_images_folder",
        type=str,
        nargs="+",
        help="Folder containing test images",
        default = ["/home/workspace/social_interaction_dataset_test_without_boxes/images_without_boxes/","/home/workspace/social_interaction_dataset_1_test_without_boxes/images_without_boxes/","/home/workspace/social_interaction_dataset_2_new_test_without_boxes/images_without_boxes/","/home/workspace/social_interaction_dataset_3_test_without_boxes/images_without_boxes/"]
    )
    # smooth and quantization options
    parser.add_argument("--quant_llm", action="store_true")
    parser.add_argument("--quant_VT", action="store_true")
    parser.add_argument("--smooth_VT", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--fakequant_VT",
        action="store_true",
        help="Use fake quant or real quant for VisionTower",
    )
    parser.add_argument(
        "--custom_stop_token",
        type=bool,
        help="Custom stop token for text generation",  
        default=True,         
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        help="Camera index",
        default=0,
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        choices=["opencv", "realsense"],
        help="Camera backend",
        default="realsense",
    )
    parser.add_argument(
        "--camera_output_dir",
        type=str,
        help="Directory to save camera output images",  
        default='./camera_output',         
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        help="Maximum number of frames to process from the camera (0 for infinite)",  
        default=10,         
    )
    args = parser.parse_args()
    main(args)
"""
python nvila_demo.py --model-path /home/yuming/workspace/qwen/models/nvila-video       \
    --quant_path /home/yuming/workspace/awq4nvila/nvila-video-w4-g128.pt      \
    --media ../figures/nvila_demo_video.mp4     \
    --act_scale_path /home/yuming/workspace/awq4nvila/nvila-video-VT-smooth-scale.pt \
    --all --chunk --vis-image

python nvila_demo.py --model-path Efficient-Large-Model/nvila-internal-8b-v1       \
    --quant_path /home/yuming/workspace/awq4nvila/nvila-internal-8b-v1-w4-g128.pt      \
    --media ../figures/vila-logo.jpg    \
    --act_scale_path /home/yuming/workspace/awq4nvila/nvila-internal-8b-v1-VT-smooth-scale.pt \
    --all --chunk --vis-image

python nvila_demo.py --model-path /home/yuming/workspace/qwen/models/nvila-lite-internal-8b-v1   \
    --quant_path /home/yuming/workspace/awq4nvila/nvila-lite-internal-8b-v1-w4-g128.pt  \
     --act_scale_path /home/yuming/workspace/awq4nvila/nvila-lite-internal-8b-v1-VT-smooth-scale.pt --all \
    --media ../figures/vila-logo.jpg  --chunk --vis-image
"""
