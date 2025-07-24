import argparse

from termcolor import colored
import cv2
import llava
from llava import conversation as clib
from llava.media import Image, Video
import torch
from awq.quantize import fake_quant
from tinychat.models.nvila_qwen2 import NVILAQwen2
from transformers import AutoConfig
from tinychat.models.qwen2 import Qwen2ForCausalLM
from tinychat.utils.load_quant import load_awq_model
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
from tinychat.evaluator.evaluator import Evaluator, elaborate_predictions_and_gt
from tinychat.evaluator.evaluator import extract_json_block
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def main(args):
    # Accelerate model initialization
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
        model = NVILAQwen2(config, True).half()

    if args.smooth_VT or args.all:
        from awq.quantize import smooth_lm

        act_scales = torch.load(args.act_scale_path)
        smooth_lm(model.vision_tower, act_scales, 0.3)
    if args.quant_llm or args.all:
        model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
        model.llm = load_awq_model(model.llm, args.quant_path, 4, 128, args.device)
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
    model = model.cuda().eval()
    #print(model)
    device_warmup(args.device)
    tune_llava_patch_embedding(model.vision_tower, device=args.device)
    count = 0
    # Pre-prepare media
    all_predictions = []
    all_ground_truth = []
    for files in args.test_images_folder:
        for root, _, images in os.walk(files):
            #print(root)
            #print(images)
            #image_dir = os.path.dirname(file)
            for image_file in images:
                if any(image_file.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                    media_path = os.path.join(root, image_file)
                    img = cv2.imread(media_path)
                    resized_img = cv2.resize(img, (640, 480))
                    cv2.imwrite(media_path,resized_img)
                    print(f"Elaborate image {media_path}")
                    prompt = []
                    if media_path is not None:
                        media = Image(media_path)
                    else:
                        raise ValueError(f"Unsupported media: {media_path}")
                    if args.vis_image:
                        print("=" * 50)
                        print("Input Image:")
                        vis_images([media_path])
                    prompt.append(media)
                    conversation = [{"from": "human", "value": prompt}]
                    media, media_cfg = model.prepare_media(conversation)
                    # Prepare streaming
                    stream_generator = NVILAStreamGenerator
                    # Prepare prompt
                    if args.max_seq_len <= 1024:
                        short_prompt = True
                    else:
                        short_prompt = False
                    model_prompter = get_prompter(
                        args.model_type, args.model_path, short_prompt, args.empty_prompt
                    )
                    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)

                    if args.empty_prompt:
                        input_indicator = "Input: "
                        output_indicator = "Generated: "
                    else:
                        input_indicator = "USER: "
                        output_indicator = "ASSISTANT: "

                    model.eval()
                    time_stats = TimeStats()
                    start_pos = 0
                    print("=" * 50)
                    input_prompt = """
                    You are a robot in a room with multiple people. Your task is to identify individuals who are available for interaction. Use the following criteria to determine the most suitable candidates for interaction: 
                    1. Group Dynamics: Prefer isolated individuals over people in small groups. Two people are likely engaged in a conversation if they are facing each other, maintaining close proximity, and showing body language cues such as hand gestures or mutual gaze. 
                    2. Availability Cues: Focus on individuals who are not engaged in conversations, using a phone or laptop, wearing headphones or reading a book. Cell phones, books and headphones are located close to their hand, face or chest area. 
                    3. Orientation: Prioritize people whose head are directed towards the robot. 
                    4. Proximity: If multiple people meet the above criteria, prioritize those closest to the robot. 
                    Available individuals should come first, sorted by priority based on the established criteria. 
                    If an individual is heavily occluded or not completely visible in the image, they should be considered not available and assigned the lowest priority.
                    Non-available individuals should follow, prioritized as follows: 
                    1. Prioritize individuals who are oriented toward the robot (those facing it with both head and body) over those who are partially or fully turned away. When individuals are at the same distance, prefer the one with a more direct orientation toward the robot. 
                    2. Among individuals with similar orientation (head and body), rank them based on proximity, giving higher priority to those closer to the robot. 
                    In the images, people are identified with bounding boxes and numeric identifiers. The numeric identifier is drawn at the bottom left of each person. 
                    Now, process this image, considering that it originates from a camera located up to you: <image> 
                    Output format. Return a JSON object with two lists: 
                    1. Priority: a list of person numeric identifiers, sorted from most to least suitable for interaction, according to the aforementioned criteria. 
                    2. Availability: a list of binary values (1 = available, 0 = not available). Ensure that individuals with the highest priority appearing first in the list. 
                    If two or more individuals have equal priority, sort them by their numeric identifiers in ascending order. 
                    Explain me the reasoning behind your choices.
                    """
                    #input_prompt = "<image>"+"What does this image represent?"
                    #print(input_prompt)
                    input_prompt = input_indicator + input_prompt
                    #print(input_prompt)
                    print("-" * 50)
                    time_stats.show()

                    model_prompter.insert_prompt(input_prompt)
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
                    label_file = media_path.replace("images", "labels").replace(".jpg", ".txt")
                    if not os.path.exists(label_file):
                        print(f"Label file {label_file} does not exist, skipping evaluation.")
                        continue
                    with open(label_file, "r") as f:
                        content = f.read()
                        #print(content)
                    try:
                        json_str_label = extract_json_block(content)
                        json_str_response = extract_json_block(outputs)
                        print(json_str_response)
                    except Exception as e:
                        print(f"Error decoding JSON from {label_file}: {e}")
                        continue
                    predictions, ground_truth = elaborate_predictions_and_gt([json_str_response], [json_str_label])
                    all_predictions.append(predictions)
                    #print(f"Debug: {all_predictions}")
                    all_ground_truth.append(ground_truth)
                    #print(f"Debug: {all_ground_truth}")
                    time_stats.show()
                    if args.chunk_prefilling:
                        start_pos += total_tokens
                    if (
                        args.single_round is not True and args.max_seq_len > 512
                    ):  # Only memorize previous conversations when kv_cache_size > 512
                        model_prompter.update_template(outputs, args.chunk_prefilling)
                    count += 1
            else:
                continue
    #print(f"Debug: {all_predictions}")
    #print(f"Debug: {all_ground_truth}")
    evaluator = Evaluator(all_predictions, all_ground_truth)
    results = evaluator.evaluate()
    print(f"Evaluation results: {results}")


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
        "--act_scale_path",
        type=str,
        default="/PATH/TO/SCALE",
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
    parser.add_argument("--test_images_folder", type=str, default=["/home/workspace/social_interaction_dataset/social_interaction_dataset_test/images", "/home/workspace/social_interaction_dataset_1/social_interaction_dataset_1_test/images"])

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

"""
    Now I will show you some examples:
    Example 1. """+"<image>"+"""\n Expected output: {"priority": [1, 2, 4, 3]} .
    Example 2. """+"<image>"+"""\n Expected output: {"priority": [1, 2]} .
    Example 3. """+"<image>"+"""\n Expected output: {"priority": [2, 1]} .
    Example 4. """+"<image>"+"""\n Expected output: {"priority": [1, 3, 2]} ."""


"""
   You are a robot in a room with multiple people. Your task is to identify individuals who are available for interaction. 
    Use the following criteria to determine the most suitable candidates for interaction: 
    1. Group Dynamics: Prefer isolated individuals over people in small groups. Two people are likely engaged in a conversation if they are facing each other, maintaining close proximity, and showing body language cues such as hand gestures or mutual gaze.
    2. Availability Cues: Focus on individuals who are not engaged in conversations, using a phone or laptop or wearing headphones. Cell phones and headphones are located close to their hand, face or chest area.
    3. Orientation: Prioritize people whose head are directed towards the robot.
    4. Proximity: If multiple people meet the above criteria, prioritize those closest to the robot.
    Available individuals should come first, sorted by priority based on the established criteria.
    Non-available individuals should follow, prioritized as follows:
    1. Individuals facing the robot are ranked higher than those facing away.
    2. Among individuals with the same head orientation, prioritize based on proximity (closer individuals ranked higher).
    In the images, people are identified with bounding boxes and numerical identifiers. The numeric identifier is drawn at the bottom left of each person.
    Now, I will show you some examples:
    EXAMPLE 1:
    INPUT: """ + "<image>" + """
    REASONING:
    Person 1 is available: isolated, device-free, facing the robot. 
    Person 2 is unavailable: isolated and closest, but using a phone, not facing the robot. 
    Person 3 is unavailable: in a group with Person 4, not facing the robot, and fully turned around. 
    Person 4 is unavailable: in a group with Person 3, not facing the robot, but closer than Person 4. 
    OUTPUT: {"priority": [1, 2, 4, 3]} . 
    EXAMPLE 2:
    INPUT: """ + "<image>" + """
    REASONING:
    Person 2 is available: facing the robot, not using any device, smiling, and not engaged in conversation.
    Person 1 is unavailable: facing away from the robot, focused on a laptop, with posture indicating engagement.
    OUTPUT: {"priority": [2, 1]} .
    EXAMPLE 3:
    INPUT: """ + "<image>" + """
    REASONING:
    Person 1 is available: isolated, not using any device, and closest to the robot.    
    Person 2 is unavailable: in a group with Person 3, not facing the robot, farther than Person 1.
    Person 3 is unavailable: in a group with Person 2, facing away from the robot, slightly farther than Person 2.
    OUTPUT: {"priority": [1, 2, 3]} .
    EXAMPLE 4:
    INPUT: """ + "<image>" + """
    REASONING:
    Person 2 is available: isolated, facing the robot, not using any device or wearing headphones.
    Person 1 is unavailable: isolated, wearing headphones and using a phone, not directly facing the robot.
    OUTPUT: {"priority": [2, 1]} .
    Now, I will show you an image. Please follow the instructions and provide the output in the same format as the examples.
    INPUT: """ + "<image>" + """
    OUTPUT FORMAT: Tell me how many people you see in the image. Then, return the list of numeric identifiers sorted by priority in the same format as the examples. Explain me the reasoning behind that choice. 
    """
