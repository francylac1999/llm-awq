import torch
import gc
import time
from typing import Optional

from .llava_stream_gen import prepare_logits_processor

context_tokens = 0
context_time = 0.0
total_tokens = 0
generation_time_list = []


@torch.inference_mode()
def NVILAStreamGenerator(
    model,
    gen_params,
    input: str,
    media=None,
    media_cfg=None,
    start_pos: int = 0,
    device: str = "cuda:0",
    stream_interval: int = 2,
    echo: bool = False,
    stop_token_ids=[],
    image_tensor: Optional[torch.FloatTensor] = None,
    chunk_prefilling: bool = False,
    quant_llm: bool = False,
):
    if chunk_prefilling and start_pos != 0:
        input = "<|im_start|>" + input
    input_ids = model.tokenizer(input)["input_ids"]
    output_ids = list(input_ids)
    input_echo_len = len(output_ids)
    len_input = len(input)
    if gen_params.top_k <= 0:
        top_k = gen_params.n_vocab
    else:
        top_k = gen_params.top_k
    logits_processor = prepare_logits_processor(
        gen_params.temp, gen_params.repeat_penalty, gen_params.top_p, top_k
    )
    past_key_values = out = None
    stop_token_ids.append(model.tokenizer.eos_token_id)
    max_new_tokens = gen_params.n_predict
    # If we see token 335, we should stop at the next generated token.
    stop_on_next = False

    for i in range(max_new_tokens):
        torch.cuda.synchronize()
        t_st = time.time()

        if i == 0:
            inputs = torch.as_tensor([input_ids], device=device)
        else:
            inputs = torch.as_tensor([[token]], device=device)
        out, length = model.stream_gen(
            input_ids=inputs,
            media=media,
            media_cfg=media_cfg,
            start_pos=start_pos,
            chunk_prefilling=chunk_prefilling,
            quant_llm=quant_llm,
        )
        start_pos += length
        logits = out
        torch.cuda.synchronize()
        t_ed = time.time()
        media = None
        media_cfg = None
        if torch.sum(torch.isinf(logits)):
            print(
                "{a} of {b}".format(
                    a=torch.sum(torch.isinf(logits)).item(), b=logits.numel()
                )
            )
            print("{},{}".format(torch.max(logits), torch.min(logits)))
        # Processing the logits
        if logits_processor:
            if gen_params.repeat_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                # tmp_output_ids = output_ids[0].unsqueeze(0)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[:, -1, :]
        if gen_params.temp < 1e-5 or gen_params.top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits.float(), dim=-1)
            if torch.any(torch.isinf(probs)):
                print("Error: Invalid probabilities detected (Inf exists). Exiting...")
                exit()
            token = int(torch.multinomial(probs, num_samples=1))
        output_ids.append(token)

        global context_time
        global context_tokens
        global total_tokens
        global generation_time_list
        if i == 0:
            context_time = t_ed - t_st
            context_tokens = length
            generation_time_list = []
        else:
            generation_time_list.append(t_ed - t_st)

        # Stop if current token is an explicit stop id, or if a previous
        # generation set the "stop_on_next" flag because it encountered 335.
        if token in stop_token_ids:
            stopped = True
        elif stop_on_next:
            stopped = True
        else:
            stopped = False

        # If current token is 335 we mark to stop on the next generated token
        # (do not stop immediately on 335 itself).
        if token == 335:
            stop_on_next = True

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_input
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = model.tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                    "timing": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    total_tokens = context_tokens + len(generation_time_list)
    
    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
        "timing": {
            "context_tokens": context_tokens,
            "context_time": context_time,
            "total_tokens": total_tokens,
            "generation_time_list": generation_time_list,
        },
    }

    # Print detailed generation statistics
    try:
        gen_times = list(generation_time_list) if generation_time_list is not None else []
        gen_tokens = len(gen_times)
        total_gen_time = float(sum(gen_times)) if gen_times else 0.0
        avg_time = total_gen_time / gen_tokens if gen_tokens else 0.0
        median_time = 0.0
        min_time = 0.0
        max_time = 0.0
        if gen_tokens:
            sorted_times = sorted(gen_times)
            median_time = sorted_times[gen_tokens // 2]
            min_time = sorted_times[0]
            max_time = sorted_times[-1]

        tokens_per_sec = gen_tokens / total_gen_time if total_gen_time > 0 else (float('inf') if gen_tokens > 0 else 0.0)

        total_time = float(context_time) + total_gen_time
        generated_tokens = gen_tokens

        print("\n=== Generation statistics ===")
        print(f"Context tokens: {context_tokens}")
        print(f"Context time (s): {context_time:.4f}")
        print(f"Generated tokens: {generated_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time (s): {total_time:.4f}")
        print("-- per-token generation time (s) --")
        print(f"  mean: {avg_time:.6f}")
        print(f"  median: {median_time:.6f}")
        print(f"  min: {min_time:.6f}")
        print(f"  max: {max_time:.6f}")
        print(f"Generation speed (tokens/sec): {tokens_per_sec:.2f}")
        print("==============================\n")
    except Exception:
        # Best-effort stats printing; do not fail the generator if something goes wrong
        pass

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()

    # return context_tokens, context_time, total_tokens, generation_time_list
