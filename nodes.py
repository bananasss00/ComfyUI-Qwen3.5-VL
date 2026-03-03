import os
import gc
import re
import time
import shutil
import sys
import torch
import logging
import numpy as np
from PIL import Image
from threading import Thread
from contextlib import contextmanager

import folder_paths
import comfy.model_management as mm
import comfy.utils


@contextmanager
def vendored_transformers():
    vendor_path = os.path.join(os.path.dirname(__file__), "vendor")

    targets = ["transformers", "huggingface_hub"]
    
    saved_modules = {}
    for k in list(sys.modules):
        if any(k == t or k.startswith(t + ".") for t in targets):
            saved_modules[k] = sys.modules.pop(k)
    
    sys.path.insert(0, vendor_path)
    try:
        yield
    finally:
        for k in list(sys.modules):
            if any(k == t or k.startswith(t + ".") for t in targets):
                del sys.modules[k]
        
        sys.modules.update(saved_modules)
        
        if vendor_path in sys.path:
            sys.path.remove(vendor_path)

# Global variables
GLOBAL_MODEL = None
GLOBAL_PROCESSOR = None
GLOBAL_MODEL_NAME = None
GLOBAL_IS_4BIT = None
GLOBAL_ATTN_MODE = None 
GLOBAL_COMPILED = None

class Qwen35_VL_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (["0.8B", "2B", "4B", "9B"], {"default": "4B"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe in detail what is happening in this image/video."}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "disable_thinking": ("BOOLEAN", {"default": True}),
                "use_4bit": ("BOOLEAN", {"default": True}),
                "attention_mode": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_or_video": ("IMAGE", ), 
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "raw_text")
    FUNCTION = "generate"
    CATEGORY = "Qwen3.5"

    def generate(self, model_size, prompt, use_torch_compile, disable_thinking, use_4bit, attention_mode, max_new_tokens, temperature, 
                 top_p, num_beams, repetition_penalty, seed, frame_count, 
                 keep_model_loaded, image_or_video=None):
        
        global GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_MODEL_NAME, GLOBAL_IS_4BIT, GLOBAL_ATTN_MODE, GLOBAL_COMPILED

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        repo_id = f"Qwen/Qwen3.5-{model_size}"
        base_models_dir = os.path.join(folder_paths.models_dir, "qwen35")
        model_dir_name = repo_id.replace("/", "_")
        model_path = os.path.join(base_models_dir, model_dir_name)

        with vendored_transformers():
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
            from huggingface_hub import snapshot_download

            if not os.path.exists(model_path):
                logging.info(f"Model not found locally. Downloading {repo_id}...")
                snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)
                cache_dir = os.path.join(model_path, ".cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)

            device = mm.get_torch_device()

            output_text = ""

            should_reload = (
                GLOBAL_MODEL is None or 
                GLOBAL_MODEL_NAME != repo_id or 
                GLOBAL_IS_4BIT != use_4bit or 
                GLOBAL_ATTN_MODE != attention_mode or
                GLOBAL_COMPILED != use_torch_compile
            )

            if should_reload:
                if GLOBAL_MODEL is not None:
                    del GLOBAL_MODEL
                    del GLOBAL_PROCESSOR
                    mm.soft_empty_cache()
                    gc.collect()

                logging.info(f"Loading {repo_id} using vendored transformers 5.x...")
                
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if use_4bit else None
                GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(model_path)
                
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    device_map=device,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    attn_implementation=attention_mode
                )

                if use_torch_compile:
                    logging.info("Compiling model via torch.compile()...")
                    try:
                        model = torch.compile(model, backend="inductor", mode="default")
                    except Exception as e:
                        logging.error(f"Compilation failed: {e}")

                GLOBAL_MODEL = model
                GLOBAL_MODEL_NAME = repo_id
                GLOBAL_IS_4BIT = use_4bit
                GLOBAL_ATTN_MODE = attention_mode
                GLOBAL_COMPILED = use_torch_compile

            # 4. Input Processing
            content_list =[]
            processor_kwargs = {}

            if image_or_video is not None:
                batch_size = image_or_video.shape[0]
                pil_frames =[]
                
                if batch_size == 1:
                    img_np = (image_or_video[0].cpu().numpy() * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(img_np))
                    content_list.append({"type": "image", "image": pil_frames[0]})
                    processor_kwargs["images"] = pil_frames
                else:
                    indices = np.linspace(0, batch_size - 1, frame_count, dtype=int) if batch_size > frame_count else np.arange(batch_size)
                    for i in indices:
                        img_np = (image_or_video[i].cpu().numpy() * 255).astype(np.uint8)
                        pil_frames.append(Image.fromarray(img_np))
                    content_list.append({"type": "video", "video": pil_frames})
                    processor_kwargs["videos"] = pil_frames

            content_list.append({"type": "text", "text": prompt})
            messages =[{"role": "user", "content": content_list}]

            # 5. Prepare Inputs
            text_input = GLOBAL_PROCESSOR.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=not disable_thinking
            )

            inputs = GLOBAL_PROCESSOR(
                text=[text_input],
                **processor_kwargs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # 6. Streamer Setup
            streamer = TextIteratorStreamer(
                GLOBAL_PROCESSOR.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
                "streamer": streamer,
                **inputs
            }

            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            logging.info("Starting generation...")
            thread = Thread(target=GLOBAL_MODEL.generate, kwargs=gen_kwargs)
            thread.start()

            pbar = comfy.utils.ProgressBar(max_new_tokens)
            start_time = time.time()
            token_count = 0
            last_log_time = start_time
            
            for new_text in streamer:
                output_text += new_text
                token_count += 1
                
                current_time = time.time()
                if current_time - last_log_time > 1.0:
                    elapsed = current_time - start_time
                    speed = token_count / elapsed if elapsed > 0 else 0
                    logging.info(f"Tokens: {token_count} | Speed: {speed:.2f} t/s")
                    last_log_time = current_time
                
                pbar.update(1)

            thread.join()

            total_time = time.time() - start_time
            avg_speed = token_count / total_time if total_time > 0 else 0
            logging.info(f"Generation done! Avg Speed: {avg_speed:.2f} t/s.")

        # 9. Cleanup and Return
        filtered_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        filtered_text = filtered_text.replace('<think>', '').replace('</think>', '').strip()

        if not keep_model_loaded:
            logging.info("Unloading model...")
            del GLOBAL_MODEL
            del GLOBAL_PROCESSOR
            GLOBAL_MODEL = None
            GLOBAL_PROCESSOR = None
            GLOBAL_MODEL_NAME = None
            GLOBAL_IS_4BIT = None
            GLOBAL_ATTN_MODE = None
            GLOBAL_COMPILED = None
            mm.soft_empty_cache()
            gc.collect()

        return (filtered_text, output_text)

NODE_CLASS_MAPPINGS = {
    "Qwen35_VL_Node": Qwen35_VL_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35_VL_Node": "Qwen 3.5 VL (Vendored TF)"
}