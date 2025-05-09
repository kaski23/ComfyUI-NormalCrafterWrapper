# ComfyUI/custom_nodes/ComfyUI-NormalCrafter/normal_crafter_nodes.py

import torch
import numpy as np
from PIL import Image
import os

import comfy.model_management
import comfy.utils
import folder_paths # ComfyUI's way to get model paths

# Try to import required components and provide guidance if they are missing
try:
    from diffusers import AutoencoderKLTemporalDecoder
    from huggingface_hub import snapshot_download
except ImportError:
    print("ComfyUI-NormalCrafter: Missing essential libraries. Please ensure 'diffusers', 'transformers', 'accelerate', and 'huggingface_hub' are installed.")
    # Consider raising an error or making the node unusable if critical dependencies are missing.

try:
    # This assumes the 'normalcrafter' directory is correctly placed within ComfyUI-NormalCrafter
    from .normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
    from .normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
except ImportError as e:
    print(f"ComfyUI-NormalCrafter: Error importing NormalCrafter components: {e}. "
          "Ensure the 'normalcrafter' directory is correctly placed inside 'ComfyUI-NormalCrafter'.")

# Global variable to cache the pipeline
NORMALCRAFTER_PIPE = None
CURRENT_PIPE_CONFIG = {} # Stores the config {"device": "cuda"/"cpu", "dtype": "float16"/"float32"}

# Define model paths and repo ID
NORMALCRAFTER_REPO_ID = "Yanrui95/NormalCrafter"
NORMALCRAFTER_MODELS_SUBDIR_NAME = "normalcrafter_models" # Subdirectory in ComfyUI/models/
SVD_XT_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


class NormalCrafterNode:
    def __init__(self):
        self.pipe = None # Instance variable for the pipeline

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_res_dimension": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "window_size": ("INT", {"default": 14, "min": 2, "max": 64}),
                "time_step_size": ("INT", {"default": 10, "min": 1, "max": 64}),
                "decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 64}),
                "offload_pipe_to_cpu_on_finish": ("BOOLEAN", {"default": True}), # <<< NEW INPUT
            },
            "optional": {
                "pipe_override": ("NORMALCRAFTER_PIPE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Video/NormalCrafter"

    def _get_local_nc_model_path(self):
        base_models_dir = os.path.join(folder_paths.models_dir, NORMALCRAFTER_MODELS_SUBDIR_NAME)
        os.makedirs(base_models_dir, exist_ok=True)
        model_name_folder = NORMALCRAFTER_REPO_ID.split('/')[-1]
        specific_model_path = os.path.join(base_models_dir, model_name_folder)
        return specific_model_path

    def _download_normalcrafter_model_if_needed(self):
        local_nc_path = self._get_local_nc_model_path()
        unet_config_path = os.path.join(local_nc_path, "unet", "config.json")

        if not os.path.exists(unet_config_path):
            print(f"ComfyUI-NormalCrafter: Downloading {NORMALCRAFTER_REPO_ID} model to {local_nc_path}...")
            try:
                snapshot_download(
                    repo_id=NORMALCRAFTER_REPO_ID,
                    local_dir=local_nc_path,
                    local_dir_use_symlinks=False,
                )
                print(f"ComfyUI-NormalCrafter: Model {NORMALCRAFTER_REPO_ID} download complete.")
            except Exception as e:
                print(f"ComfyUI-NormalCrafter: Failed to download model {NORMALCRAFTER_REPO_ID}: {e}")
                raise
        else:
            # print(f"ComfyUI-NormalCrafter: Model {NORMALCRAFTER_REPO_ID} found at {local_nc_path}.")
            pass
        return local_nc_path

    def _load_pipeline(self, device_str="cuda", dtype_str="float16"):
        global NORMALCRAFTER_PIPE, CURRENT_PIPE_CONFIG

        # This is the configuration requested for the *current* run
        requested_config = {
            "device": device_str,
            "dtype": dtype_str,
        }

        # If pipe exists and its last known config matches what's requested for *this* run
        if NORMALCRAFTER_PIPE is not None and CURRENT_PIPE_CONFIG == requested_config:
            target_device_for_this_run = comfy.model_management.get_torch_device() if device_str == "cuda" else torch.device("cpu")
            # Safety check: ensure it's actually on the target_device_for_this_run.
            # This handles cases where ComfyUI might have moved it, or if our offload didn't update CURRENT_PIPE_CONFIG.
            # However, with proper CURRENT_PIPE_CONFIG updates, this `if` might rarely trigger if CURRENT_PIPE_CONFIG already reflects "cpu".
            if NORMALCRAFTER_PIPE.device != target_device_for_this_run:
                 print(f"ComfyUI-NormalCrafter: Moving existing pipeline from {NORMALCRAFTER_PIPE.device} to {target_device_for_this_run}.")
                 NORMALCRAFTER_PIPE.to(target_device_for_this_run)
            self.pipe = NORMALCRAFTER_PIPE
            return self.pipe

        # If pipe doesn't exist, or its last known config (CURRENT_PIPE_CONFIG)
        # is different from what's requested for this run (e.g., it was on CPU, now GPU is requested),
        # then we proceed to load/reload.

        print("ComfyUI-NormalCrafter: Loading NormalCrafter pipeline...")
        local_nc_model_path = self._download_normalcrafter_model_if_needed()

        if dtype_str == "float16": torch_dtype = torch.float16
        elif dtype_str == "bf16": torch_dtype = torch.bfloat16
        else: torch_dtype = torch.float32

        print(f"ComfyUI-NormalCrafter: Loading UNet from subfolder 'unet' in {local_nc_model_path}")
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            local_nc_model_path, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch_dtype # Initially load to CPU
        )
        print(f"ComfyUI-NormalCrafter: Loading VAE from subfolder 'vae' in {local_nc_model_path}")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            local_nc_model_path, subfolder="vae", low_cpu_mem_usage=True, torch_dtype=torch_dtype # Initially load to CPU
        )

        svd_xt_variant = "fp16" if torch_dtype == torch.float16 else None
        print(f"ComfyUI-NormalCrafter: Loading base SVD pipeline ({SVD_XT_REPO_ID}) and integrating custom UNet/VAE.")
        pipe = NormalCrafterPipeline.from_pretrained(
            SVD_XT_REPO_ID, unet=unet, vae=vae, torch_dtype=torch_dtype, variant=svd_xt_variant,
        )

        target_device_for_this_run = comfy.model_management.get_torch_device() if device_str == "cuda" else torch.device("cpu")
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("ComfyUI-NormalCrafter: Xformers memory efficient attention enabled.")
        except Exception as e:
            print(f"ComfyUI-NormalCrafter: Xformers not available or failed to enable: {e}. Proceeding without it.")
            # Consider pipe.enable_attention_slicing() as a fallback if VRAM is an issue during processing

        pipe.to(target_device_for_this_run) # Move the newly loaded pipe to the target device
        NORMALCRAFTER_PIPE = pipe
        CURRENT_PIPE_CONFIG = requested_config # Update global config to reflect the state of the loaded pipe
        self.pipe = NORMALCRAFTER_PIPE
        print(f"ComfyUI-NormalCrafter: Pipeline loaded to {target_device_for_this_run} with dtype {torch_dtype}.")
        return self.pipe

    def tensor_to_pil_list(self, images_tensor: torch.Tensor) -> list:
        pil_images = []
        for i in range(images_tensor.shape[0]):
            img_np = (images_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        return pil_images

    def resize_pil_images(self, pil_images: list, max_res_dim: int) -> list:
        resized_images = []
        if not pil_images: return []
        for img in pil_images:
            original_width, original_height = img.size
            if max(original_height, original_width) > max_res_dim:
                scale = max_res_dim / max(original_height, original_width)
                target_height = round(original_height * scale)
                target_width = round(original_width * scale)
            else:
                target_height = original_height
                target_width = original_width
            resized_images.append(img.resize((target_width, target_height), Image.LANCZOS))
        return resized_images

    def process(self, images: torch.Tensor, seed: int, max_res_dimension: int,
                window_size: int, time_step_size: int, decode_chunk_size: int,
                offload_pipe_to_cpu_on_finish: bool, # <<< NEW PARAMETER
                pipe_override=None):

        default_fps_for_time_ids = 7
        default_motion_bucket_id = 127
        default_noise_aug_strength = 0.0

        if pipe_override is not None:
            self.pipe = pipe_override
            print("ComfyUI-NormalCrafter: Using provided pipe_override.")
        else:
            current_comfy_device = comfy.model_management.get_torch_device()
            device_str = "cuda" if current_comfy_device.type == 'cuda' else "cpu"
            # Determine dtype based on device (float16 for CUDA, float32 for CPU)
            dtype_str = "float16" if device_str == "cuda" and comfy.model_management.should_use_fp16() else "float32"
            self._load_pipeline(device_str, dtype_str)

        if self.pipe is None:
            raise RuntimeError("ComfyUI-NormalCrafter: Pipeline could not be loaded.")

        # Ensure the pipe instance self.pipe is on the correct device for processing *before* using it
        # This is important if self.pipe came from the global cache and might have been on CPU
        processing_device = comfy.model_management.get_torch_device()
        if self.pipe.device != processing_device:
            print(f"ComfyUI-NormalCrafter: Moving self.pipe from {self.pipe.device} to {processing_device} for processing.")
            self.pipe.to(processing_device)


        pil_frames = self.tensor_to_pil_list(images)
        if not pil_frames: return (torch.zeros_like(images),)

        resized_pil_frames = self.resize_pil_images(pil_frames, max_res_dimension)
        
        num_actual_frames = len(resized_pil_frames)
        effective_frames_for_pipeline = list(resized_pil_frames) 

        if num_actual_frames == 0:
             print("ComfyUI-NormalCrafter: Warning - No frames to process after resizing.")
             return (torch.zeros_like(images),)

        if num_actual_frames < window_size:
            print(f"ComfyUI-NormalCrafter: Number of frames ({num_actual_frames}) is less than window_size ({window_size}). Padding...")
            padding_needed = window_size - num_actual_frames
            last_frame_to_duplicate = resized_pil_frames[-1] 
            for _ in range(padding_needed):
                effective_frames_for_pipeline.append(last_frame_to_duplicate)
        
        generator_device = self.pipe.device # Should be processing_device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        print(f"ComfyUI-NormalCrafter: Processing {len(effective_frames_for_pipeline)} frames (effective) with seed {seed}. Original: {num_actual_frames}.")
        print(f"ComfyUI-NormalCrafter: Using (internal defaults) fps={default_fps_for_time_ids}, motion_id={default_motion_bucket_id}, noise_aug={default_noise_aug_strength}")

        pbar = comfy.utils.ProgressBar(len(effective_frames_for_pipeline)) # This pbar seems not used by the pipe.

        with torch.inference_mode():
            output_frames_np = self.pipe( # self.pipe should be on processing_device here
                images=effective_frames_for_pipeline,
                decode_chunk_size=decode_chunk_size,
                time_step_size=time_step_size,
                window_size=window_size,
                fps=default_fps_for_time_ids,
                motion_bucket_id=default_motion_bucket_id,
                noise_aug_strength=default_noise_aug_strength,
                generator=generator
                # SVD pipeline has its own progress bar, no need to pass pbar here
            ).frames[0]

        if len(effective_frames_for_pipeline) > num_actual_frames:
            output_frames_np = output_frames_np[:num_actual_frames, :, :, :]
        
        output_normals_0_1 = (output_frames_np.clip(-1., 1.) * 0.5) + 0.5
        output_tensor = torch.from_numpy(output_normals_0_1).float()

        # --- Explicit Offload After Processing ---
        # Only offload the globally managed pipe (NORMALCRAFTER_PIPE), not an overridden one.
        # And only if no pipe_override was used for this run.
        if pipe_override is None and \
           offload_pipe_to_cpu_on_finish and \
           NORMALCRAFTER_PIPE is not None and \
           NORMALCRAFTER_PIPE.device.type == 'cuda': # Check if it's on CUDA before moving
            print("ComfyUI-NormalCrafter: Offloading globally cached pipeline to CPU after processing.")
            try:
                NORMALCRAFTER_PIPE.to("cpu")
                global CURRENT_PIPE_CONFIG # Make sure to get the global
                if "device" in CURRENT_PIPE_CONFIG:
                    CURRENT_PIPE_CONFIG["device"] = "cpu" # Update its state to "cpu"
                else: # Should not happen if config is always set
                    CURRENT_PIPE_CONFIG = {"device": "cpu", "dtype": CURRENT_PIPE_CONFIG.get("dtype", "float32")}

                comfy.model_management.soft_empty_cache() # Ask ComfyUI to try and free VRAM
            except Exception as e:
                print(f"ComfyUI-NormalCrafter: Error offloading pipeline to CPU: {e}")
        
        print(f"ComfyUI-NormalCrafter: Processing complete. Output tensor shape: {output_tensor.shape}")
        return (output_tensor,)
