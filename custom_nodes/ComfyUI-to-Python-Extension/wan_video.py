import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS, LoadImage


def main():
    import_custom_nodes()
    with torch.inference_mode():
        loadwanvideot5textencoder = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        loadwanvideot5textencoder_11 = loadwanvideot5textencoder.loadmodel(
            model_name="umt5-xxl-enc-bf16.safetensors",
            precision="bf16",
            load_device="offload_device",
            quantization="fp8_e4m3fn",
        )

        wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        wanvideovaeloader_38 = wanvideovaeloader.loadmodel(
            model_name="wan_2.1_vae.safetensors", precision="bf16"
        )

        wanvideoenhanceavideo = NODE_CLASS_MAPPINGS["WanVideoEnhanceAVideo"]()
        wanvideoenhanceavideo_55 = wanvideoenhanceavideo.setargs(
            weight=2, start_percent=0, end_percent=1
        )

        loadimage = LoadImage()
        loadimage_433 = loadimage.load_image(image="example.png")

        jwinteger = NODE_CLASS_MAPPINGS["JWInteger"]()
        jwinteger_436 = jwinteger.execute(value=832)

        mxslider = NODE_CLASS_MAPPINGS["mxSlider"]()
        mxslider_439 = mxslider.main(Xi=81, Xf=81, isfloatX=0)

        mxslider_440 = mxslider.main(Xi=8, Xf=8, isfloatX=0)

        wanvideoloraselectmulti = NODE_CLASS_MAPPINGS["WanVideoLoraSelectMulti"]()
        wanvideoloraselectmulti_470 = wanvideoloraselectmulti.getlorapath(
            lora_0="Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
            strength_0=1,
            lora_1="none",
            strength_1=1,
            lora_2="none",
            strength_2=1,
            lora_3="none",
            strength_3=1,
            lora_4="none",
            strength_4=1,
            low_mem_load=False,
            merge_loras=False,
        )

        wanvideoloraselectmulti_471 = wanvideoloraselectmulti.getlorapath(
            lora_0="Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
            strength_0=1,
            lora_1="none",
            strength_1=1,
            lora_2="none",
            strength_2=1,
            lora_3="none",
            strength_3=1,
            lora_4="none",
            strength_4=1,
            low_mem_load=False,
            merge_loras=False,
        )

        wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        wanvideoblockswap_474 = wanvideoblockswap.setargs(
            blocks_to_swap=10,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=0,
            block_swap_debug=False,
        )

        wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS[
            "WanVideoTorchCompileSettings"
        ]()
        wanvideotorchcompilesettings_475 = wanvideotorchcompilesettings.set_args(
            backend="inductor",
            fullgraph=False,
            mode="default",
            dynamic=False,
            dynamo_cache_size_limit=64,
            compile_transformer_blocks_only=True,
            dynamo_recompile_limit=128,
        )

        cr_prompt_text = NODE_CLASS_MAPPINGS["CR Prompt Text"]()
        cr_prompt_text_482 = cr_prompt_text.get_value(prompt="prompt")

        wanvideotextencode = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        divide_dvb = NODE_CLASS_MAPPINGS["Divide [DVB]"]()
        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        wanvideosetblockswap = NODE_CLASS_MAPPINGS["WanVideoSetBlockSwap"]()
        wanvideosetloras = NODE_CLASS_MAPPINGS["WanVideoSetLoRAs"]()
        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        wanvideoimagetovideoencode = NODE_CLASS_MAPPINGS["WanVideoImageToVideoEncode"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        playsoundpysssss = NODE_CLASS_MAPPINGS["PlaySound|pysssss"]()

        for q in range(1):
            wanvideotextencode_16 = wanvideotextencode.process(
                positive_prompt=get_value_at_index(cr_prompt_text_482, 0),
                negative_prompt="still image, slowmotion, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, ",
                force_offload=True,
                use_disk_cache=True,
                device="gpu",
                t5=get_value_at_index(loadwanvideot5textencoder_11, 0),
            )

            divide_dvb_443 = divide_dvb.result(
                divisor=2, float=0, int=get_value_at_index(mxslider_440, 0)
            )

            wanvideomodelloader_469 = wanvideomodelloader.loadmodel(
                model="Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors",
                base_precision="fp16_fast",
                quantization="fp8_e4m3fn_scaled",
                load_device="offload_device",
                attention_mode="sageattn",
                compile_args=get_value_at_index(wanvideotorchcompilesettings_475, 0),
            )

            wanvideosetblockswap_467 = wanvideosetblockswap.loadmodel(
                model=get_value_at_index(wanvideomodelloader_469, 0),
                block_swap_args=get_value_at_index(wanvideoblockswap_474, 0),
            )

            wanvideosetloras_468 = wanvideosetloras.setlora(
                model=get_value_at_index(wanvideosetblockswap_467, 0),
                lora=get_value_at_index(wanvideoloraselectmulti_471, 0),
            )

            imageresize_435 = imageresize.execute(
                width=get_value_at_index(jwinteger_436, 0),
                height=get_value_at_index(jwinteger_436, 0),
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=16,
                image=get_value_at_index(loadimage_433, 0),
            )

            wanvideoimagetovideoencode_432 = wanvideoimagetovideoencode.process(
                width=get_value_at_index(imageresize_435, 1),
                height=get_value_at_index(imageresize_435, 2),
                num_frames=get_value_at_index(mxslider_439, 0),
                noise_aug_strength=0.030000000000000006,
                start_latent_strength=1,
                end_latent_strength=1,
                force_offload=True,
                fun_or_fl2v_model=False,
                tiled_vae=False,
                vae=get_value_at_index(wanvideovaeloader_38, 0),
                start_image=get_value_at_index(imageresize_435, 0),
            )

            wanvideomodelloader_465 = wanvideomodelloader.loadmodel(
                model="Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
                base_precision="fp16_fast",
                quantization="fp8_e4m3fn_scaled",
                load_device="offload_device",
                attention_mode="sageattn",
                compile_args=get_value_at_index(wanvideotorchcompilesettings_475, 0),
            )

            wanvideosetblockswap_464 = wanvideosetblockswap.loadmodel(
                model=get_value_at_index(wanvideomodelloader_465, 0),
                block_swap_args=get_value_at_index(wanvideoblockswap_474, 0),
            )

            wanvideosetloras_466 = wanvideosetloras.setlora(
                model=get_value_at_index(wanvideosetblockswap_464, 0),
                lora=get_value_at_index(wanvideoloraselectmulti_470, 0),
            )

            wanvideosampler_308 = wanvideosampler.process(
                steps=get_value_at_index(mxslider_440, 0),
                cfg=1.0000000000000002,
                shift=8.000000000000002,
                seed=random.randint(1, 2**64),
                force_offload=True,
                scheduler="dpm++_sde",
                riflex_freq_index=0,
                denoise_strength=1,
                batched_cfg=False,
                rope_function="comfy",
                start_step=0,
                end_step=get_value_at_index(divide_dvb_443, 1),
                add_noise_to_samples=False,
                model=get_value_at_index(wanvideosetloras_466, 0),
                image_embeds=get_value_at_index(wanvideoimagetovideoencode_432, 0),
                text_embeds=get_value_at_index(wanvideotextencode_16, 0),
                feta_args=get_value_at_index(wanvideoenhanceavideo_55, 0),
            )

            wanvideosampler_392 = wanvideosampler.process(
                steps=get_value_at_index(mxslider_440, 0),
                cfg=1.0000000000000002,
                shift=8.000000000000002,
                seed=random.randint(1, 2**64),
                force_offload=True,
                scheduler="dpm++_sde",
                riflex_freq_index=0,
                denoise_strength=1,
                batched_cfg=False,
                rope_function="comfy",
                start_step=get_value_at_index(divide_dvb_443, 1),
                end_step=-1,
                add_noise_to_samples=False,
                model=get_value_at_index(wanvideosetloras_468, 0),
                image_embeds=get_value_at_index(wanvideoimagetovideoencode_432, 0),
                text_embeds=get_value_at_index(wanvideotextencode_16, 0),
                samples=get_value_at_index(wanvideosampler_308, 0),
                feta_args=get_value_at_index(wanvideoenhanceavideo_55, 0),
            )

            wanvideodecode_28 = wanvideodecode.decode(
                enable_vae_tiling=False,
                tile_x=512,
                tile_y=512,
                tile_stride_x=256,
                tile_stride_y=256,
                normalization="default",
                vae=get_value_at_index(wanvideovaeloader_38, 0),
                samples=get_value_at_index(wanvideosampler_392, 1),
            )

            vhs_videocombine_30 = vhs_videocombine.combine_video(
                frame_rate=24,
                loop_count=0,
                filename_prefix="WAN 2.2 I2V",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=8,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(wanvideodecode_28, 0),
                unique_id=9059575849642698232,
            )

            playsoundpysssss_219 = playsoundpysssss.nop(
                mode="always",
                volume=0.5,
                file="notify.mp3",
                any=get_value_at_index(vhs_videocombine_30, 0),
            )


if __name__ == "__main__":
    main()
