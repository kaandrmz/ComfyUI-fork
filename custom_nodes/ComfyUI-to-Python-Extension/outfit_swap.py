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


from nodes import (
    UNETLoader,
    StyleModelLoader,
    ConditioningZeroOut,
    VAEDecode,
    KSampler,
    CLIPTextEncode,
    VAELoader,
    InpaintModelConditioning,
    ImageScaleBy,
    DualCLIPLoader,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    LoraLoaderModelOnly,
    CLIPVisionLoader,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        dualcliploader = DualCLIPLoader()
        dualcliploader_2 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_3 = cliptextencode.encode(
            text="", clip=get_value_at_index(dualcliploader_2, 0)
        )

        vaeloader = VAELoader()
        vaeloader_9 = vaeloader.load_vae(vae_name="ae.safetensors")

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_5 = fluxguidance.append(
            guidance=50, conditioning=get_value_at_index(cliptextencode_3, 0)
        )

        stylemodelloader = StyleModelLoader()
        stylemodelloader_82 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_83 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        loadimage = LoadImage()
        loadimage_17 = loadimage.load_image(
            image="jacket.png" # clothing
        )

        ailab_colorinput = NODE_CLASS_MAPPINGS["AILab_ColorInput"]()
        ailab_colorinput_67 = ailab_colorinput.get_color(preset="white", color="")

        rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
        rmbg_66 = rmbg.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Color",
            image=get_value_at_index(loadimage_17, 0),
            background_color=get_value_at_index(ailab_colorinput_67, 0),
        )

        createcontextwindow = NODE_CLASS_MAPPINGS["CreateContextWindow"]()
        createcontextwindow_13 = createcontextwindow.create_context_window(
            patch_mode="auto",
            patch_type="3:4",
            output_length=1536,
            pixel_buffer=64,
            input_image=get_value_at_index(rmbg_66, 0),
            input_mask=get_value_at_index(loadimage_17, 1),
        )

        reduxadvanced = NODE_CLASS_MAPPINGS["ReduxAdvanced"]()
        reduxadvanced_81 = reduxadvanced.apply_stylemodel(
            downsampling_factor=1,
            downsampling_function="area",
            mode="keep aspect ratio",
            weight=0.30000000000000004,
            autocrop_margin=0.1,
            conditioning=get_value_at_index(fluxguidance_5, 0),
            style_model=get_value_at_index(stylemodelloader_82, 0),
            clip_vision=get_value_at_index(clipvisionloader_83, 0),
            image=get_value_at_index(createcontextwindow_13, 0),
        )

        conditioningzeroout = ConditioningZeroOut()
        conditioningzeroout_6 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_3, 0)
        )

        loadimage_18 = loadimage.load_image(
            image="model.png" # model
        )

        comfyuivtonmaskloader = NODE_CLASS_MAPPINGS["ComfyUIVtonMaskLoader"]()
        comfyuivtonmaskloader_85 = comfyuivtonmaskloader.load_mask_model(device="cpu")

        comfyuivtonmaskgenerator = NODE_CLASS_MAPPINGS["ComfyUIVtonMaskGenerator"]()
        comfyuivtonmaskgenerator_84 = comfyuivtonmaskgenerator.generate_mask(
            category="Upper-body",
            offset_top=0,
            offset_bottom=0,
            offset_left=0,
            offset_right=0,
            mask_model=get_value_at_index(comfyuivtonmaskloader_85, 0),
            vton_image=get_value_at_index(loadimage_18, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_86 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(comfyuivtonmaskgenerator_84, 1)
        )

        createcontextwindow_14 = createcontextwindow.create_context_window(
            patch_mode="auto",
            patch_type="3:4",
            output_length=1536,
            pixel_buffer=128,
            input_image=get_value_at_index(loadimage_18, 0),
            input_mask=get_value_at_index(imagetomask_86, 0),
        )

        concatcontextwindow = NODE_CLASS_MAPPINGS["ConcatContextWindow"]()
        concatcontextwindow_15 = concatcontextwindow.concat_context_window(
            patch_mode="auto",
            patch_type="3:4",
            output_length=1536,
            patch_color="#FF0000",
            first_image=get_value_at_index(createcontextwindow_13, 0),
            second_image=get_value_at_index(createcontextwindow_14, 0),
            second_mask=get_value_at_index(createcontextwindow_14, 1),
        )

        inpaintmodelconditioning = InpaintModelConditioning()
        inpaintmodelconditioning_11 = inpaintmodelconditioning.encode(
            noise_mask=False,
            positive=get_value_at_index(reduxadvanced_81, 0),
            negative=get_value_at_index(conditioningzeroout_6, 0),
            vae=get_value_at_index(vaeloader_9, 0),
            pixels=get_value_at_index(concatcontextwindow_15, 0),
            mask=get_value_at_index(concatcontextwindow_15, 1),
        )

        unetloader = UNETLoader()
        unetloader_50 = unetloader.load_unet(
            unet_name="flux1-fill-dev.safetensors", weight_dtype="default"
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_12 = loraloadermodelonly.load_lora_model_only(
            lora_name="20250321_steps5000_pytorch_lora_weights.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_50, 0),
        )

        seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        seed_rgthree_69 = seed_rgthree.main(
            seed=random.randint(1, 2**64), unique_id=16515944297173639232
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop"]()
        imagescaleby = ImageScaleBy()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()

        for q in range(1):
            ksampler_7 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=25,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(loraloadermodelonly_12, 0),
                positive=get_value_at_index(inpaintmodelconditioning_11, 0),
                negative=get_value_at_index(inpaintmodelconditioning_11, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_11, 2),
            )
            print("Sampling complete")
            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_7, 0),
                vae=get_value_at_index(vaeloader_9, 0),
            )
            print("Decoding complete")
            imagecrop_30 = imagecrop.crop(
                width=get_value_at_index(concatcontextwindow_15, 2),
                height=get_value_at_index(concatcontextwindow_15, 3),
                x=get_value_at_index(concatcontextwindow_15, 4),
                y=get_value_at_index(concatcontextwindow_15, 5),
                image=get_value_at_index(vaedecode_8, 0),
            )
            print("Cropping complete")
            imagescaleby_32 = imagescaleby.upscale(
                upscale_method="lanczos",
                scale_by=get_value_at_index(createcontextwindow_14, 5),
                image=get_value_at_index(imagecrop_30, 0),
            )
            print("Upscaling complete")
            imagecompositemasked_34 = imagecompositemasked.composite(
                x=get_value_at_index(createcontextwindow_14, 3),
                y=get_value_at_index(createcontextwindow_14, 4),
                resize_source=False,
                destination=get_value_at_index(loadimage_18, 0),
                source=get_value_at_index(imagescaleby_32, 0),
                mask=get_value_at_index(createcontextwindow_14, 7),
            )
            print("Compositing complete")
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
            saveimage.save_images(
                images=get_value_at_index(imagecompositemasked_34, 0),
                filename_prefix="outfit_swap"
            )
            print("Image saved")


if __name__ == "__main__":
    main()
