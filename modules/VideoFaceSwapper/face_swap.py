import os

import numpy as np
from PIL import Image
from typing import List, Union
from tqdm import tqdm

from inswapper.swapper import process
from inswapper.restoration import models
from inswapper.restoration import check_ckpts
from inswapper.swapper import get_face_swap_model, get_face_analyser
from CodeFormer.basicsr.archs.codeformer_arch import CodeFormer


def perform_face_swap(
    images: List[np.ndarray],
    inswapper_source_image: Union[Image.Image, np.ndarray],
    inswapper_source_image_indicies: str,
    inswapper_target_image_indicies: str,
) -> List[np.ndarray]:
    swapped_images = []

    swap_model_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "inswapper/checkpoints/models/inswapper_128.onnx",
    )
    faceanalyser_model_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "inswapper/checkpoints"
    )
    
    codeformer_path: str = ""

    for model_entry_name, model_info in models.items():
        if model_entry_name == "faceswapmodel":
            model_dir = model_info["dir"]
            model_name = model_info["filename"]  # with extension
            full_model_path = os.path.join(model_dir, model_name)
            if os.path.exists(full_model_path):
                swap_model_path = full_model_path
        if model_entry_name == "faceanalyser":
            model_dir = model_info["buffalo_l"]["dir"]
            if os.path.exists(model_dir):
                faceanalyser_model_dir = model_dir
        if model_entry_name == "codeformer":
            model_dir = model_info["dir"]
            model_name = model_info["filename"]  # with extension
            full_model_path = os.path.join(model_dir, model_name)
            #print(full_model_path + " " + str(os.path.exists(full_model_path)))
            if os.path.exists(full_model_path):
                codeformer_path = full_model_path

    # make sure the ckpts downloaded successfully
    check_ckpts()

    from inswapper.restoration import (
        face_restoration,
        set_realesrgan,
        torch,
        ARCH_REGISTRY,
        cv2,
    )

    # https://huggingface.co/spaces/sczhou/CodeFormer
    upsampler = set_realesrgan()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    codeformer_net: CodeFormer = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt_path = codeformer_path
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    # load machine default available providers
    # providers = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # load face_analyser
    face_analyser = get_face_analyser(faceanalyser_model_dir, providers)

    # load face_swapper
    face_swapper = get_face_swap_model(swap_model_path)

    source_face_image = (
        inswapper_source_image
        if isinstance(inswapper_source_image, Image.Image)
        else Image.fromarray(inswapper_source_image)
    )
    print(f"Inswapper: Source indicies: {inswapper_source_image_indicies}")
    print(f"Inswapper: Target indicies: {inswapper_target_image_indicies}")

    for image in tqdm(images):
        target_image = (
            image if isinstance(image, Image.Image) else Image.fromarray(image)
        )
        swapped_face_image = process(
            source_face_image,
            target_image,
            inswapper_source_image_indicies,
            inswapper_target_image_indicies,
            face_analyser,
            face_swapper,
        )

        swapped_image = cv2.cvtColor(np.array(swapped_face_image), cv2.COLOR_RGB2BGR)
        swapped_image_rest = face_restoration(
            swapped_image, True, True, 1, 0.5, upsampler, codeformer_net, device
        )

        swapped_images.append(swapped_image_rest)

    return swapped_images
