import os
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import normalize

from CodeFormer.basicsr.utils import imwrite, img2tensor, tensor2img
from CodeFormer.basicsr.utils.download_util import load_file_from_url
from CodeFormer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from CodeFormer.facelib.utils.misc import is_gray
from CodeFormer.basicsr.archs.rrdbnet_arch import RRDBNet
from CodeFormer.basicsr.archs.codeformer_arch import CodeFormer
from CodeFormer.basicsr.utils.realesrgan_utils import RealESRGANer
from CodeFormer.basicsr.utils.registry import ARCH_REGISTRY

def find_directory(root: Path, target_name: str) -> Path:
    # Check if root directory itself matches
    if root.name == target_name and root.is_dir():
        return root

    # Otherwise search recursively inside the root directory
    for path in root.rglob(target_name):
        if path.is_dir():
            return path
    return None

video_face_swapper_dir = find_directory(Path.cwd(), "VideoFaceSwapper")

models = {
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "dir": f"{video_face_swapper_dir / 'CodeFormer' / 'weights' / 'CodeFormer'}",
        "filename": "codeformer.pth",
    },
    "detection": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "dir": f"{video_face_swapper_dir / 'CodeFormer' / 'weights' / 'CodeFormer.facelib'}",
        "filename": "detection_Resnet50_Final.pth",
    },
    "parsing": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        "dir": f"{video_face_swapper_dir / 'CodeFormer' / 'weights' / 'CodeFormer.facelib'}",
        "filename": "parsing_parsenet.pth",
    },
    "realesrgan": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        "dir": f"{video_face_swapper_dir / 'CodeFormer' / 'weights' / 'realesrgan'}",
        "filename": "RealESRGAN_x2plus.pth",
    },
    # buffalo_l will be downloaded automatically
    "faceanalyser": {
        "buffalo_l": {
            "dir": f"{video_face_swapper_dir / 'inswapper' / 'checkpoints'}",
            "1k3d68": {
                "url": "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/1k3d68.onnx",
                "filename": "1k3d68.onnx",
            },
            "2d106det": {
                "url": "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/2d106det.onnx",
                "filename": "2d106det.onnx",
            },
            "det_10g": {
                "url": "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
                "filename": "det_10g.onnx",
            },
            "genderage.onnx": {
                "url": "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/genderage.onnx",
                "filename": "genderage.onnx",
            },
            "w600k_r50": {
                "url": "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
                "filename": "w600k_r50.onnx",
            },
        },
    },
    # inswapper_128
    "faceswapmodel": {
        "url": "https://huggingface.co/Rookiehan/facefusion/resolve/d8a1f4006b0e1e3318ab883a5858eb0037d77c98/inswapper_128.onnx",
        "dir": f"{video_face_swapper_dir / 'inswapper' / 'checkpoints' / 'models'}",
        "filename": "inswapper_128.onnx",
    },
}


def check_ckpts():
    for model_entry_name, model_info in models.items():
        if model_entry_name == "faceanalyser":
            continue
        model_url = model_info["url"]
        model_dir = model_info["dir"]
        model_name = model_info["filename"]  # with extension
        full_model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(full_model_path):
            load_file_from_url(
                url=model_url, model_dir=model_dir, progress=True, file_name=model_name
            )


# set enhancer with RealESRGAN
def set_realesrgan() -> RealESRGANer:
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    realesrgan_model_path = os.path.join(
        models["realesrgan"]["dir"], models["realesrgan"]["filename"]
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path=realesrgan_model_path,  # ../CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


def face_restoration(
    img: Image.Image,
    background_enhance: bool,
    face_upsample: bool,
    upscale: int,
    codeformer_fidelity: int,
    upsampler: RealESRGANer,
    codeformer_net: CodeFormer,
    device: torch.device,
) -> Image.Image:
    """Run a single prediction on the model"""
    try:  # global try
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        background_enhance = (
            background_enhance if background_enhance is not None else True
        )
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        upscale = int(upscale)  # convert type to int
        if upscale > 4:  # avoid memory exceeded due to too large upscale
            upscale = 4
        if (
            upscale > 2 and max(img.shape[:2]) > 1000
        ):  # avoid memory exceeded due to too large img resolution
            upscale = 2
        if (
            max(img.shape[:2]) > 1500
        ):  # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print("Global exception", error)
        return None, None
