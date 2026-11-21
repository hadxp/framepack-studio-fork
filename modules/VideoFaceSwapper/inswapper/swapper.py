"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""

import cv2
import copy
import insightface
import numpy as np
from PIL import Image
from typing import List
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper


def get_face_swap_model(model_path: str) -> INSwapper:
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(
    faceanalyser_model_path_dir: str, providers, det_size=(320, 320)
) -> FaceAnalysis:
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root=faceanalyser_model_path_dir, providers=providers
    )
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser: FaceAnalysis, frame: np.ndarray) -> List[np.ndarray]:
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser: FaceAnalysis, frame: np.ndarray) -> List[np.ndarray]:
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(
    face_swapper: INSwapper,
    source_faces: List[np.ndarray],
    target_faces: List[np.ndarray],
    source_index: int,
    target_index: int,
    temp_frame: np.ndarray,
) -> np.ndarray:
    """
    paste source_face on target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(
    source_img: Image.Image,
    target_img: Image.Image,
    source_indexes: str,
    target_indexes: str,
    face_analyser: FaceAnalysis,
    face_swapper: INSwapper,
) -> Image.Image:
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in the target image
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = 1

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(
                face_analyser, cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            )
            num_source_faces = len(source_faces)
            # print(f"Source faces: {num_source_faces}")
            # print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    # print("Replacing all faces in target image with the same face from the source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    # print("There are fewer faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    # print("There are fewer faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    # print("Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame,
                    )
            else:
                # print("Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ",".join(
                        map(lambda x: str(x), range(num_source_faces))
                    )

                if target_indexes == "-1":
                    target_indexes = ",".join(
                        map(lambda x: str(x), range(num_target_faces))
                    )

                source_indexes = source_indexes.split(",")
                target_indexes = target_indexes.split(",")
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception(
                        "Number of source indexes is greater than the number of faces in the source image"
                    )

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception(
                        "Number of target indexes is greater than the number of faces in the target image"
                    )

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces - 1:
                            raise ValueError(
                                f"Source index {source_index} is higher than the number of faces in the source image"
                            )

                        if target_index > num_target_faces - 1:
                            raise ValueError(
                                f"Target index {target_index} is higher than the number of faces in the target image"
                            )

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame,
                        )
        else:
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        print("No target faces found!")

    # entire_mask_image = np.zeros_like(np.array(target_img))
    # result_image = apply_face_mask(result, target_img, target_faces[target_index], entire_mask_image)
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image
