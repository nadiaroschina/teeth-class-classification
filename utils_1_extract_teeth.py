import os
import json
import logging
import cv2
import numpy as np
from tqdm.auto import tqdm

from utils_1_detect_grid_lines import detect_grid_lines
from utils_1_has_tooth_clip import has_tooth_clip

def extract_teeth(input_folder, output_folder, clip_classifier, debug=True, start_tooth_id=0):

    logger = logging.getLogger(__name__)

    os.makedirs(output_folder, exist_ok=True)

    if debug:
        debug_dir = os.path.join(output_folder, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        edge_dir = os.path.join(debug_dir, "edges")
        grid_dir = os.path.join(debug_dir, "grid")
        proj_dir = os.path.join(debug_dir, "projection")
        os.makedirs(edge_dir, exist_ok=True)
        os.makedirs(grid_dir, exist_ok=True)
        os.makedirs(proj_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ]

    logger.info(f"Found {len(files)} images")

    tooth_id = start_tooth_id

    for file in tqdm(files):

        try:

            logger.info(f"Processing {file}")
            path = os.path.join(input_folder, file)
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Could not read {file}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray,(9,9),1.5)

            binary = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                5
            )

            if debug:
                cv2.imwrite(
                    os.path.join(edge_dir,f"{file}_binary.jpg"),
                    binary
                )

            h,w = binary.shape

            # ---- crop ROI to avoid edge noise ----
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)

            roi = binary[
                margin_y:h-margin_y,
                margin_x:w-margin_x
            ]

            logger.debug(
                f"{file}: ROI margins x={margin_x} y={margin_y}"
            )

            # detect lines
            vertical = detect_grid_lines(
                roi,
                axis="vertical",
                expected_lines=4,
                offset=margin_x,
                debug_path=(
                    os.path.join(proj_dir,f"{file}_vertical.png")
                    if debug else None
                )
            )

            horizontal = detect_grid_lines(
                roi,
                axis="horizontal",
                expected_lines=4,
                offset=margin_y,
                debug_path=(
                    os.path.join(proj_dir, f"{file}_horizontal.png")
                    if debug else None
                )
            )

            if len(vertical) not in (2, 3, 4) or len(horizontal) not in (2, 3, 4):
                logger.warning(
                    f"{file}: incorrect line count "
                    f"H={len(horizontal)} V={len(vertical)}"
                )
                continue

            vertical = np.concatenate(([0],vertical,[w]))
            horizontal = np.concatenate(([0],horizontal,[h]))

            # ---- debug grid overlay ----
            grid_img = img.copy()

            for x in vertical:
                cv2.line(grid_img,(x,0),(x,h),(107, 37, 227),30)

            for y in horizontal:
                cv2.line(grid_img,(0,y),(w,y),(107, 37, 227),30)

            if debug:
                cv2.imwrite(
                    os.path.join(grid_dir,f"{file}_grid.jpg"),
                    grid_img
                )

            if debug:
                accepted_dir = os.path.join(debug_dir, "accepted")
                rejected_dir = os.path.join(debug_dir, "rejected")
                classification_info_dir = os.path.join(debug_dir, "classification_info")
                os.makedirs(accepted_dir, exist_ok=True)
                os.makedirs(rejected_dir, exist_ok=True)
                os.makedirs(classification_info_dir, exist_ok=True)

            for i in range(len(horizontal)-1):
                for j in range(len(vertical)-1):

                    y1, y2 = horizontal[i], horizontal[i+1]
                    x1, x2 = vertical[j], vertical[j+1]

                    tooth = img[y1:y2, x1:x2]

                    keep, info = has_tooth_clip(tooth, clip_classifier)
                    
                    if debug:
                        with open(os.path.join(classification_info_dir, f"{os.path.splitext(file)[0]}_{i}_{j}.json"), "w") as f:
                            json.dump(info, f)

                    dbg_name = f"{os.path.splitext(file)[0]}_{i}_{j}.jpg"
                    if keep:
                        if debug:
                            cv2.imwrite(os.path.join(accepted_dir, dbg_name), tooth)
                        name = f"{tooth_id:05d}.jpg"
                        cv2.imwrite(os.path.join(output_folder, name), tooth)
                        tooth_id += 1
                    else:
                        if debug:
                            cv2.imwrite(os.path.join(rejected_dir, dbg_name), tooth)


        except Exception as e:
            logger.error(f"{file} failed: {e}")

    return tooth_id
