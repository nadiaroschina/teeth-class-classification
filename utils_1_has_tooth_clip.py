from PIL import Image
import cv2


def inner_crop(cell, border_frac=0.12):
    h, w = cell.shape[:2]
    dx = int(w * border_frac)
    dy = int(h * border_frac)

    if w - 2 * dx < 20 or h - 2 * dy < 20:
        return cell

    return cell[dy:h-dy, dx:w-dx]


POS_LABELS = [
    "a close-up photo of a tooth",
    "a dental photograph",
    "an image of a tooth"
]

NEG_LABELS = [
    # "an empty cardboard square",
    "a piece of cardboard with a red checkered blanket",
    "an empty piece of cardboard"
]

def has_tooth_clip(cell, clip_classifier):
    roi = inner_crop(cell, border_frac=0.12)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    labels = POS_LABELS + NEG_LABELS
    result = clip_classifier(pil, candidate_labels=labels)

    pos_score = max(x["score"] for x in result if x["label"] in POS_LABELS)
    neg_score = max(x["score"] for x in result if x["label"] in NEG_LABELS)

    keep = (pos_score > neg_score) and (pos_score > 0.25)

    return keep, {
        "pos_score": pos_score,
        "neg_score": neg_score,
        "raw": result
    }