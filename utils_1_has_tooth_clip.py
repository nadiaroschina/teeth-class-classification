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
    "an empty piece of cardboard",
    "a piece of cardboard with pink gum",
    "a piece of cardboard with pinkish play-doh"
]

def _cell_to_pil(cell):
    """Convert BGR numpy cell to PIL RGB image after inner crop."""
    roi = inner_crop(cell, border_frac=0.12)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _parse_scores(result):
    """Extract pos/neg scores and keep decision from a single CLIP result list."""
    pos_score = max(x["score"] for x in result if x["label"] in POS_LABELS)
    neg_score = max(x["score"] for x in result if x["label"] in NEG_LABELS)
    keep = (pos_score > neg_score) and (pos_score > 0.25)
    return keep, {"pos_score": pos_score, "neg_score": neg_score, "raw": result}

def has_tooth_clip(cell, clip_classifier):
    """Classify a single cell. Returns (keep: bool, info: dict)."""
    pil = _cell_to_pil(cell)
    labels = POS_LABELS + NEG_LABELS
    result = clip_classifier(pil, candidate_labels=labels)
    return _parse_scores(result)

def has_tooth_clip_batch(cells, clip_classifier):
    """
    Classify a batch of cells in a single CLIP call.

    Parameters
    ----------
    cells : list of np.ndarray
        BGR images (grid cells from one group photo).
    clip_classifier : transformers.Pipeline
        Zero-shot image classification pipeline.

    Returns
    -------
    list of (keep: bool, info: dict)
        One entry per input cell, same order.
    """
    if not cells:
        return []

    labels = POS_LABELS + NEG_LABELS
    pil_images = [_cell_to_pil(c) for c in cells]

    # The HuggingFace pipeline accepts a list of images when called with
    # candidate_labels; it returns a list of result-lists, one per image.
    batch_results = clip_classifier(pil_images, candidate_labels=labels)

    return [_parse_scores(result) for result in batch_results]