"""Try several versions of one crop on the OCR reader and compare them.

Usage (on the OCR machine, in the same Python that runs the remover):
    python ocr_variants.py "baby pregnancy- 1308 303 (Hindi) 2.mp4"

It prints one table per frame and saves every crop to ./ocr_variants_out,
so you can look at what the reader was given.
"""

import os
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np

VIDEO = sys.argv[1]
TRUTH = "तुम्हारे यहाँ बच्चा होगा!"
BOX = (107, 605, 384, 503)          # xmin, xmax, ymin, ymax, from the job log
TIMES = [13.8, 14.2, 14.8, 15.5, 16.5, 17.5]
MODELS = ["devanagari_PP-OCRv5_mobile_rec", "devanagari_PP-OCRv5_server_rec"]
OUT = "ocr_variants_out"
DEVICE = os.environ.get("OCR_DEVICE", "gpu")


def frame_at(seconds):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def crop_of(img):
    # Same crop as read_boxes: 25% more room above and below.
    xmin, xmax, ymin, ymax = BOX
    pad_y = (ymax - ymin) // 4
    return img[max(0, ymin - pad_y):min(img.shape[0], ymax + pad_y), xmin:xmax]


def variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Keep only the white inside of the letters, drop the coloured outline,
    # and make it black text on white.
    white = np.all(crop > 200, axis=2).astype(np.uint8) * 255
    white_core = 255 - white
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    found = {
        "original": crop,
        "gray": gray,
        "inverted": 255 - crop,
        "otsu": otsu,
        "otsu_inverted": 255 - otsu,
        "clahe": cv2.createCLAHE(clipLimit=2.0).apply(gray),
        "white_core": white_core,
        "white_core_thick": cv2.erode(white_core, np.ones((3, 3), np.uint8)),
        "scaled_x2": big,
        "padded": padded,
    }
    # The reader wants three channels.
    return {name: cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) if v.ndim == 2 else v
            for name, v in found.items()}


def load_reader(name):
    import paddle
    paddle.disable_signal_handler()
    from paddleocr import TextRecognition
    return TextRecognition(model_name=name, device=DEVICE, enable_hpi=False)


def detect_boxes(img):
    """What the detector finds near the line, to see how it cuts it."""
    from paddleocr import TextDetection
    det = TextDetection(device=DEVICE, enable_hpi=False, box_thresh=0.6, thresh=0.45)
    boxes = []
    for poly in det.predict(img)[0]["dt_polys"]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        box = (int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys)))
        if box[3] > 360 and box[2] < 530:
            boxes.append(box)
    return boxes


def main():
    os.makedirs(OUT, exist_ok=True)
    readers = {}
    for name in MODELS:
        try:
            readers[name] = load_reader(name)
        except Exception as error:      # the model may not exist
            print(f"model {name}: cannot load ({error.__class__.__name__}: {error})")

    for t in TIMES:
        img = frame_at(t)
        if img is None:
            print(f"\nt={t}: no frame")
            continue
        cv2.imwrite(f"{OUT}/frame_{t}.png", img)
        print(f"\n=== t={t}s  truth={TRUTH!r}")
        try:
            print(f"detector boxes near the line: {detect_boxes(img)}")
        except Exception as error:
            print(f"detector failed: {error}")
        crops = variants(crop_of(img))
        for name, v in crops.items():
            cv2.imwrite(f"{OUT}/t{t}_{name}.png", v)
        for model, reader in readers.items():
            print(f"-- {model}")
            results = reader.predict(list(crops.values()), batch_size=len(crops))
            rows = []
            for name, res in zip(crops, results):
                text = res["rec_text"] or ""
                score = float(res["rec_score"] or 0.0)
                match = SequenceMatcher(None, text, TRUTH, autojunk=False).ratio()
                rows.append((match, score, name, text))
            for match, score, name, text in sorted(rows, reverse=True):
                print(f"   match={match:.2f} score={score:.2f} {name:18s} {text!r}")


if __name__ == "__main__":
    main()
