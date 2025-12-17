import os
import random
from pathlib import Path
import cv2
import yaml

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------- Helpers ----------
def list_images(folder: Path):
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    imgs.sort()
    return imgs

def ensure_structure(out_root: Path):
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

def yolo_label_path(split_dir: Path, image_name: str):
    return split_dir / (Path(image_name).stem + ".txt")

def copy_image_to_dataset(src: Path, out_root: Path):
    dst = out_root / "images" / "train" / src.name
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    return dst

def save_yolo_txt(label_file: Path, boxes, img_w, img_h):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        cx_n = cx / img_w
        cy_n = cy / img_h
        bw_n = bw / img_w
        bh_n = bh / img_h

        lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def load_existing_boxes(label_file: Path, img_w, img_h):
    if not label_file.exists():
        return []
    txt = label_file.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    boxes = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        cx *= img_w
        cy *= img_h
        bw *= img_w
        bh *= img_h
        x1 = int(round(cx - bw / 2.0))
        y1 = int(round(cy - bh / 2.0))
        x2 = int(round(cx + bw / 2.0))
        y2 = int(round(cy + bh / 2.0))
        boxes.append((cls, x1, y1, x2, y2))
    return boxes

def write_data_yaml(out_root: Path, class_names):
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

def count_labeled(out_root: Path):
    labels_dir = out_root / "labels" / "train"
    if not labels_dir.exists():
        return 0
    c = 0
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix == ".txt" and p.read_text(encoding="utf-8").strip() != "":
            c += 1
    return c

def ensure_label_exists_for_image(out_root: Path, img_name: str):
    lbl = out_root / "labels" / "train" / (Path(img_name).stem + ".txt")
    if not lbl.exists():
        lbl.write_text("", encoding="utf-8")

def make_train_val_split(out_root: Path, val_ratio: float, seed: int = 42):
    """
    Move a random subset of images/labels from train -> val.
    Keeps pairing by filename stem.
    """
    if val_ratio <= 0:
        return

    train_img_dir = out_root / "images" / "train"
    train_lbl_dir = out_root / "labels" / "train"
    val_img_dir = out_root / "images" / "val"
    val_lbl_dir = out_root / "labels" / "val"

    imgs = [p for p in train_img_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    imgs.sort()

    if not imgs:
        print("[SPLIT] No train images found, skipping split.")
        return

    # Ensure every image has a label file (empty = negative example is OK)
    for img in imgs:
        ensure_label_exists_for_image(out_root, img.name)

    n_total = len(imgs)
    n_val = int(round(n_total * val_ratio))
    n_val = max(1, n_val) if n_total >= 2 else 0  # avoid val with 0 when possible

    random.seed(seed)
    val_imgs = set(random.sample(imgs, n_val)) if n_val > 0 else set()

    moved = 0
    for img_path in imgs:
        if img_path not in val_imgs:
            continue

        lbl_path = train_lbl_dir / (img_path.stem + ".txt")

        dst_img = val_img_dir / img_path.name
        dst_lbl = val_lbl_dir / lbl_path.name

        # MOVE (not copy) to avoid duplicates between train and val
        img_path.replace(dst_img)
        if lbl_path.exists():
            lbl_path.replace(dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

        moved += 1

    print(f"[SPLIT] Moved {moved}/{n_total} items to val (val_ratio={val_ratio}).")

# ---------- Main Annotator ----------
class Annotator:
    def __init__(self, in_dir: Path, out_root: Path, class_names, val_ratio: float, seed: int):
        self.in_dir = in_dir
        self.out_root = out_root
        self.class_names = class_names
        self.class_id = 0
        self.val_ratio = val_ratio
        self.seed = seed

        ensure_structure(out_root)
        write_data_yaml(out_root, class_names)

        self.images = list_images(in_dir)
        if not self.images:
            raise SystemExit(f"No images found in: {in_dir}")

        self.idx = 0
        self.drawing = False
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.boxes = []
        self.win = "YOLO Annotator"

    def _load_current(self):
        src = self.images[self.idx]
        self.dataset_img_path = copy_image_to_dataset(src, self.out_root)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {src}")
        self.img = img
        self.h, self.w = img.shape[:2]

        self.label_file = self.out_root / "labels" / "train" / (src.stem + ".txt")
        self.boxes = load_existing_boxes(self.label_file, self.w, self.h)

    def _draw_overlay(self):
        vis = self.img.copy()

        for (cls, x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{cls}:{self.class_names[cls]}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.drawing:
            cv2.rectangle(vis, (self.x1, self.y1), (self.x2, self.y2), (0, 200, 255), 2)

        labeled = count_labeled(self.out_root)
        total = len(self.images)
        fname = self.images[self.idx].name
        status = f"[{self.idx+1}/{total}] labeled:{labeled} class:{self.class_id}({self.class_names[self.class_id]}) file:{fname}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = x, y
            self.x2, self.y2 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = x, y

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.x2, self.y2 = x, y
            if abs(self.x2 - self.x1) >= 3 and abs(self.y2 - self.y1) >= 3:
                self.boxes.append((self.class_id, self.x1, self.y1, self.x2, self.y2))

    def _save_current(self):
        save_yolo_txt(self.label_file, self.boxes, self.w, self.h)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._mouse)

        self._load_current()

        print("Keys: [n]=next(save)  [p]=prev(save)  [s]=save  [d]=delete last box  "
              "[c]=change class  [x]=clear boxes  [q]=quit (then split)")

        while True:
            vis = self._draw_overlay()
            cv2.imshow(self.win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                self._save_current()
                break
            elif key == ord('s'):
                self._save_current()
            elif key == ord('n'):
                self._save_current()
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self._load_current()
            elif key == ord('p'):
                self._save_current()
                if self.idx > 0:
                    self.idx -= 1
                    self._load_current()
            elif key == ord('d'):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord('x'):
                self.boxes = []
            elif key == ord('c'):
                self.class_id = (self.class_id + 1) % len(self.class_names)

        cv2.destroyAllWindows()

        # Do the split once at the end
        make_train_val_split(self.out_root, val_ratio=self.val_ratio, seed=self.seed)
        print("[DONE] Dataset ready:", self.out_root)

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("YOLO Dataset Annotator")
    parser.add_argument("--images", type=Path, required=True, help="Input folder of images to label")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--classes", type=str, default="bright_spot", help="Comma-separated class names")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction to move to val after quitting (e.g. 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    args = parser.parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        raise SystemExit("No classes provided")

    app = Annotator(args.images, args.out, class_names, val_ratio=args.val_ratio, seed=args.seed)
    app.run()
