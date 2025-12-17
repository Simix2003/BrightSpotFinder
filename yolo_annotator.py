import os
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

def yolo_label_path(out_root: Path, image_name: str):
    return out_root / "labels" / "train" / (Path(image_name).stem + ".txt")

def copy_image_to_dataset(src: Path, out_root: Path):
    dst = out_root / "images" / "train" / src.name
    if not dst.exists():
        # copy without metadata to keep it simple and fast
        dst.write_bytes(src.read_bytes())
    return dst

def save_yolo_txt(label_file: Path, boxes, img_w, img_h):
    # boxes: list of (class_id, x1, y1, x2, y2)
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        # YOLO normalized
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
    boxes = []
    txt = label_file.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        # denormalize
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
        "val": "images/train",  # you can change later
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

def count_labeled(out_root: Path):
    labels_dir = out_root / "labels" / "train"
    if not labels_dir.exists():
        return 0
    return sum(1 for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt" and p.read_text().strip() != "")

# ---------- Main Annotator ----------
class Annotator:
    def __init__(self, in_dir: Path, out_root: Path, class_names):
        self.in_dir = in_dir
        self.out_root = out_root
        self.class_names = class_names
        self.class_id = 0

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
        # copy image into dataset so the dataset is ready immediately
        self.dataset_img_path = copy_image_to_dataset(src, self.out_root)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {src}")
        self.img = img
        self.h, self.w = img.shape[:2]

        # load existing labels if any
        self.label_file = yolo_label_path(self.out_root, src.name)
        self.boxes = load_existing_boxes(self.label_file, self.w, self.h)

    def _draw_overlay(self):
        vis = self.img.copy()

        # draw existing boxes
        for (cls, x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{cls}:{self.class_names[cls]}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw current drag box
        if self.drawing:
            cv2.rectangle(vis, (self.x1, self.y1), (self.x2, self.y2), (0, 200, 255), 2)

        # status bar
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
            # add box if big enough
            if abs(self.x2 - self.x1) >= 3 and abs(self.y2 - self.y1) >= 3:
                self.boxes.append((self.class_id, self.x1, self.y1, self.x2, self.y2))

    def _save_current(self):
        save_yolo_txt(self.label_file, self.boxes, self.w, self.h)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._mouse)

        self._load_current()

        help_text = (
            "Keys: [n]=next(save)  [p]=prev(save)  [s]=save  [d]=delete last box  "
            "[c]=change class  [x]=clear boxes  [q]=quit"
        )
        print(help_text)

        while True:
            vis = self._draw_overlay()
            cv2.imshow(self.win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                # save before exit (optional)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("YOLO Dataset Annotator")
    parser.add_argument("--images", type=Path, required=True, help="Input folder of images to label")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--classes", type=str, default="bright_spot", help="Comma-separated class names")
    args = parser.parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        raise SystemExit("No classes provided")

    app = Annotator(args.images, args.out, class_names)
    app.run()
