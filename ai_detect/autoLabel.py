"""
Hệ thống tự động gán nhãn ảnh dựa trên YOLO
Hỗ trợ 3 loại ảnh: ảnh băng rộng, ảng băng hẹp, ảnh tín hiệu WiFi
"""

import os
import json
import ctypes
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import yaml


def _configure_cuda_runtime():
    """Expose CUDA shared libraries installed by pip before torch loads."""
    site_packages = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
    cuda_lib_dir = site_packages / "nvidia" / "cu13" / "lib"
    nvrtc_builtins = cuda_lib_dir / "libnvrtc-builtins.so.13.0"

    if not nvrtc_builtins.exists():
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_lib_path = str(cuda_lib_dir)
    if cuda_lib_path not in current_ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cuda_lib_path}:{current_ld_path}" if current_ld_path else cuda_lib_path
        )

    try:
        ctypes.CDLL(str(nvrtc_builtins), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_configure_cuda_runtime()

try:
    import numpy as np
    import cv2
except ImportError as exc:
    message = str(exc)
    if "numpy.core.multiarray failed to import" in message or "_ARRAY_API" in message:
        raise ImportError() from exc
    raise

from ultralytics import YOLO

class AutoLabelingSystem:
    def __init__(self, model_path: str = "yolov8m.pt", conf_threshold: float = 0.5):
        """
        Khởi tạo hệ thống tự động gán nhãn
        
        Args:
            model_path: Đường dẫn mô hình YOLO
            conf_threshold: Ngưỡng độ tin cậy tối thiểu
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.detected_objects = {}

    def _find_images(self, images_dir: str) -> List[Path]:
        image_dir = Path(images_dir)
        image_extensions = {'.jpg', '.jpeg', '.png'}
        return sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
    
    def _load_class_names(self, classes_file: str) -> List[str]:
        path = Path(classes_file)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file classes: {classes_file}")
        with open(path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError(f"File classes rỗng: {classes_file}")
        return names

    def train_on_labeled_data(self,
                             labeled_images_dir: str = "data/rotate2",
                             labeled_annotations_dir: str = "data/labels_yolo",
                             classes_file: str = "data/classes.txt",
                             epochs: int = 50,
                             batch_size: int = 16,
                             imgsz: int = 640,
                             project: str = "runs/detect",
                             name: str = "train"):
        print("🔄 Chuẩn bị dữ liệu training...")

        class_names = self._load_class_names(classes_file)

        # Tạo cấu trúc thư mục cho YOLO (xóa dữ liệu cũ để tránh lẫn với lần chạy trước)
        base_dir = Path("yolo_dataset")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(exist_ok=True)

        for split in ['train', 'val']:
            (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Sao chép dữ liệu (80% train, 20% val)
        image_files = [
            img_file
            for img_file in self._find_images(labeled_images_dir)
            if (Path(labeled_annotations_dir) / f"{img_file.stem}.txt").exists()
        ]
        if not image_files:
            raise FileNotFoundError(
                "Không tìm thấy cặp ảnh/label hợp lệ. "
                "Kiểm tra lại labeled_images_dir và labeled_annotations_dir."
            )

        np.random.shuffle(image_files)

        if len(image_files) < 5:
            train_files = image_files
            val_files = []
        else:
            split_idx = int(0.8 * len(image_files))
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
        
        # Copy files
        for img_file in train_files:
            txt_file = Path(labeled_annotations_dir) / f"{img_file.stem}.txt"
            if txt_file.exists():
                shutil.copy(img_file, base_dir / 'train' / 'images' / img_file.name)
                shutil.copy(txt_file, base_dir / 'train' / 'labels' / txt_file.name)
        
        for img_file in val_files:
            txt_file = Path(labeled_annotations_dir) / f"{img_file.stem}.txt"
            if txt_file.exists():
                shutil.copy(img_file, base_dir / 'val' / 'images' / img_file.name)
                shutil.copy(txt_file, base_dir / 'val' / 'labels' / txt_file.name)
        
        # Tạo file data.yaml
        data_yaml = {
            'path': str(base_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images' if val_files else 'train/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(base_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"✅ Dữ liệu training: {len(train_files)} ảnh")
        print(f"✅ Dữ liệu validation: {len(val_files)} ảnh")
        
        # Fine-tune model
        print("\n🚀 Bắt đầu fine-tuning YOLO...")
        results = self.model.train(
            data=str(base_dir / 'data.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=0,  # GPU device, thay đổi nếu cần
            patience=20,
            save=True,
            verbose=True,
            project=project,
            name=name
        )

        save_dir = getattr(results, "save_dir", None)
        print("✅ Hoàn thành training!")
        if save_dir:
            print(f"Mô hình đã lưu tại: {Path(save_dir) / 'weights' / 'best.pt'}")
        return results
    
    def predict_and_label(self, 
                         unlabeled_images_dir: str,
                         output_dir: str,
                         conf_threshold: float = None,
                         visualize_dir: str = None) -> Dict:
        """
        Dự đoán nhãn cho ảnh chưa được gán nhãn
        
        Args:
            unlabeled_images_dir: Thư mục chứa ảnh cần gán nhãn
            output_dir: Thư mục lưu kết quả
            conf_threshold: Ngưỡng độ tin cậy
            visualize_dir: Thư mục lưu ảnh có bounding box để kiểm tra trực quan
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if visualize_dir:
            Path(visualize_dir).mkdir(parents=True, exist_ok=True)
        
        if not Path(unlabeled_images_dir).exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {unlabeled_images_dir}")

        image_files = self._find_images(unlabeled_images_dir)
        results_summary = {
            'total_images': len(image_files),
            'labeled_images': 0,
            'details': []
        }
        
        print(f"\n📸 Xử lý {len(image_files)} ảnh...")
        
        for idx, img_path in enumerate(image_files, 1):
            # Dự đoán vật thể
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                verbose=False
            )
            
            # Xử lý kết quả
            detections = []
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            detection = {
                                'class': int(box.cls[0]),
                                'confidence': float(box.conf[0]),
                                'bbox': box.xyxy[0].tolist()
                            }
                            detections.append(detection)
            
            # Lưu nhãn (định dạng YOLO)
            if detections:
                label_path = Path(output_dir) / f"{img_path.stem}.txt"
                self._save_yolo_labels(str(img_path), detections, str(label_path))
                results_summary['labeled_images'] += 1

            if visualize_dir:
                output_viz_path = Path(visualize_dir) / img_path.name
                self._save_visualized_image(str(img_path), detections, str(output_viz_path))
            
            results_summary['details'].append({
                'image': img_path.name,
                'detections': len(detections),
                'confidence_avg': np.mean([d['confidence'] for d in detections]) if detections else 0
            })
            
            if idx % 10 == 0:
                print(f"  ✅ Đã xử lý {idx}/{len(image_files)} ảnh")
        
        print(f"\n✅ Gán nhãn hoàn thành!")
        self._print_summary(results_summary)
        
        return results_summary
    
    def _save_yolo_labels(self, image_path: str, detections: List[Dict], label_path: str):
        """
        Lưu nhãn theo định dạng YOLO
        """
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        with open(label_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                # Chuyển đổi sang định dạng YOLO (center_x, center_y, width, height)
                center_x = (x1 + x2) / 2 / width
                center_y = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                f.write(f"{det['class']} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    def _get_class_name(self, class_id: int) -> str:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _save_visualized_image(self, image_path: str, detections: List[Dict], output_path: str):
        """
        Lưu ảnh có bounding box, tên nhãn và confidence để kiểm tra trực quan.
        """
        img = cv2.imread(image_path)
        if img is None:
            return

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        for det in detections:
            class_id = det['class']
            confidence = det['confidence']
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = colors[class_id % len(colors)]
            label = f"{self._get_class_name(class_id)} {confidence:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size
            y_text = max(y1, text_h + baseline + 4)
            cv2.rectangle(
                img,
                (x1, y_text - text_h - baseline - 4),
                (x1 + text_w + 6, y_text),
                color,
                -1
            )
            cv2.putText(
                img,
                label,
                (x1 + 3, y_text - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imwrite(output_path, img)
    
    def visualize_predictions(self, 
                             images_dir: str,
                             labels_dir: str,
                             output_viz_dir: str,
                             num_samples: int = 10):
        """
        Trực quan hóa kết quả dự đoán
        """
        Path(output_viz_dir).mkdir(parents=True, exist_ok=True)
        
        image_files = self._find_images(images_dir)[:num_samples]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        
        print(f"\n🎨 Trực quan hóa {len(image_files)} ảnh...")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            
            if label_path.exists():
                height, width = img.shape[:2]
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1]) * width
                            center_y = float(parts[2]) * height
                            box_width = float(parts[3]) * width
                            box_height = float(parts[4]) * height
                            
                            x1 = int(center_x - box_width / 2)
                            y1 = int(center_y - box_height / 2)
                            x2 = int(center_x + box_width / 2)
                            y2 = int(center_y + box_height / 2)
                            
                            color = colors[class_id % len(colors)]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, self._get_class_name(class_id), (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = Path(output_viz_dir) / img_path.name
            cv2.imwrite(str(output_path), img)
        
        print(f"✅ Đã lưu hình ảnh trực quan hóa tới: {output_viz_dir}")
    
    def _print_summary(self, summary: Dict):
        """In tóm tắt kết quả"""
        print("\n" + "="*50)
        print("📊 TÓM TẮT KẾT QUẢ")
        print("="*50)
        print(f"Tổng số ảnh: {summary['total_images']}")
        print(f"Ảnh được gán nhãn: {summary['labeled_images']}")
        if summary['total_images'] == 0:
            print("Tỷ lệ: 0.0%")
            print("Không tìm thấy ảnh đầu vào để xử lý.")
        else:
            print(f"Tỷ lệ: {summary['labeled_images']/summary['total_images']*100:.1f}%")
        print("="*50 + "\n")


def main():
    system = AutoLabelingSystem(model_path="yolov8m.pt", conf_threshold=0.5)

    # 1. Fine-tune trên dữ liệu mới (data/rotate2 + data/labels_yolo, 10 lớp theo data/classes.txt)
    print("=" * 60)
    print("BƯỚC 1: FINE-TUNE YOLO TRÊN DỮ LIỆU MỚI")
    print("=" * 60)

    system.train_on_labeled_data(
        labeled_images_dir="data/rotate2",
        labeled_annotations_dir="data/labels_yolo",
        classes_file="data/classes.txt",
        epochs=30,
        batch_size=16
    )
    
    # 2. Tự động gán nhãn cho ảnh mới
    print("\n" + "=" * 60)
    print("BƯỚC 2: TỰ ĐỘNG GÁN NHÃN CHO ẢNH MỚI")
    print("=" * 60)
    
    results = system.predict_and_label(
        unlabeled_images_dir="unlabled_data/images",
        output_dir="output_labels",
        conf_threshold=0.5,
        visualize_dir="visualized_results"
    )
    
    # 3. Trực quan hóa kết quả
    print("\n" + "=" * 60)
    print("BƯỚC 3: TRỰC QUAN HÓA KẾT QUẢ")
    print("=" * 60)
    
    system.visualize_predictions(
        images_dir="data/rotate2",
        labels_dir="output_labels",
        output_viz_dir="visualized_results",
        num_samples=20
    )
    
    # Lưu kết quả
    with open("labeling_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Toàn bộ quá trình hoàn thành!")


if __name__ == "__main__":
    main()
