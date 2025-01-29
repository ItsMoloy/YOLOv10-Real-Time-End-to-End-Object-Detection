# YOLOv10: Real-Time End-to-End Object Detection

YOLOv10 is the latest advancement in real-time object detection, eliminating non-maximum suppression (NMS) and optimizing model architecture to achieve superior performance with reduced computational overhead. This repository provides implementation, training, and inference examples using the **Ultralytics YOLO framework**.

## 🚀 Features
- **NMS-Free Training**: Uses consistent dual assignments to improve accuracy and efficiency.
- **Optimized Architecture**: Incorporates CSPNet backbone, PAN neck, and One-to-Many/One-to-One heads.
- **Multiple Model Variants**:
  - YOLOv10-N: Nano version for low-resource environments.
  - YOLOv10-S: Small, balancing speed and accuracy.
  - YOLOv10-M, B, L, X: Larger models for high accuracy.
- **High Performance**: Outperforms previous YOLO versions and other SOTA detectors.
- **Flexible Deployment**: Supports ONNX, TensorRT, OpenVINO, and TF models.

## 📂 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ItsMoloy/YOLOv10-Real-Time-End-to-End-Object-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python matplotlib
   ```
3. Verify installation:
   ```bash
   python -c "import ultralytics; ultralytics.checks()"
   ```

## 🎯 Inference on Images
```python
from ultralytics import YOLO

# Load YOLOv10 model
model = YOLO("yolov10n.pt")

# Run inference on an image
results = model("image.jpg")
results[0].show()
```

## 📊 Training on a Custom Dataset
```python
# Load YOLOv10 model
model = YOLO("yolov10n.yaml")

# Train the model (replace dataset.yaml with your dataset file)
model.train(data="dataset.yaml", epochs=50, imgsz=640)
```

## 📦 Exporting the Model
```python
# Export trained model for deployment
model.export(format="onnx")  # Formats: torchscript, onnx, openvino, coreml, etc.
```

## 🏆 Performance Comparison
| Model   | Input Size | APval | FLOPs (G) | Latency (ms) |
|---------|-----------|-------|-----------|--------------|
| YOLOv10-N | 640 | 38.5 | 6.7 | 1.84 |
| YOLOv10-S | 640 | 46.3 | 21.6 | 2.49 |
| YOLOv10-M | 640 | 51.1 | 59.1 | 4.74 |
| YOLOv10-B | 640 | 52.5 | 92.0 | 5.74 |
| YOLOv10-L | 640 | 53.2 | 120.3 | 7.28 |
| YOLOv10-X | 640 | 54.4 | 160.4 | 10.70 |

## 📜 Citation
If you use YOLOv10 in your research, please cite:
```bibtex
@article{THU-MIGyolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Ao Wang, Hui Chen, Lihao Liu, et al.},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024},
  institution={Tsinghua University},
  license = {AGPL-3.0}
}
```

## 📌 Contributing
Contributions are welcome! To contribute:
1. Fork the repo.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a PR.

## 🔗 Resources
- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)
- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv10 Documentation](https://docs.ultralytics.com/)

---
Developed by **Moloy Banerjee**. 🚀
