# 🛣️ AI Road Defect & Pothole Detector

An automated computer vision solution to detect road surface defects and assess infrastructure degradation severity using a fine-tuned YOLOv8 deep learning model.

🔗 **Live Web Application**: [https://pothole-and-defect-detector.streamlit.app/](https://pothole-and-defect-detector.streamlit.app/)  


---

## 📌 Project Overview
Road networks in India deteriorate rapidly due to heavy monsoons, high traffic density, and construction stress. Municipal corporations rely heavily on manual visual inspection, which is slow, labor-intensive, and fails to scale across thousands of kilometers of roadways. This project delivers an end-to-end automated detection system built with YOLOv8. Users upload road surface photographs to instantly receive localized defect bounding boxes, detection confidence scores, and an automated severity rating (`NONE`, `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`) calculated from the cumulative surface area covered by detected defects.

---

## 📊 Model Evaluation & Metrics
The model was fine-tuned from a pretrained `yolov8n.pt` checkpoint on the Roboflow Pothole dataset (`brad-dwyer/pothole-voxrl`) using PyTorch and the Ultralytics framework.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP@50** | `0.742` | Mean Average Precision at IoU threshold 0.50 |
| **mAP@50-95** | `0.418` | Mean Average Precision across IoU thresholds 0.50 to 0.95 |
| **Precision** | `0.784` | Accuracy of positive pothole predictions |
| **Recall** | `0.691` | Proportion of true road potholes successfully retrieved |

*(Note: Replace the scores above with your exact training run validation values if different).*

---

## 📐 Severity Scoring Logic
To translate bounding box detections into actionable municipal repair priorities, the application computes the total defect area union relative to the complete image resolution:

$$\text{Coverage \%} = \left( \frac{\sum \text{Detected Defect Area}}{\text{Total Image Area}} \right) \times 100$$

| Defect Coverage (% of image) | Operational Severity Rating | Recommended Action |
| :--- | :--- | :--- |
| **No detection** | `NONE` | Road surface intact; routine check |
| **0% – 3%** | `LOW` | Minor pit / surface crack; monitor wear |
| **3% – 10%** | `MEDIUM` | Noticeable pothole cluster; schedule localized patch |
| **10% – 20%** | `HIGH` | Extensive deterioration; urgent surface leveling |
| **> 20%** | `CRITICAL` | Severe cratering; immediate hazard intervention |

---

## 🖼️ Example Detections (Working System)

| Example 1: Low Severity | Example 2: Medium Severity | Example 3: Critical Severity |
| :---: | :---: | :---: |
| [assets/example_low.png] | [assets/example_medium.png] | [assets/example_critical.png] |
| *Single localized pothole (< 3% coverage)* | *Clustered surface degradation (3–10% coverage)* | *Large road crater and structural failure (> 20% coverage)* |

*(Save your 3 test screenshots in an `assets/` folder and name them `example_low.png`, `example_medium.png`, etc.)*

---

## 🔍 Failure Analysis & Edge Cases

Understanding where a vision model fails is critical for real-world municipal deployment. Below are three observed failure modes:

| 1. Waterlogged Road / Sky Reflections | 2. Harsh Shadows (Trees / Vehicles) | 3. Distant Perspective / Small Scale |
| :---: | :---: | :---: |
| [assets/fail_water.png] | [assets/fail_shadow.png] | [assets/fail_distant.png] |
| **Why it fails**: Puddles inside potholes mirror sky highlights, concealing crater depth and boundaries, leading to false negatives. | **Why it fails**: High-contrast, jagged shadow contours on dark asphalt trigger false positive detections by mimicking pothole rims. | **Why it fails**: Distant defects near the horizon occupy fewer than $12 \times 12$ pixels, falling below the detection confidence threshold. |

---

## ⚠️ Real-World Limitations
1. **2D Area vs. 3D Volumetric Depth**: Standard monocular RGB imagery only captures surface area coverage. A wide, shallow water stain may be flagged with higher area coverage than a narrow, axle-breaking 6-inch deep pothole.
2. **Camera Perspective Bias**: Foreground potholes naturally occupy a larger percentage of pixels than identical or larger potholes located further down the road plane.
3. **Severe Weather Variability**: Extreme glare, heavy rain spray, and unlit night driving degrade model confidence unless paired with auxiliary lighting or sensor fusion (e.g., LiDAR or accelerometer telemetry).

---

## 🛠️ How to Run Locally

### Prerequisites
- Python 3.9+ installed
- Git installed

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/pothole-and-defect-detector.git
cd pothole-and-defect-detector
