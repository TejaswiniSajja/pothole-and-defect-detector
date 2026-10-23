from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# Configuration & Model Path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

# Checks both the models/ directory and the runs/ training path
CANDIDATE_PATHS = [
    ROOT / "models" / "best.pt",
    ROOT / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "pothole" / "pothole_img416_fixed" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "pothole_model" / "weights" / "best.pt",
]

MODEL_PATH = None
for p in CANDIDATE_PATHS:
    if p.exists():
        MODEL_PATH = p
        break

st.set_page_config(
    page_title="AI Pothole & Defect Detector",
    page_icon="🚧",
    layout="wide",
)

# --------------------------------------------------
# Page styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #777777;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🚧 AI Pothole & Defect Detector</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a road image to detect potholes and estimate their severity.'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

if MODEL_PATH is None:
    st.error("Trained model weights (best.pt) not found.")
    st.info("Please place your `best.pt` file inside the `models/` directory or your runs path.")
    st.stop()

model = load_model(MODEL_PATH)

# --------------------------------------------------
# Severity calculation
# --------------------------------------------------

def calculate_severity(area_percentage):
    """
    Severity thresholds based on assignment rubric.
    """
    if area_percentage == 0:
        return "NONE"
    elif area_percentage <= 3:
        return "LOW"
    elif area_percentage <= 10:
        return "MEDIUM"
    elif area_percentage <= 20:
        return "HIGH"
    else:
        return "CRITICAL"

def calculate_union_area(boxes, image_width, image_height):
    """
    Estimate the combined area covered by detection boxes.
    Overlapping regions are counted only once.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return int(mask.sum())

# --------------------------------------------------
# Sidebar Settings
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Higher values keep only more confident detections.",
)

# --------------------------------------------------
# File Upload & Display Logic
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"],
)

st.caption("Supported formats: JPG, JPEG, PNG, WEBP.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    image_height, image_width = image_array.shape[:2]

    # Clear old results if a new file is uploaded
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_image_key") != file_id:
        st.session_state["current_image_key"] = file_id
        st.session_state.pop("annotated_rgb", None)
        st.session_state.pop("summary_data", None)
        st.session_state.pop("detection_df", None)

    # 1. Side-by-Side Columns: Left = Original, Right = Detected
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📷 Original Image")
        st.image(image, caption="Uploaded Road Image", use_container_width=True)
        run_detection = st.button(
            "🚧 Run Road Damage Detection",
            type="primary",
            use_container_width=True
        )

    with col_right:
        st.subheader("🎯 Detection Results")
        # If detection hasn't run yet, show a clean placeholder
        if "annotated_rgb" not in st.session_state and not run_detection:
            st.info("👈 Click **Run Road Damage Detection** on the left to see annotated detections here.")
        elif "annotated_rgb" in st.session_state and not run_detection:
            st.image(
                st.session_state["annotated_rgb"],
                caption="Detected Potholes & Defects",
                use_container_width=True
            )

    # 2. Trigger Inference when Button is Clicked
    if run_detection:
        try:
            with st.spinner("Analyzing road defects with YOLOv8..."):
                results = model.predict(
                    source=image_array,
                    conf=confidence_threshold,
                    device="cpu",
                    verbose=False,
                )
                result = results[0]

                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.session_state["annotated_rgb"] = annotated_rgb

                # Parse detection coordinates and confidence
                boxes = []
                confidences = []
                class_names = []

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    class_names = [model.names[cid] for cid in class_ids]

                # Severity calculation
                covered_area = calculate_union_area(boxes, image_width, image_height)
                image_area = image_width * image_height
                area_pct = (covered_area / image_area) * 100 if image_area > 0 else 0.0
                severity = calculate_severity(area_pct) if len(boxes) > 0 else "NONE"

                # Store summary metrics
                st.session_state["summary_data"] = {
                    "count": len(boxes),
                    "max_conf": f"{max(confidences) * 100:.1f}%" if len(confidences) > 0 else "N/A",
                    "area_pct": f"{area_pct:.2f}%",
                    "severity": severity,
                }

                # Build details DataFrame
                rows = []
                for index, (box, conf, cname) in enumerate(zip(boxes, confidences, class_names), start=1):
                    x1, y1, x2, y2 = box
                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    rows.append({
                        "Detection": f"#{index}",
                        "Class": cname,
                        "Confidence (%)": round(float(conf) * 100, 2),
                        "X1": round(float(x1), 1),
                        "Y1": round(float(y1), 1),
                        "X2": round(float(x2), 1),
                        "Y2": round(float(y2), 1),
                        "Box Area (px)": round(float(box_area), 1),
                    })

                st.session_state["detection_df"] = pd.DataFrame(rows)

            # Rerun so col_right displays the new image smoothly
            st.rerun()

        except Exception as error:
            st.error("Detection failed.")
            st.exception(error)

    # 3. Display Summary & Data if results exist
    if "summary_data" in st.session_state:
        summary = st.session_state["summary_data"]
        df = st.session_state.get("detection_df", pd.DataFrame())

        st.divider()
        st.subheader("📊 Detection Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Potholes Detected", summary["count"])
        m2.metric("Max Confidence", summary["max_conf"])
        m3.metric("Defect Coverage Area", summary["area_pct"])

        sev_colors = {
            "NONE": "🟢 NONE",
            "LOW": "🟢 LOW",
            "MEDIUM": "🟡 MEDIUM",
            "HIGH": "🟠 HIGH",
            "CRITICAL": "🔴 CRITICAL",
        }
        m4.metric("Severity Level", sev_colors.get(summary["severity"], summary["severity"]))

        st.caption(
            "Severity is based on total defect area coverage as per project guidelines."
        )

        st.divider()
        st.subheader("📋 Detailed Detections")

        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download Detection CSV Report",
                    data=csv_data,
                    file_name="pothole_detection_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with btn_col2:
                annotated_pil = Image.fromarray(st.session_state["annotated_rgb"])
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.success("✅ No potholes detected above the selected confidence threshold.")

else:
    st.info("Upload a road image above to begin.")
    st.markdown(
        """
        ### How to use:
        1. **Upload** a road surface photograph.
        2. Inspect the **Original Image** on the left.
        3. Click **Run Road Damage Detection** beneath the image.
        4. View the **Annotated Detections** side by side on the right.
        5. Check the severity rating and download reports below.
        """
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Powered by YOLOv8 | Road Pothole & Defect Detection System")