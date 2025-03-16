import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import altair as alt
import pandas as pd
import cv2
from ultralytics import YOLO
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Sidebar for navigation
st.sidebar.title("Classification & Detection")
app_mode = st.sidebar.radio("Choose an App", ["Know your terrain", "Know What's in your terrain"])

if app_mode == "Know your terrain":
    # Function to load and convert the image to base64
    def get_base64_of_bin_file(bin_file):
       with open(bin_file, 'rb') as f:
          data = f.read()
       return base64.b64encode(data).decode()


    
    # Load Model
    model_path = r"terrain_classifier_eurosat (1).keras"
    try:
        model = load_model(model_path)
        st.success(f"Model loaded successfully from {model_path}")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Load your logo and convert it to base64 format
    logo_path = r"logo (1).png" 
    logo_base64 = get_base64_of_bin_file(logo_path)

    # Centering the logo using HTML and CSS
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
        <img src="data:image/jpeg;base64,{logo_base64}" alt="logo" style="width:450px;">
        </div>
        """,
        unsafe_allow_html=True)
    # Define the function to preprocess the image
    def preprocess_image(image, target_size=(64, 64)):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
    
        return image

    # Define the prediction function
    def make_prediction(image):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction

    # Function to map predicted classes to emojis
    def get_emoji_for_class(predicted_class):
        emoji_dict = {
            'Annual Crop': '🌾',
            'Forest': '🌲',
            'Herbaceous Vegetation': '🌿',
            'Highway': '🛣️',
            'Industrial': '🏭',
            'Pasture': '🐄',
            'Permanent Crop': '🌳',
            'Residential': '🏠',
            'River': '🌊',
            'Sea Lake': '⛵'
        }
        return emoji_dict.get(predicted_class, '🌍')  # Default to globe emoji if not found

    # Function to display a bar chart using Altair
    def display_confidence_bar_chart(prediction, class_names):
        confidence_scores = prediction[0]
    
        # Create a DataFrame for Altair
        data = pd.DataFrame({
            'Class': class_names,
            'Confidence': confidence_scores
        })

        #  Create an Altair bar chart
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Class', sort=None, title='Terrain Class'),
            y=alt.Y('Confidence', title='Confidence Score'),
            color=alt.Color('Class', legend=None),
            tooltip=['Class', 'Confidence']
        ).properties(
            width=600,
            height=400,
            title="Model Confidence for Each Terrain Class"
        )

        st.altair_chart(chart, use_container_width=True)

    # Streamlit app UI
    st.markdown("<h1 style='text-align: center;'>🌏 Terrain Classification App 🌏</h1>", unsafe_allow_html=True)

    with st.expander("How to Use", expanded=False):
        st.markdown(
            """
            1. Upload a terrain image in JPG, JPEG, or PNG format.
            2. The app will process the image and display the predicted terrain class.
            3. Confidence scores for each terrain type will be shown.
            """
        )

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.header("Classified image as:", divider=True)

        # Make the prediction
        prediction = make_prediction(image)
    
        # Define class names (update these based on your model’s output)
        class_names = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
                       'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea Lake']
    
        predicted_class = class_names[np.argmax(prediction)]  # Get predicted class
        emoji = get_emoji_for_class(predicted_class)  # Get emoji for the predicted class
    
        # Customized Predictive Model Output - Improved Design
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border: 3px solid #feb47b;
                border-radius: 15px;
                background: linear-gradient(135deg, #ff7e5f, #feb47b);
                text-align: center;
                font-size: 24px;
                color: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                margin: 20px 0;
            ">
                <h2 style="margin-bottom: 10px;">Predicted Class</h2>
                <strong>{predicted_class} {emoji}</strong>
            </div>
            """, unsafe_allow_html=True
        )

        # Display the Altair bar chart
        display_confidence_bar_chart(prediction, class_names)












elif app_mode == "Know What's in your terrain":
    # Load YOLO Models
    model_yolov8m = YOLO("yolov8m.pt")
    model_yolo11m_obb = YOLO("yolo11n-obb.pt")

    # Streamlit UI
    st.title("Know What's in your terrain")
    with st.expander("How to Use", expanded=False):
        st.markdown(
            """
            1. Multiple images can be uploaded in JPG, JPEG, or PNG format.
            2. The app will process the image and display the objects within the image.
            3. It will generate the summary of the set of images with detection frequency, proportion of detected objects, confidence score heat map, and confidence score distribution, and a CVS file.
            """
        )
    # Upload Images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    model_choice = st.selectbox(
    "Choose a Model",
    ["Balanced Accuracy & Speed", "For Oriented Objects"]
    )

    
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    iou_thresh = st.slider("IOU Threshold", 0.1, 1.0, 0.5, 0.05)
    
    def process_image(image, model, class_list):
        image = np.array(image)
        results = model(image, conf=conf_thresh, iou=iou_thresh)
        detected_objects = []
    
        if not results or (not hasattr(results[0], 'boxes') or results[0].boxes is None) and (not hasattr(results[0], 'obb') or results[0].obb is None):
            return None, "No objects detected!"
    
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = map(int, box[:4])
                    class_id = int(cls)
                    class_name = class_list.get(class_id, f"Class {class_id}")
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_objects.append({"class": class_name, "confidence": float(conf)})
        
            if hasattr(result, 'obb') and result.obb is not None:
                for obb, conf, cls in zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls):
                    if conf < conf_thresh:
                        continue
                    class_id = int(cls)
                    class_name = class_list.get(class_id, f"Class {class_id}")
                    label = f"{class_name} {conf:.2f}"
                    obb_points = np.array(obb).reshape(-1, 2).astype(int)
                    cv2.polylines(image_bgr, [obb_points], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.putText(image_bgr, label, (obb_points[0][0], obb_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    detected_objects.append({"class": class_name, "confidence": float(conf)})

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb, detected_objects

    
    if uploaded_files:
        model = model_yolov8m if model_choice == "YOLOv8m" else model_yolo11m_obb
        class_list = model.model.names if hasattr(model.model, 'names') else {i: f"Class {i}" for i in range(100)}
        all_results_data = []
    
        if st.button("Run Inference on All Images"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            
                st.write("Running detection...")
                processed_image, detected_objects = process_image(image, model, class_list)
            
                if processed_image is None:
                   st.error("No objects detected! Check model and input image.")
                else:
                    st.image(processed_image, caption=f"Detected Objects: {uploaded_file.name}", use_column_width=True)
                    st.success("Inference Completed!")
                    all_results_data.append((uploaded_file.name, detected_objects))

        if all_results_data:
            combined_results = []
            for filename, detected_objects in all_results_data:
                for obj in detected_objects:
                    combined_results.append({"Image": filename, "Class": obj["class"], "Confidence": obj["confidence"]})
            results_df = pd.DataFrame(combined_results)

            # Summary Metrics
            st.write("### Summary of Detections")
            st.metric("Total Images Processed", len(uploaded_files))
            st.metric("Total Detections", len(results_df))
            if not results_df.empty:
                st.metric("Most Common Class", results_df["Class"].mode()[0])

            # Enhanced Bar Chart
            st.write("### Detection Frequency per Image")
            grouped_counts = results_df.groupby(["Image", "Class"]).size().reset_index(name="Count")
            fig_bar = px.bar(grouped_counts, x="Class", y="Count", color="Image", barmode="group", title="Detection Frequency")
            st.plotly_chart(fig_bar)

            # Pie Chart of Class Distribution
            fig_pie = px.pie(results_df, names="Class", title="Proportion of Detected Objects", hole=0.4)
            st.plotly_chart(fig_pie)

            # Confidence Score Heatmap
            if len(results_df) > 1:
                st.write("### Confidence Score Heatmap")
                fig, ax = plt.subplots()
                heatmap_data = results_df.pivot_table(index="Image", columns="Class", values="Confidence", aggfunc="mean")
                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            # Histogram of Confidence Scores
            fig_hist = px.histogram(results_df, x="Confidence", nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig_hist)

            # Display Separate Tables for Each Image
            for filename, detected_objects in all_results_data:
                st.write(f"### Detection Results for {filename}")
                df = pd.DataFrame(detected_objects)
                st.table(df)
            # Download JSON Button
            results_json = json.dumps(combined_results, indent=4)
            st.download_button("Download Results as JSON", results_json, "results.json", "application/json")

          
