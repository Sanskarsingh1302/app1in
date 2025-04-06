import streamlit as st
from PIL import Image

# Open the image
img = Image.open("photo_2025-03-25_15-44-20.jpg")

# Set the Streamlit page config
st.set_page_config(
    page_title="Terrain Classification",
    page_icon=img  # PIL image works fine here
)

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
import plotly.graph_objects as go
from io import BytesIO
import google.generativeai as genai



# Add this at the top of your app, before the sidebar navigation
def show_welcome_page():
        # Function to load and convert the image to base64
    def get_base64_of_bin_file(bin_file):
       with open(bin_file, 'rb') as f:
          data = f.read()
       return base64.b64encode(data).decode()
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
        unsafe_allow_html=True)  # Replace with your logo path
    
    st.title("Welcome to Drone Terrain Analysis")
    st.markdown("""
    ### Your Complete Solution for Terrain Analysis and Object Detection
    
    This application helps drone operators and surveyors understand terrain characteristics 
    and identify objects within their flight areas.
    
    **Key Features:**
    - 🌏 **Terrain Classification**: Analyze your terrain images to identify the type of environment
    - 🔍 **Object Detection**: Identify and analyze objects present in your drone imagery
    - 📊 **Detailed Metrics**: Get specific recommendations for drone operations based on terrain
    
    *Get started by selecting an option from the sidebar!*
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Terrain Classification**\nUpload aerial images to classify terrain types and get drone flight recommendations.")
        if st.button("Start Terrain Analysis", key="terrain_btn"):
            st.session_state.app_mode = "Know your terrain"
            st.rerun()
    
    with col2:
        st.info("**Object Detection**\nUpload images to detect and analyze objects present in your drone's view and fligt path.")
        if st.button("Start Object Detection", key="object_btn"):
            st.session_state.app_mode = "Know What's in your terrain"
            st.rerun()
    
    # Quick tutorial expander
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Select a mode** from the sidebar (Terrain Classification or Object Detection)
        2. **Upload your drone images** in the selected mode
        3. **Analyze the results** and use the recommendations for your drone operations
        
        For terrain classification, we recommend images taken from 30-100m altitude.
        For object detection, images with clear visibility work best.
        """)






# Set your Gemini API key securely
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
genai.configure(api_key=GEMINI_API_KEY)


# Modify your existing sidebar navigation
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Welcome"
with st.sidebar:
    # Sidebar for navigation
    st.sidebar.title("Classification & Detection")
    app_mode = st.sidebar.radio(
        "Choose an App", 
        ["Welcome", "Know your terrain", "Know What's in your terrain"],
        index=["Welcome", "Know your terrain", "Know What's in your terrain"].index(st.session_state.app_mode)
    )
    st.markdown("---")
    st.markdown("### 🤖 AI Assistant")

    # Gemini-based Chatbot
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            "You are a helpful assistant for a Streamlit app called 'Drone-Assisted AI Terrain Mapping and Classification'. The app supports terrain classification, object detection, change detection, and elevation mapping from drone images. Help users understand and use these features."
        ]

    chat_box = st.container(height=300)

    for i, msg in enumerate(st.session_state.chat_history[1:], start=1):
        role = "user" if i % 2 != 0 else "assistant"
        with chat_box.chat_message(role):
            st.markdown(msg)

    if prompt := st.chat_input("Ask me anything about the app..."):
        st.session_state.chat_history.append(prompt)

        with chat_box.chat_message("user"):
            st.markdown(prompt)

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content(prompt)
            reply = response.text
        except Exception as e:
            reply = f"Error: {e}"

        st.session_state.chat_history.append(reply)
        with chat_box.chat_message("assistant"):
            st.markdown(reply)



# Main app logic
if app_mode == "Welcome":
    show_welcome_page()

elif app_mode == "Know your terrain":
    # Function to load and convert the image to base64
    def get_base64_of_bin_file(bin_file):
       with open(bin_file, 'rb') as f:
          data = f.read()
       return base64.b64encode(data).decode()
    
    
    def get_image_download_link(img, filename, text):
        """Generate a link to download an image"""
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
        return href

    # Add these functions to your "Know your terrain" section
    def calculate_terrain_metrics(predicted_class, confidence):
        """Calculate terrain-specific metrics and recommendations"""
        metrics = {}
    
        # Basic metrics for all terrain types
        metrics["suitability"] = {
            "drone_flight": min(confidence * 0.8 + 0.2, 1.0),
            "image_quality": min(confidence * 0.9 + 0.1, 1.0)
        }
    
        # Terrain-specific metrics
        if predicted_class == "Forest":
            metrics["vegetation_density"] = 0.8 + (confidence * 0.2)
            metrics["risk_factors"] = ["Limited visibility", "GPS signal interference", "Variable elevation"]
            metrics["flight_recommendations"] = [
                "Maintain higher altitude (minimum 30m above canopy)",
                "Use visual navigation backup",
                "Implement collision avoidance"
            ]

        elif predicted_class == "Annual Crop" or predicted_class == "Permanent Crop":
            metrics["vegetation_density"] = 0.5 + (confidence * 0.2)
            metrics["risk_factors"] = ["Seasonal changes", "Agricultural machinery", "Irrigation systems"]
            metrics["flight_recommendations"] = [
                "Ideal for regular monitoring",
                "Maintain medium altitude (15-30m)",
                "Consider time of day for shadows"
            ]

        elif predicted_class == "Highway":
            metrics["urban_density"] = 0.3 + (confidence * 0.3)
            metrics["risk_factors"] = ["Moving vehicles", "Power lines", "Legal restrictions"]
            metrics["flight_recommendations"] = [
                "Check local regulations",
                "Avoid peak traffic hours",
                "Maintain safe distance from traffic",
                "Be aware of surrounding infrastructure"
            ]

        elif predicted_class == "Industrial":
            metrics["urban_density"] = 0.7 + (confidence * 0.2)
            metrics["risk_factors"] = ["Electromagnetic interference", "Tall structures", "Legal restrictions"]
            metrics["flight_recommendations"] = [
                "Obtain proper permissions",
                "Maintain high altitude (30m+)",
                "Monitor signal strength",
                "Avoid active machinery areas"
            ]

        elif predicted_class == "Residential":
            metrics["urban_density"] = 0.6 + (confidence * 0.3)
            metrics["risk_factors"] = ["Privacy concerns", "Legal restrictions", "Obstacles"]
            metrics["flight_recommendations"] = [
                "Ensure compliance with privacy laws",
                "Maintain regulatory minimum altitude",
                "Plan flight path to avoid direct views into properties"
            ]

        elif predicted_class in ["River", "Sea Lake"]:
            metrics["water_coverage"] = 0.8 + (confidence * 0.2)
            metrics["risk_factors"] = ["Water reflections", "Wind over water", "Emergency landing difficulty"]
            metrics["flight_recommendations"] = [
                "Use polarizing filter for camera",
                "Monitor wind conditions carefully",
                "Maintain safe distance from shoreline",
                "Have recovery plan for water landings"
            ]

        elif predicted_class == "Herbaceous Vegetation" or predicted_class == "Pasture":
            metrics["vegetation_density"] = 0.4 + (confidence * 0.2)
            metrics["risk_factors"] = ["Varied visibility", "Wildlife", "Wind gusts in open areas"]
            metrics["flight_recommendations"] = [
                "Ideal for mapping and surveying",
                "Maintain medium altitude (10-20m)",
                "Watch for sudden weather changes"
            ]

        return metrics

        
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
    # Add this after your existing terrain classification display code
    if uploaded_file is not None:
        # After displaying the classification result and confidence chart
        st.header("Terrain Analysis Metrics", divider=True)

        # Calculate metrics based on the predicted class
        terrain_metrics = calculate_terrain_metrics(predicted_class, np.max(prediction))
    
        # Display the metrics in multiple columns
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Suitability Metrics")

            # Display suitability metrics as progress bars
            st.markdown("**Drone Flight Suitability:**")
            st.progress(float(terrain_metrics["suitability"]["drone_flight"]))
            st.markdown(f"{terrain_metrics['suitability']['drone_flight']:.2f} / 1.0")

            st.markdown("**Image Quality Expected:**")
            st.progress(float(terrain_metrics["suitability"]["image_quality"]))
            st.markdown(f"{terrain_metrics['suitability']['image_quality']:.2f} / 1.0")

            # Display any additional specific metrics
            if "vegetation_density" in terrain_metrics:
                st.markdown("**Vegetation Density:**")
                st.progress(float(terrain_metrics["vegetation_density"]))
                st.markdown(f"{terrain_metrics['vegetation_density']:.2f} / 1.0")

            if "urban_density" in terrain_metrics:
                st.markdown("**Urban Density:**")
                st.progress(float(terrain_metrics["urban_density"]))
                st.markdown(f"{terrain_metrics['urban_density']:.2f} / 1.0")

            if "water_coverage" in terrain_metrics:
                st.markdown("**Water Coverage:**")
                st.progress(float(terrain_metrics["water_coverage"]))
                st.markdown(f"{terrain_metrics['water_coverage']:.2f} / 1.0")

        with col2:
            st.subheader("Flight Recommendations")

            # Display risk factors
            st.markdown("**Risk Factors:**")
            for risk in terrain_metrics.get("risk_factors", []):
                st.markdown(f"- {risk}")

            # Display recommendations
            st.markdown("**Recommendations:**")
            for recommendation in terrain_metrics.get("flight_recommendations", []):
                st.markdown(f"- {recommendation}")

        # Add a section for detailed terrain analysis
        with st.expander("Detailed Terrain Analysis", expanded=False):
            st.write("""
            This section provides a detailed analysis of the terrain and its implications for drone mapping.
            The metrics above are calculated based on the classification confidence and typical characteristics
            of the identified terrain type.
            """)

            # Create a radar chart for terrain characteristics
            characteristics = {
                "Annual Crop": {"Flatness": 0.8, "Accessibility": 0.9, "Visibility": 0.9, "Obstacles": 0.3, "Signal": 0.8},
                "Forest": {"Flatness": 0.4, "Accessibility": 0.5, "Visibility": 0.3, "Obstacles": 0.9, "Signal": 0.5},
                "Herbaceous Vegetation": {"Flatness": 0.7, "Accessibility": 0.8, "Visibility": 0.8, "Obstacles": 0.4, "Signal": 0.9},
                "Highway": {"Flatness": 0.9, "Accessibility": 0.7, "Visibility": 0.9, "Obstacles": 0.6, "Signal": 0.7},
                "Industrial": {"Flatness": 0.8, "Accessibility": 0.5, "Visibility": 0.7, "Obstacles": 0.8, "Signal": 0.6},
                "Pasture": {"Flatness": 0.8, "Accessibility": 0.9, "Visibility": 0.9, "Obstacles": 0.2, "Signal": 0.9},
                "Permanent Crop": {"Flatness": 0.7, "Accessibility": 0.8, "Visibility": 0.7, "Obstacles": 0.5, "Signal": 0.8},
                "Residential": {"Flatness": 0.8, "Accessibility": 0.6, "Visibility": 0.7, "Obstacles": 0.7, "Signal": 0.7},
                "River": {"Flatness": 0.5, "Accessibility": 0.4, "Visibility": 0.8, "Obstacles": 0.3, "Signal": 0.8},
                "Sea Lake": {"Flatness": 0.9, "Accessibility": 0.3, "Visibility": 0.9, "Obstacles": 0.1, "Signal": 0.7}
            }

            if predicted_class in characteristics:
                # Prepare data for radar chart
                categories = list(characteristics[predicted_class].keys())
                values = list(characteristics[predicted_class].values())

                # Create the radar chart
                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=predicted_class
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title=f"Terrain Characteristics: {predicted_class}"
                )

                st.plotly_chart(fig)

                # Add textual explanation
                st.subheader("What This Means For Drone Mapping")
                explanations = {
                    "Annual Crop": "Annual crop areas are ideal for drone mapping due to their flat terrain, good visibility, and minimal obstacles. Seasonal changes should be considered when planning surveys.",
                    "Forest": "Forest areas present challenges for drone mapping due to canopy cover, potential GPS signal interference, and limited visual line of sight. Higher altitudes and careful flight planning are recommended.",
                    "Herbaceous Vegetation": "Areas with herbaceous vegetation offer good conditions for drone mapping with minimal obstacles and good visibility. Wind conditions can be a factor in these open areas.",
                    "Highway": "Highway areas have excellent flatness and visibility but present regulatory challenges and safety concerns due to moving vehicles and nearby infrastructure.",
                    "Industrial": "Industrial areas require special attention to regulations, electromagnetic interference, and obstacle avoidance. Detailed pre-flight planning is essential.",
                    "Pasture": "Pasture lands are among the most suitable for drone operations with excellent visibility, flat terrain, and minimal obstacles.",
                    "Permanent Crop": "Permanent crop areas provide good mapping conditions with some consideration needed for tree height and spacing.",
                    "Residential": "Residential areas present privacy and regulatory challenges that must be addressed before flight. Obstacle avoidance is a key consideration.",
                    "River": "River mapping requires careful consideration of reflective surfaces, which can affect image quality. Emergency landing options are limited.",
                    "Sea Lake": "Water bodies offer excellent visibility and flatness but present challenges for emergency landings and sometimes wind conditions."
                }

                st.write(explanations.get(predicted_class, "No specific guidance available for this terrain type."))
    terrain_comparison_view = st.radio("terrain comparison view",[":terrain_comparison_view"])
    if terrain_comparison_view == ":terrain_comparison_view":
    
        st.header("Terrain Comparison Tool", divider=True)




        with st.expander("How to use the comparison tool", expanded=False):
            st.write("""
            Upload two or more terrain images to compare their characteristics side by side.
            This tool helps you:
           1. Compare different areas before deciding where to fly
            2. Analyze how the same area changes over time or seasons
            3. Understand the differences between similar terrain types
            """)
    
        # File uploader for multiple images
        uploaded_files = st.file_uploader("Upload images to compare", 
                                         type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) >= 2:
            # Create comparison gallery
            cols = st.columns(min(len(uploaded_files), 3))

            # Process each image and display them side by side
            comparison_results = []

            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i % 3]:
                    # Display the image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_Container_width=True)

                    # Process the image and get prediction
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        processed_image = preprocess_image(image)
                        prediction = model.predict(processed_image)

                        # Get the predicted class
                        class_names = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
                                      'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea Lake']
                        predicted_class = class_names[np.argmax(prediction)]
                        confidence = np.max(prediction)

                        # Calculate metrics
                        metrics = calculate_terrain_metrics(predicted_class, confidence)

                        # Store results
                        comparison_results.append({
                            'filename': uploaded_file.name,
                            'image': image,
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'metrics': metrics
                        })

                        # Display basic results
                        st.markdown(f"**Prediction: {predicted_class}**")
                        st.progress(float(confidence))
                        st.markdown(f"Confidence: {confidence:.2f}")

            # Compare the results in a table
            st.subheader("Comparison Results")

            # Create a comparison dataframe
            comparison_data = []
            for result in comparison_results:
                row = {
                    'Image': result['filename'],
                    'Terrain Type': result['predicted_class'],
                    'Confidence': f"{result['confidence']:.2f}",
                    'Drone Flight Suitability': f"{result['metrics']['suitability']['drone_flight']:.2f}",
                    'Image Quality Expectation': f"{result['metrics']['suitability']['image_quality']:.2f}"
                }

                # Add optional metrics if they exist
                if 'vegetation_density' in result['metrics']:
                    row['Vegetation Density'] = f"{result['metrics']['vegetation_density']:.2f}"
                if 'urban_density' in result['metrics']:
                    row['Urban Density'] = f"{result['metrics']['urban_density']:.2f}"
                if 'water_coverage' in result['metrics']:
                    row['Water Coverage'] = f"{result['metrics']['water_coverage']:.2f}"

                comparison_data.append(row)

            # Create dataframe and display
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

            # Create visual comparison chart
            st.subheader("Visual Comparison")

            # Extract metrics for visualization
            chart_data = {
                'Image': [result['filename'] for result in comparison_results],
                'Drone Flight Suitability': [result['metrics']['suitability']['drone_flight'] for result in comparison_results],
                'Image Quality': [result['metrics']['suitability']['image_quality'] for result in comparison_results]
            }

            # Add optional metrics if they exist in any result
            if any('vegetation_density' in result['metrics'] for result in comparison_results):
                chart_data['Vegetation Density'] = [result['metrics'].get('vegetation_density', 0) for result in comparison_results]
            if any('urban_density' in result['metrics'] for result in comparison_results):
                chart_data['Urban Density'] = [result['metrics'].get('urban_density', 0) for result in comparison_results]
            if any('water_coverage' in result['metrics'] for result in comparison_results):
                chart_data['Water Coverage'] = [result['metrics'].get('water_coverage', 0) for result in comparison_results]

            # Create DataFrame for plotting
            chart_df = pd.DataFrame(chart_data)

            # Melt the DataFrame for easier plotting
            melted_df = pd.melt(chart_df, id_vars=['Image'], var_name='Metric', value_name='Value')

            # Create a grouped bar chart
            fig = px.bar(melted_df, x='Image', y='Value', color='Metric', barmode='group',
                        title='Terrain Metrics Comparison', height=500)
            st.plotly_chart(fig)

            # Recommendation based on comparison
            st.subheader("Comparison Analysis")

            # Find the best terrain for drone flight
            best_flight_index = np.argmax([result['metrics']['suitability']['drone_flight'] for result in comparison_results])
            best_flight_terrain = comparison_results[best_flight_index]['filename']

            st.markdown(f"""
            **Best terrain for drone flight: {best_flight_terrain}** 

            This terrain offers the most suitable conditions for drone operations based on our analysis.
            """)

            # Add a download button for the comparison report
            st.markdown("### Download Comparison Report")

            # Create a buffer for the report
            buffer = BytesIO()

            # Write the comparison data to the buffer as CSV
            comparison_df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Create a download button
            st.download_button(
                label="Download Comparison CSV",
                data=buffer,
                file_name="terrain_comparison.csv",
                mime="text/csv"
            )





elif app_mode == "Know What's in your terrain":
    # Load YOLO Models
    model_yolov8m = YOLO("yolov8m.pt")
    model_yolo11m_obb = YOLO("yolo11n-obb.pt")

    # Streamlit UI
    st.title("YOLO Object Detection App")

    # Upload Images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    model_choice = st.radio("Select a model:", ["YOLOv8m", "YOLO 11m-OBB"])
    
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








elif app_mode == "Know What's in your terrain":
    # Load YOLO Models
    model_yolov8m = YOLO("yolov8m.pt")
    model_yolo11m_obb = YOLO("yolo11n-obb.pt")

    # Streamlit UI
    st.title("YOLO Object Detection App")

    # Upload Images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    model_choice = st.radio("Select a model:", ["YOLOv8m", "YOLO 11m-OBB"])
    
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
                    st.image(processed_image, caption=f"Detected Objects: {uploaded_file.name}", use_container_width=True)
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

