"""
🍅 Tomato Ripeness Detector — Upload & Live Camera with Accurate Color Detection
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# RTC configuration (required for camera)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("tomato_simple_model.h5")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

def analyze_tomato_color(img):
    """
    Analyze the dominant color of the tomato to help with classification
    Returns: 'red', 'green', 'yellow', or 'mixed'
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    # Red color (ripe tomatoes)
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Green color (unripe tomatoes)
    green_lower = np.array([36, 50, 50])
    green_upper = np.array([85, 255, 255])
    
    # Yellow color (partially ripe)
    yellow_lower = np.array([20, 100, 50])
    yellow_upper = np.array([35, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Count pixels for each color
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    yellow_pixels = np.sum(yellow_mask > 0)
    total_pixels = img.shape[0] * img.shape[1]
    
    # Calculate percentages
    red_percent = red_pixels / total_pixels
    green_percent = green_pixels / total_pixels
    yellow_percent = yellow_pixels / total_pixels
    
    # Determine dominant color
    if red_percent > 0.3 and red_percent > green_percent and red_percent > yellow_percent:
        return "red", red_percent
    elif green_percent > 0.3 and green_percent > red_percent and green_percent > yellow_percent:
        return "green", green_percent
    elif yellow_percent > 0.3 and yellow_percent > red_percent and yellow_percent > green_percent:
        return "yellow", yellow_percent
    else:
        return "mixed", max(red_percent, green_percent, yellow_percent)

# Preprocess image for model
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Classify tomato using model with color validation
def classify_tomato(img):
    if model is None:
        return "Ripe", 0.5  # Default if model not loaded
    
    processed = preprocess_image(img)
    prediction = model.predict(processed, verbose=0)
    confidence = float(prediction[0][0])
    
    # Analyze actual color for validation
    dominant_color, color_confidence = analyze_tomato_color(img)
    
    # Combine model prediction with color analysis
    if confidence > 0.5:
        model_says = "Unripe"
        model_confidence = confidence
    else:
        model_says = "Ripe"
        model_confidence = 1.0 - confidence
    
    # Color-based validation
    if model_says == "Ripe" and dominant_color in ["green", "yellow"]:
        # Model says ripe but color says unripe - trust color more
        if color_confidence > 0.4:
            return "Unripe", color_confidence
    elif model_says == "Unripe" and dominant_color == "red":
        # Model says unripe but color says ripe - trust color more
        if color_confidence > 0.4:
            return "Ripe", color_confidence
    
    return model_says, model_confidence

def get_tomato_contours(img):
    """Get tomato contours using advanced color detection"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Tomato color ranges (red, yellow, green)
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([20, 100, 50])
    yellow_upper = np.array([35, 255, 255])
    
    green_lower = np.array([36, 50, 50])
    green_upper = np.array([85, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Combine all tomato color masks
    tomato_mask = cv2.bitwise_or(red_mask, yellow_mask)
    tomato_mask = cv2.bitwise_or(tomato_mask, green_mask)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, kernel)
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:  # Minimum area for tomatoes
            # Check circularity (tomatoes are generally round)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Tomatoes are somewhat round
                    valid_contours.append(cnt)
    
    return valid_contours

# Video transformer for real-time analysis
class TomatoDetector(VideoTransformerBase):
    def __init__(self):
        self.last_analysis_time = 0
        self.analysis_interval = 1.0  # Analyze every 1 second
        self.current_predictions = {}
        self.last_detected_label = "None"  # Store last label globally

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        output = img.copy()
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Get contours for possible tomatoes
        contours = get_tomato_contours(img)
        
        if len(contours) == 0:
            # No tomato detected
            self.last_detected_label = "Not a Tomato"
            return output
        
        detected_any_tomato = False
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            if w > 50 and h > 50 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                tomato_crop = img[y:y+h, x:x+w]
                detected_any_tomato = True
                
                contour_key = f"{x}_{y}_{w}_{h}"
                
                if (current_time - self.last_analysis_time > self.analysis_interval or 
                    contour_key not in self.current_predictions):
                    
                    label, confidence = classify_tomato(tomato_crop)
                    self.current_predictions[contour_key] = (label, confidence)
                    self.last_analysis_time = current_time
                
                label, confidence = self.current_predictions.get(contour_key, ("Unripe", 0.5))
                
                if label == "Ripe":
                    color = (0, 0, 255)
                    message = "RIPE TOMATO"
                else:
                    color = (0, 255, 255)
                    message = "UNRIPE TOMATO"
                
                # Draw bounding box & text
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
                label_text = f"{message} ({confidence*100:.1f}%)"
                cv2.putText(output, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # If no tomato-like color found, mark as "Not a Tomato"
        if not detected_any_tomato:
            self.last_detected_label = "Not a Tomato"
        else:
            self.last_detected_label = label

        return output

# Streamlit UI
st.set_page_config(
    page_title="🍅 Tomato Ripeness Detector",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato Ripeness Detector")
st.markdown("**Upload an image or use live camera to detect tomato ripeness!**")

# Model status check
if model is None:
    st.error("""
    **Model not found!** Please make sure:
    1. You have trained the model first
    2. The file `tomato_simple_model.h5` exists in the same directory
    3. The model file is not corrupted
    """)
    st.stop()

# Create tabs for Upload and Live Camera
tab1, tab2 = st.tabs(["📁 Upload Image", "🎥 Live Camera"])

# TAB 1: Upload Image
with tab1:
    st.subheader("📁 Upload Tomato Image")
    st.markdown("Upload a clear image of a tomato to check if it's ripe or unripe!")
    
    uploaded_file = st.file_uploader(
        "Choose a tomato image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a tomato"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_container_width=True)
            
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 2:  # Grayscale
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                elif image_array.shape[2] == 3:  # RGB
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing your tomato..."):
                # Classify the tomato
                result, confidence = classify_tomato(image_array)
                
                # Analyze color
                dominant_color, color_conf = analyze_tomato_color(image_array)
                
                if result == "Ripe":
                    st.success(f"🔴 **RIPE TOMATO** (Confidence: {confidence:.1%})")
                    st.markdown("This tomato appears to be **ripe** and ready to eat! 🍅")
                    st.balloons()
                else:
                    st.warning(f"🟡 **UNRIPE TOMATO** (Confidence: {confidence:.1%})")
                    st.markdown("This tomato appears to be **unripe**. Give it more time to ripen! 🕒")
                
                # Show confidence bar
                st.progress(confidence)
                st.caption(f"Model confidence: {confidence:.1%}")
                
                # Show color analysis
                st.info(f"**Detected color:** {dominant_color.upper()} ({color_conf:.1%} of pixels)")

# TAB 2: Live Camera (Keep existing working code)
with tab2:
    st.subheader("🎥 Live Camera Detection")
    st.markdown("""
    Show tomatoes to your webcam and the app will detect them with:
    - **🔴 RED BOX** = RIPE tomatoes (ready to eat)
    - **🟡 YELLOW BOX** = UNRIPE tomatoes (needs more time)
    """)

    
    # Camera stream
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="tomato-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=TomatoDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    # Display detection results outside camera
    if 'webrtc_ctx' in locals() and webrtc_ctx.video_transformer:
        label = webrtc_ctx.video_transformer.last_detected_label
        
        if label == "Not a Tomato":
            st.warning("⚠️ This object is not a tomato. The system only detects tomato ripeness.")
        elif label == "Ripe":
            st.success("✅ Detected a *Ripe Tomato* — ready to eat! 🍅")
        elif label == "Unripe":
            st.info("🕒 Detected an *Unripe Tomato* — needs more time to ripen.")

    with col2:
        st.markdown("### 🎯 Detection Legend")
        
        st.markdown("""
        <div style='border: 3px solid #FF0000; padding: 10px; border-radius: 5px; margin: 10px 0; background: #FFE6E6;'>
        <h4>🔴 RIPE TOMATO</h4>
        <p>Red color, ready to eat! 🍅</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='border: 3px solid #FFFF00; padding: 10px; border-radius: 5px; margin: 10px 0; background: #FFFFE6;'>
        <h4>🟡 UNRIPE TOMATO</h4>
        <p>Green/Yellow color, needs time ⏳</p>
        </div>
        """, unsafe_allow_html=True)

# Tips and information
st.markdown("---")
st.markdown("### 💡 Tips for Best Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **✅ DO:**
    - Good lighting
    - Steady camera
    - Clear tomato view
    - Single tomato focus
    """)

with col2:
    st.markdown("""
    **❌ DON'T:**
    - Poor lighting
    - Fast movements
    - Busy background
    - Multiple tomatoes
    """)

with col3:
    st.markdown("""
    **🎯 EXPECT:**
    - Red tomatoes → RIPE
    - Green tomatoes → UNRIPE  
    - Yellow tomatoes → UNRIPE
    - High confidence scores
    """)

st.info("""
**Note:** The system uses both AI model prediction and color analysis to ensure accurate ripeness detection. 
Red tomatoes will always be classified as RIPE, while green/yellow tomatoes will be classified as UNRIPE.
""")