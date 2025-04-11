import streamlit as st
import requests
import io
from PIL import Image
import base64
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set page config
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .enhancement-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-text {
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        color: #2E4053;
    }
    .subheader-text {
        font-family: 'Arial', sans-serif;
        color: #566573;
    }
    .upload-section {
        border: 2px dashed #bbb;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'params' not in st.session_state:
    st.session_state.params = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

# Header with animation
col1, col2 = st.columns([1, 3])
with col1:
    lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_mdbdc5l7.json")
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, height=150, key="header_animation")
    else:
        st.image("https://www.svgrepo.com/show/530453/image-editing.svg", width=150)
with col2:
    st.markdown("<h1 class='header-text'>AI Image Enhancer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader-text'>Upload an image and let AI enhance it automatically based on image metrics</p>", unsafe_allow_html=True)

st.divider()

API_URL = "https://perfectpixel.onrender.com"  # Change if your FastAPI runs on different port

# Create a sidebar with info
with st.sidebar:
    st.header("Configuration")
    api_endpoint = st.text_input("API Endpoint", API_URL)
    st.write("---")
    st.markdown("<h3 class='header-text'>How It Works</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. Upload an image
    2. Our system analyzes the image's brightness, contrast, and sharpness
    3. AI determines optimal enhancement parameters
    4. The image is enhanced using these parameters
    """)
    
    st.markdown("<h3 class='header-text'>Image Metrics Explained</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Brightness**: Average pixel intensity (0-255)
    - **Contrast**: Standard deviation of pixel values
    - **Sharpness**: Laplacian variance - measures edge definition
    """)

# Function to send image to API
def enhance_image(uploaded_file):
    try:
        # Replace with your actual API endpoint
        api_url = "https://perfectpixel.onrender.com/enhance-image"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            # Convert base64 image back to bytes
            if isinstance(result["enhanced_image"], str):
                result["enhanced_image"] = base64.b64decode(result["enhanced_image"])
            return result
        else:
            st.error(f"API Error: {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# File upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="file_uploader")
st.markdown("</div>", unsafe_allow_html=True)

# Process uploaded image when it's new or when triggered by the user
if uploaded_file is not None:
    # Check if this is a new file or we've already processed this one
    file_changed = (st.session_state.current_file_name != uploaded_file.name) or not st.session_state.file_processed
    
    # Only process if it's a new file or we haven't processed it yet
    if file_changed and not st.session_state.processing:
        try:
            # Display original image
            image_bytes = uploaded_file.getvalue()
            original_image = Image.open(io.BytesIO(image_bytes))
            st.session_state.original_image = original_image
            
            # Update current file
            st.session_state.current_file_name = uploaded_file.name
            
            # Process image with API
            with st.spinner("Enhancing image with AI..."):
                st.session_state.processing = True
                
                # Progress bar animation
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                
                result = enhance_image(uploaded_file)
                
                if result:
                    enhanced_image_bytes = result["enhanced_image"]
                    st.session_state.enhanced_image = Image.open(io.BytesIO(enhanced_image_bytes))
                    st.session_state.metrics = result["original_metrics"]
                    st.session_state.params = result["params"]
                    st.session_state.file_processed = True
                
                st.session_state.processing = False
                st.rerun()
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.session_state.processing = False
            st.session_state.file_processed = False

# Display results
if st.session_state.original_image is not None and st.session_state.enhanced_image is not None:
    st.markdown("<h2 class='header-text'>Results</h2>", unsafe_allow_html=True)
    
    # Image comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='enhancement-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader-text'>Original Image</h3>", unsafe_allow_html=True)
        st.image(st.session_state.original_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='enhancement-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader-text'>Enhanced Image</h3>", unsafe_allow_html=True)
        st.image(st.session_state.enhanced_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics and parameters
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='enhancement-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader-text'>Original Metrics</h3>", unsafe_allow_html=True)
        
        # Create a radar chart for original metrics
        metrics = st.session_state.metrics
        
        # Scale metrics for better visualization
        brightness_normalized = min(metrics["brightness"] / 255, 1)
        contrast_normalized = min(metrics["contrast"] / 100, 1)  # Normalize for visualization
        sharpness_normalized = min(metrics["sharpness"] / 1000, 1)  # Normalize for visualization
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[brightness_normalized, contrast_normalized, sharpness_normalized, brightness_normalized],
            theta=['Brightness', 'Contrast', 'Sharpness', 'Brightness'],
            fill='toself',
            name='Image Metrics',
            line_color='#1E88E5'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Brightness', 'Contrast', 'Sharpness'],
            'Value': [
                f"{metrics['brightness']:.2f} (0-255)",
                f"{metrics['contrast']:.2f}",
                f"{metrics['sharpness']:.2f}"
            ]
        })
        st.table(metrics_df)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='enhancement-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader-text'>Enhancement Parameters</h3>", unsafe_allow_html=True)
        
        params = st.session_state.params
        
        # Create a bar chart for the enhancement factors
        params_df = pd.DataFrame({
            'Parameter': ['Brightness', 'Contrast', 'Sharpness'],
            'Factor': [
                params['brightness_factor'],
                params['contrast_factor'],
                params['sharpness_factor']
            ]
        })
        
        fig = px.bar(
            params_df,
            x='Parameter',
            y='Factor',
            color='Parameter',
            color_discrete_map={
                'Brightness': '#4CAF50',
                'Contrast': '#2196F3',
                'Sharpness': '#FF9800'
            },
            text=params_df['Factor'].apply(lambda x: f"{x:.2f}x")
        )
        
        fig.update_layout(
            title='Enhancement Factors Applied',
            yaxis=dict(
                title='Factor (Multiplier)',
                range=[0, max(params_df['Factor']) * 1.2]
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display explanation
        if params['brightness_factor'] > 1.05:
            brightness_effect = "increased"
        elif params['brightness_factor'] < 0.95:
            brightness_effect = "decreased"
        else:
            brightness_effect = "maintained"
            
        if params['contrast_factor'] > 1.05:
            contrast_effect = "increased"
        elif params['contrast_factor'] < 0.95:
            contrast_effect = "decreased"
        else:
            contrast_effect = "maintained"
            
        if params['sharpness_factor'] > 1.05:
            sharpness_effect = "increased"
        elif params['sharpness_factor'] < 0.95:
            sharpness_effect = "decreased"
        else:
            sharpness_effect = "maintained"
        
        st.markdown(f"""
        <p class='subheader-text'>
        The AI determined that your image would look best with:
        <ul>
            <li><b>Brightness {brightness_effect}</b> by a factor of {params['brightness_factor']:.2f}x</li>
            <li><b>Contrast {contrast_effect}</b> by a factor of {params['contrast_factor']:.2f}x</li>
            <li><b>Sharpness {sharpness_effect}</b> by a factor of {params['sharpness_factor']:.2f}x</li>
        </ul>
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download button for enhanced image
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buf = io.BytesIO()
        st.session_state.enhanced_image.save(buf, format="PNG")
        btn = st.download_button(
            label="Download Enhanced Image",
            data=buf.getvalue(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

    # Reset button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Process New Image"):
            st.session_state.original_image = None
            st.session_state.enhanced_image = None
            st.session_state.metrics = None
            st.session_state.params = None
            st.session_state.file_processed = False
            st.session_state.current_file_name = None
            st.rerun()

else:
    # Show placeholder if no image is uploaded
    if not st.session_state.processing:
        st.markdown("<div class='enhancement-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader-text' style='text-align: center;'>Upload an image to get started!</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Supported formats: JPG, JPEG, PNG</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Powered by AI Image Enhancement Technology</p>", unsafe_allow_html=True)
