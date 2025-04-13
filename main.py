from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import cv2
import io
import os
import json
import base64
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

app = FastAPI()

vision_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=1024,
)

def load_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def calculate_brightness(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.mean[0]

def calculate_contrast(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.stddev[0]

def calculate_sharpness(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_image_with_langchain(image_data: bytes, metrics: Dict[str, float], image_format: str) -> Dict[str, Any]:
    """Send image to LangChain vision model for analysis and enhancement recommendations"""
    
    # Encode image to base64 with correct MIME type
    encoded_image = base64.b64encode(image_data).decode('ascii')
    mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
    image_url = f"data:{mime_type};base64,{encoded_image}"
    
    system_prompt = SystemMessage(
        content="You are an expert image analyst and photo enhancement specialist. Analyze the image and provide enhancement recommendations."
    )
    
    human_prompt = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": f"""
                Please analyze this image based on both its visual content and the following technical metrics:
                - Brightness: {metrics['brightness']:.2f} (0-255 scale)
                - Contrast: {metrics['contrast']:.2f} (standard deviation)
                - Sharpness: {metrics['sharpness']:.2f} (Laplacian variance)
                
                Provide the following in your response:
                1. A concise description (2-3 sentences) of what's in the image.
                2. Specific enhancement parameters as JSON values (no other text) for:
                   - brightness_factor: a float value multiplier (e.g., 1.2 for 20% increase)
                   - contrast_factor: a float value multiplier
                   - sharpness_factor: a float value multiplier
                
                Format your response exactly like this:
                DESCRIPTION: [your image description here]
                PARAMETERS: {{\"brightness_factor\": X.X, \"contrast_factor\": Y.Y, \"sharpness_factor\": Z.Z}}
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"
                }
            }
        ]
    )
    
    print("invoking................")
    response = vision_model.invoke([system_prompt, human_prompt])
    response_text = response.content

    description_match = re.search(r"DESCRIPTION:\s*(.*?)(?=\s*PARAMETERS:|$)", response_text, re.DOTALL)
    parameters_match = re.search(r"PARAMETERS:\s*(\{.*\})", response_text, re.DOTALL)
    
    if not description_match or not parameters_match:
        raise ValueError("Failed to parse model response correctly")
    
    description = description_match.group(1).strip()
    parameters_json = parameters_match.group(1).strip()
    
    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}. Using default parameters.")
        parameters = {
            "brightness_factor": 1.0,
            "contrast_factor": 1.0,
            "sharpness_factor": 1.0
        }
    
    return {
        "description": description,
        "parameters": parameters
    }

@app.get("/")
def home():
    return {"message": "LangChain Vision-Enhanced Image API"}

@app.get("/enhance-image")
def home():
    return {"message": "Enhanced Image API"}

@app.post("/enhance-image")
async def enhance_image(file: UploadFile = File(...)):
    try:
        print("*************got the request*****************")
        image_bytes = await file.read()
        image = load_image(image_bytes)
        
        # Determine image format
        image_format = "JPEG" if file.content_type in ["image/jpeg", "image/jpg"] else "PNG"
        
        metrics = {
            "brightness": calculate_brightness(image),
            "contrast": calculate_contrast(image),
            "sharpness": calculate_sharpness(image)
        }
        
        analysis = analyze_image_with_langchain(image_bytes, metrics, image_format)
        print("invoking doneeeeeeeeeeee................")

        enhancement_params = analysis["parameters"]
        enhanced_image = image
        
        # Apply enhancements
        enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(enhancement_params["brightness_factor"])
        enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(enhancement_params["contrast_factor"])
        enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(enhancement_params["sharpness_factor"])
        
        # Convert to base64
        orig_buffer = io.BytesIO()
        image.save(orig_buffer, format=image_format)
        orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode('ascii')
        
        enhanced_buffer = io.BytesIO()
        enhanced_image.save(enhanced_buffer, format=image_format)
        enhanced_b64 = base64.b64encode(enhanced_buffer.getvalue()).decode('ascii')
        
        return JSONResponse(content={
            "success": True,
            "original_image": orig_b64,
            "enhanced_image": enhanced_b64,
            "image_description": analysis["description"],
            "enhancement_parameters": enhancement_params,
            "original_metrics": metrics
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )