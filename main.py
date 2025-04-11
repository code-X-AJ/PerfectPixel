from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import cv2
import io
import os
import json
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import init_chat_model
import base64

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

llm = init_chat_model(
    "gpt-4o-mini",  # Use GPT-4o instead of gpt-4o-mini for better quality
    model_provider="openai",
    temperature=0.3,  # Moved outside of model_kwargs
    max_tokens=1024,  # Moved outside of model_kwargs
    top_p=0.95,       # Moved outside of model_kwargs
)

# Define the output schema for LLM responses
response_schemas = [
    ResponseSchema(name="brightness_factor", type="float", description=r"Brightness multiplier (e.g., 1.2 for 20% increase)"),
    ResponseSchema(name="contrast_factor", type="float", description="Contrast multiplier"),
    ResponseSchema(name="sharpness_factor", type="float", description="Sharpness multiplier")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Create a LangChain prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert image enhancement assistant."),
    ("human", """
    Based on the following image metrics:
    - Brightness: {brightness} (0-255 scale)
    - Contrast: {contrast} (standard deviation)
    - Sharpness: {sharpness} (Laplacian variance)
    
    {format_instructions}
    """)
])

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

@app.get("/")
def home():
    return {"hey": "AJ"}

@app.get("/enhance-image")
def home2():
    return {"hey": "imageing"}


@app.post("/enhance-image")
async def enhance_image(file: UploadFile = File(...)):
    # Load and analyze image
    image_bytes = await file.read()
    image = load_image(image_bytes)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    sharpness = calculate_sharpness(image)

    print("**************got the request************************")

    # Generate LLM prompt and parse response
    try:
        formatted_prompt = prompt_template.format_messages(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            format_instructions=format_instructions
        )
        
        llm_response = llm.invoke(formatted_prompt)
        params = output_parser.parse(llm_response.content)
        print("..................", llm_response.content)
        print("..................", params)
    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}


    # # Apply enhancements
    newImage = ImageEnhance.Brightness(image).enhance(params["brightness_factor"])
    newImage = ImageEnhance.Contrast(newImage).enhance(params["contrast_factor"])
    newImage = ImageEnhance.Sharpness(newImage).enhance(params["sharpness_factor"])

    img_byte_arr = io.BytesIO()
    image_format = "JPEG" if file.content_type == "image/jpeg" else "PNG"
    newImage.save(img_byte_arr, format=image_format)
    img_byte_arr = img_byte_arr.getvalue()
    encoded_img = base64.b64encode(img_byte_arr).decode('ascii')
    
    return JSONResponse(content={
        "enhanced_image": encoded_img,
        "params": params,
        "original_metrics": {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
        },
    })
