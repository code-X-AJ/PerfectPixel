---

# 🎨 PerfectPixel  
An AI-powered image enhancement app that intelligently analyzes and improves your images using LLM-generated adjustments. Automatically fixes brightness, contrast, and more — no filters, just smart processing.

---

## 🚀 Features  
- 🖼️ Upload any image (dark, blurry, low contrast, etc.)  
- ✨ Dynamically adjusts brightness, contrast, and sharpness  
- 📋 Returns enhanced image, format, and size  
- 💡 Simple and clean frontend using Streamlit  
- ⚙️ FastAPI backend handles processing and enhancement  

---

## 🛠️ Tech Stack  
- **Frontend**: Streamlit  
- **Backend**: FastAPI  
- **AI**: OpenAI (LLM)  
- **Image Processing**: Pillow  
- **Others**: Python, io, base64, dotenv  

---

## 🔧 Getting Started  

### 1. Clone the Repo  
```bash
git clone https://github.com/your-username/perfectpixel.git  
cd perfectpixel
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Add .env File  
Create a `.env` file and include:  
```env
OPENAI_API_KEY=your_openai_key
```

### 4. Run FastAPI Backend  
```bash
cd backend
uvicorn main:app --reload
```

### 5. Run Streamlit Frontend  
```bash
cd frontend
streamlit run streamlit_app.py
```

---

## 🤝 Contributions  
Pull requests are welcome! For major changes, please open an issue to discuss your ideas.

---
