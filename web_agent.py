import os
import io
import base64
import time
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from contextlib import asynccontextmanager
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class QueryRequest(BaseModel):
    prompt: str
    image_base64: str
    temperature: float = 0.2
    max_tokens: int = 256


class QueryResponse(BaseModel):
    response: str
    processing_time: float


class TextRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 512


class TextLLMAgent:
    def __init__(self, model_path: str, device: str = "mps"):
        self.model_path = os.path.expanduser(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load text-only LLM"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        print(f"Text LLM loaded successfully on {self.device}")

    def predict(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        processing_time = time.time() - start_time
        return response, processing_time


class FastVLMAgent:
    def __init__(self, model_path: str, device: str = "mps"):
        self.model_path = os.path.expanduser(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.load_model()
    
    def load_model(self):
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, model_name, device=self.device
        )
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, prompt: str, image: Image.Image, temperature: float = 0.2, max_tokens: int = 256) -> str:
        start_time = time.time()
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(torch.device(self.device))

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=0.9 if temperature > 0 else 1.0,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True
            )
        
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()        # Remove the original prompt from response
        if prompt_formatted in response:
            response = response.replace(prompt_formatted, "").strip()
        
        processing_time = time.time() - start_time
        return response, processing_time
agent = None
text_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, text_llm
    model_path = os.getenv("FASTVLM_MODEL_PATH", "checkpoints/llava-fastvithd_0.5b_stage3")
    device = os.getenv("FASTVLM_DEVICE", "mps")
    if not os.path.exists(os.path.expanduser(model_path)):
        print(f"Warning: Model path {model_path} not found. Please set FASTVLM_MODEL_PATH.")
    else:
        agent = FastVLMAgent(model_path, device=device)

    text_model_path = os.getenv("TEXT_LLM_PATH", "checkpoints/Qwen2.5-1.5B-Instruct")
    if os.path.exists(os.path.expanduser(text_model_path)):
        print("Loading Text LLM at startup (may take 1-2 minutes)...")
        text_llm = TextLLMAgent(text_model_path, device=device)
    else:
        print(f"Warning: Text LLM path {text_model_path} not found. Hybrid mode disabled.")

    yield
app = FastAPI(
    title="FastVLM Web Agent",
    description="FastVLM Vision Language Model API for web agents",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": agent is not None,
        "device": agent.device if agent else None
    }


@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        response, processing_time = agent.predict(
            request.prompt, 
            image, 
            request.temperature, 
            request.max_tokens
        )
        
        return QueryResponse(response=response, processing_time=processing_time)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/query_url")
async def query_from_url(prompt: str, image_url: str, temperature: float = 0.2, max_tokens: int = 256):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import requests
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')

        result, processing_time = agent.predict(prompt, image, temperature, max_tokens)

        return QueryResponse(response=result, processing_time=processing_time)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image URL: {str(e)}")


@app.post("/query_text", response_model=QueryResponse)
async def query_text_model(request: TextRequest):
    if text_llm is None:
        raise HTTPException(status_code=503, detail="Text LLM not loaded")

    try:
        response, processing_time = text_llm.predict(
            request.prompt,
            request.temperature,
            request.max_tokens
        )

        return QueryResponse(response=response, processing_time=processing_time)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
