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
        """Load the FastVLM model"""
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, model_name, device=self.device
        )
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, prompt: str, image: Image.Image, temperature: float = 0.2, max_tokens: int = 256) -> str:
        """Run inference on image and prompt"""
        start_time = time.time()
        
        # Construct prompt with image tokens
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        # Use conversation template
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(torch.device(self.device))

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True
            )
        
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()        # Remove the original prompt from response
        if prompt_formatted in response:
            response = response.replace(prompt_formatted, "").strip()
        
        processing_time = time.time() - start_time
        return response, processing_time


# Global model instance
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent
    model_path = "checkpoints/llava-fastvithd_0.5b_stage3"
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found. Please update the path.")
    else:
        agent = FastVLMAgent(model_path)
    yield
app = FastAPI(
    title="FastVLM Web Agent",
    description="FastVLM Vision Language Model API for web agents",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
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
    """Alternative endpoint that accepts image URL instead of base64"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)