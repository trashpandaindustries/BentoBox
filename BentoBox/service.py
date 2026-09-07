import os
import bentoml
from PIL.Image import Image
import numpy as np
from typing import Dict
from typing import List
from pydantic import Field
from fastapi import FastAPI

MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
SERVICE_MEMORY = os.getenv("CLIP_MEMORY", "4Gi")
SERVICE_PORT = int(os.getenv("CLIP_PORT", "3000"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CLIP_CORS_ORIGINS", "http://localhost:8080").split(",")
    if origin.strip()
]
FORCE_DEVICE = os.getenv("CLIP_FORCE_DEVICE")  # "cpu", "cuda", or unset for auto-detect

runtime_image = bentoml.images.Image(
    python_version="3.11"
).requirements_file("requirements.txt")

system_app = FastAPI()


@bentoml.service(
    image=runtime_image,
    resources={"memory": SERVICE_MEMORY},
    http={
        "port": SERVICE_PORT,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": CORS_ORIGINS,
            "access_control_allow_methods": ["GET", "OPTIONS", "POST"],
            "access_control_allow_headers": ["*"],
        }
    },
)
@bentoml.asgi_app(system_app)
class CLIP:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)
    
    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if FORCE_DEVICE:
            self.device = FORCE_DEVICE
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(self.hf_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.hf_model)
        self.logit_scale = self.clip_model.logit_scale.item() if self.clip_model.logit_scale.item() else 4.60517
        print("Model clip loaded", "device:", self.device, "model:", MODEL_ID)

    @bentoml.api(batchable=True)
    async def encode_image(self, items: List[Image]) -> np.ndarray:
        '''
        generate the embeddings of the images
        '''
        # Resizing is intentionally left to CLIPProcessor, which resizes/crops
        # to whatever size the loaded model's own preprocessor config specifies.
        # Do not hardcode target dimensions here — CLIP_MODEL_ID can point at
        # variants with different native input resolutions.
        inputs = self.processor(images=items, return_tensors="pt", padding=True).to(self.device)
        image_embeddings = self.clip_model.get_image_features(**inputs)
        return image_embeddings.cpu().detach().numpy()
    

    @bentoml.api(batchable=True)
    async def encode_text(self, items: List[str]) -> np.ndarray:
        '''
        generate the embeddings of the texts
        '''
        inputs = self.processor(text=items, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings.cpu().detach().numpy()
    
    @bentoml.api
    async def rank(self, queries: List[Image], candidates : List[str] = Field(["picture of a dog", "picture of a cat"], description="list of description candidates")) -> Dict[str, List[List[float]]]:
        '''
        return the similarity between the query images and the candidate texts
        '''

        # Encode embeddings
        query_embeds = await self.encode_image(queries)
        candidate_embeds = await self.encode_text(candidates)

        # Compute cosine similarities
        cosine_similarities = self.cosine_similarity(query_embeds, candidate_embeds)
        logit_scale = np.exp(self.logit_scale)
        # Compute softmax scores
        prob_scores = self.softmax(logit_scale * cosine_similarities)
        return {
            "probabilities": prob_scores.tolist(),
            "cosine_similarities" : cosine_similarities.tolist(),
        }
        
    @staticmethod
    def cosine_similarity(query_embeds, candidates_embeds):
        # Normalize each embedding to a unit vector
        query_embeds /= np.linalg.norm(query_embeds, axis=1, keepdims=True)
        candidates_embeds /= np.linalg.norm(candidates_embeds, axis=1, keepdims=True)

        # Compute cosine similarity
        cosine_similarities = np.matmul(query_embeds, candidates_embeds.T)

        return cosine_similarities
    
    @staticmethod
    def softmax(scores):
        # Compute softmax scores (probabilities)
        exp_scores = np.exp(
            scores - np.max(scores, axis=-1, keepdims=True)
        )  # Subtract max for numerical stability
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


#add for reasons
    @system_app.get("/model")
    async def model_info(self) -> Dict:
        return {
            "model_id": MODEL_ID,
            "embedding_dimension": self.clip_model.config.projection_dim,  # 512, but derived not hardcoded
            "device": self.device,
            "model_revision": getattr(self.hf_model, "revision", None) or "unknown",
        }

    @system_app.get("/health")
    async def health(self) -> Dict:
        try:
            _ = self.clip_model.get_text_features(
                **self.processor(text=["health check"], return_tensors="pt", padding=True).to(self.device)
            )
            return {"status": "ok", "device": self.device}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
