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

# Default zero-shot category list for /classify, overridable per-request.
CLIP_CATEGORIES = [
    c.strip()
    for c in os.getenv(
        "CLIP_CATEGORIES",
        "person,animal,vehicle,building,food,landscape,document,artwork"
    ).split(",")
    if c.strip()
]
# Template used to turn a bare category label into a CLIP-friendly prompt.
# Must contain a single "{}" placeholder for the category.
CLIP_CATEGORY_TEMPLATE = os.getenv("CLIP_CATEGORY_TEMPLATE", "a photo of a {}")

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
        # Flipped true once the model is loaded and usable; used by /healthz
        # so liveness checks don't need to run inference to be meaningful.
        self.ready = True
        print("Model clip loaded", "device:", self.device, "model:", MODEL_ID)

    # -----------------------------------------------------------------
    # Public, HTTP-facing endpoints (@bentoml.api). These should only ever
    # be called from outside the class (over HTTP). Calling one of these
    # directly from another method on this class routes through BentoML's
    # internal client/proxy layer and can break (e.g. it tries to
    # re-serialize PIL Images as multipart file uploads). Cross-method
    # calls within the class must go through the plain "_impl" methods
    # below instead.
    # -----------------------------------------------------------------

    @bentoml.api(batchable=True)
    async def encode_image(self, items: List[Image]) -> np.ndarray:
        '''
        generate the embeddings of the images
        '''
        return await self._encode_image_impl(items)

    @bentoml.api(batchable=True)
    async def encode_text(self, items: List[str]) -> np.ndarray:
        '''
        generate the embeddings of the texts
        '''
        return await self._encode_text_impl(items)

    @bentoml.api
    async def rank(self, queries: List[Image], candidates : List[str] = Field(["picture of a dog", "picture of a cat"], description="list of description candidates")) -> Dict[str, List[List[float]]]:
        '''
        return the similarity between the query images and the candidate texts
        '''
        return await self._rank_impl(queries, candidates)

    @bentoml.api
    async def classify(
        self,
        items: List[Image],
        categories: List[str] = Field(
            default_factory=list,
            description="Category labels to classify against. Leave empty to use the server's CLIP_CATEGORIES env config; pass your own list to override per-request.",
        ),
        template: str = Field(
            "",
            description="Prompt template used to turn each category label into a CLIP text prompt. Must contain one '{}' placeholder. Leave empty to use the server's CLIP_CATEGORY_TEMPLATE env config.",
        ),
        top_k: int = Field(3, description="Number of top-scoring categories to return per image."),
    ) -> Dict[str, List]:
        '''
        Zero-shot classify images against a category list (env-configured default,
        or a caller-supplied list/template for one-off use).
        '''
        categories = categories or list(CLIP_CATEGORIES)
        template = template or CLIP_CATEGORY_TEMPLATE

        if "{}" not in template:
            raise ValueError("template must contain a single '{}' placeholder for the category label")
        if not categories:
            raise ValueError("categories must be a non-empty list")

        prompts = [template.format(c) for c in categories]
        result = await self._rank_impl(items, prompts)
        probs = np.array(result["probabilities"])
        k = min(top_k, len(categories))

        predictions = [
            [
                {"label": categories[i], "score": float(row[i])}
                for i in np.argsort(row)[::-1][:k]
            ]
            for row in probs
        ]
        return {"predictions": predictions}

    # -----------------------------------------------------------------
    # Internal implementations. Plain async methods (no @bentoml.api),
    # safe to call directly from other methods on this class.
    # -----------------------------------------------------------------

    async def _encode_image_impl(self, items: List[Image]) -> np.ndarray:
        # Resizing is intentionally left to CLIPProcessor, which resizes/crops
        # to whatever size the loaded model's own preprocessor config specifies.
        # Do not hardcode target dimensions here — CLIP_MODEL_ID can point at
        # variants with different native input resolutions.
        inputs = self.processor(images=items, return_tensors="pt", padding=True).to(self.device)
        image_embeddings = self.clip_model.get_image_features(**inputs)
        return image_embeddings.cpu().detach().numpy()

    async def _encode_text_impl(self, items: List[str]) -> np.ndarray:
        inputs = self.processor(text=items, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings.cpu().detach().numpy()

    async def _rank_impl(self, queries: List[Image], candidates: List[str]) -> Dict[str, List[List[float]]]:
        # Encode embeddings
        query_embeds = await self._encode_image_impl(queries)
        candidate_embeds = await self._encode_text_impl(candidates)

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

    @system_app.get("/healthz")
    async def healthz(self) -> Dict:
        '''
        Cheap liveness check: process is up and the model finished loading.
        No inference is run — safe to poll frequently (e.g. every few seconds
        from a load balancer or orchestrator).
        '''
        if getattr(self, "ready", False):
            return {"status": "ok", "device": self.device}
        return {"status": "loading"}

    @system_app.get("/health")
    async def health(self) -> Dict:
        '''
        Readiness check: runs a real inference pass to confirm the model is
        actually usable, not just loaded. Heavier than /healthz — use for
        readiness probes, not tight polling loops.
        '''
        try:
            _ = self.clip_model.get_text_features(
                **self.processor(text=["health check"], return_tensors="pt", padding=True).to(self.device)
            )
            return {"status": "ok", "device": self.device}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
