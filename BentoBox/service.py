import os
import unittest.mock
import bentoml
from PIL.Image import Image
import numpy as np
from typing import Dict
from typing import List
from typing import Optional
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

# Florence-2 is opt-in: it's only needed at ingestion time, not on the hot path
# that CLIP serves routinely, so it's kept out of memory entirely unless enabled.
FLORENCE_ENABLED = os.getenv("FLORENCE_ENABLED", "false").strip().lower() == "true"
FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-base")
# Strongly recommended once FLORENCE_ENABLED=true: Florence-2 requires
# trust_remote_code=True, which executes code shipped in the HF repo, not just
# weights. Pin this to a known-good commit hash rather than trusting `main`.
FLORENCE_MODEL_REVISION = os.getenv("FLORENCE_MODEL_REVISION") or None

CAPTION_TASKS = {
    "brief": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "very_detailed": "<MORE_DETAILED_CAPTION>",
}

runtime_image = bentoml.images.Image(
    python_version="3.11"
).requirements_file("requirements.txt")

if FLORENCE_ENABLED:
    from transformers.dynamic_module_utils import get_imports as _ORIGINAL_GET_IMPORTS

system_app = FastAPI()


def _patched_get_imports(filename):
    """
    Florence-2's remote modeling code unconditionally imports flash_attn even
    when it isn't needed (e.g. CPU inference, or attn_implementation="sdpa"),
    which raises ImportError on any machine without flash-attn installed.
    This is a well-known community workaround: strip flash_attn out of the
    detected imports for that one file before transformers tries to load it.

    _ORIGINAL_GET_IMPORTS is captured at module load time, before the
    unittest.mock.patch call below replaces the module attribute — resolving
    it lazily inside this function instead would just call this same patched
    function again (infinite recursion).
    """
    imports = _ORIGINAL_GET_IMPORTS(filename)
    if str(filename).endswith("modeling_florence2.py"):
        imports = [i for i in imports if i != "flash_attn"]
    return imports


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

    # Only declared as a BentoML model dependency when enabled - if
    # FLORENCE_ENABLED is false this attribute doesn't exist, so BentoML never
    # attempts to resolve or download it.
    if FLORENCE_ENABLED:
        florence_hf_model = bentoml.models.HuggingFaceModel(
            FLORENCE_MODEL_ID, revision=FLORENCE_MODEL_REVISION
        )

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

        self.florence_enabled = FLORENCE_ENABLED
        self.florence_model_id = FLORENCE_MODEL_ID
        if self.florence_enabled:
            self._load_florence()

    def _load_florence(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        with unittest.mock.patch(
            "transformers.dynamic_module_utils.get_imports", _patched_get_imports
        ):
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                self.florence_hf_model,
                trust_remote_code=True,
                attn_implementation="sdpa",
            ).to(self.device)
            self.florence_processor = AutoProcessor.from_pretrained(
                self.florence_hf_model, trust_remote_code=True
            )
        print(
            "Model florence loaded", "device:", self.device,
            "model:", self.florence_model_id,
            "revision:", FLORENCE_MODEL_REVISION or "unpinned (main)",
        )

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

    @bentoml.api
    async def caption(
        self,
        image: Image,
        detail: str = Field(
            "detailed",
            description="Caption granularity: 'brief', 'detailed', or 'very_detailed'",
        ),
    ) -> Dict[str, str]:
        '''
        generate a natural-language caption for an image via Florence-2.
        Intended for ingestion-time use, not routine/hot-path calls.
        '''
        return await self._caption_impl(image, detail)

    async def _caption_impl(self, image: Image, detail: str) -> Dict[str, str]:
        if not self.florence_enabled:
            raise bentoml.exceptions.BentoMLException(
                "Florence-2 captioning is disabled on this instance. "
                "Set FLORENCE_ENABLED=true and restart to enable it."
            )

        task_prompt = CAPTION_TASKS.get(detail, CAPTION_TASKS["detailed"])

        inputs = self.florence_processor(
            text=task_prompt, images=image, return_tensors="pt"
        ).to(self.device)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return {
            "detail": detail,
            "task_prompt": task_prompt,
            "caption": parsed[task_prompt],
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
            "florence_enabled": self.florence_enabled,
            "florence_model_id": self.florence_model_id if self.florence_enabled else None,
            "florence_model_revision": (
                (FLORENCE_MODEL_REVISION or "unpinned (main)")
                if self.florence_enabled else None
            ),
        }

    @system_app.get("/health")
    async def health(self) -> Dict:
        try:
            _ = self.clip_model.get_text_features(
                **self.processor(text=["health check"], return_tensors="pt", padding=True).to(self.device)
            )
            status = {"status": "ok", "device": self.device, "florence_enabled": self.florence_enabled}
            if self.florence_enabled:
                status["florence_status"] = "loaded" if hasattr(self, "florence_model") else "not_loaded"
            return status
        except Exception as e:
            return {"status": "error", "detail": str(e)}
