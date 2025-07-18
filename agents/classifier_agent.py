

import os
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
from core.llm_manager import LLMManager, ModelType  # ✅ Import ModelType enum

logging.basicConfig(level=logging.INFO)

class TopicClassifierAgent:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 llm_manager: Optional[LLMManager] = None,
                 model_type: ModelType = ModelType.LLAMA_70B):  # ✅ Use ModelType enum
        self.embedder = SentenceTransformer(model_name)
        self.llm_manager = llm_manager
        self.similarity_threshold = 0.65
        self.model_type = model_type  # ✅ Now properly typed

    def classify(self, paper: Dict) -> Dict:
        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        topics = paper.get("topics", [])

        if not topics:
            logging.warning(f"[{paper_id}] No topics provided.")
            return {
                "paper_id": paper_id,
                "predicted_topic": None,
                "confidence": 0.0
            }

        # Create a combined text input
        text_to_embed = f"{title} {abstract}"
        paper_embedding = self.embedder.encode(text_to_embed, convert_to_tensor=True)
        topic_embeddings = self.embedder.encode(topics, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(paper_embedding, topic_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_topic = topics[best_idx]

        logging.info(f"[{paper_id}] Best topic: {best_topic} (score={best_score:.2f})")

        # Fallback to LLM if similarity is low
        if best_score < self.similarity_threshold and self.llm_manager:
            logging.info(f"[{paper_id}] Confidence below threshold. Falling back to LLM...")
            
            # ✅ Use the proper classify_topic method from LLMManager
            prompt = f"Classify the paper with the following title and abstract:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nChoose the best topic from this list."
            
            best_topic = self.llm_manager.classify_topic(prompt, topics, self.model_type)
            best_score = 0.5  # Fallback confidence score

        return {
            "paper_id": paper_id,
            "predicted_topic": best_topic,
            "confidence": round(best_score, 4)
        }
