"""
RAG (Retrieval-Augmented Generation) engine for ingredient knowledge.
Stores a knowledge base of ingredient-dish pairings, storage tips, and
waste reduction strategies. Uses FAISS for vector similarity search.
"""

import json
import numpy as np
import faiss
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()

_KNOWLEDGE_BASE = [
    {"id": "k001", "text": "Spinach and tomatoes can be combined in a palak-tomato sabzi or pasta. Use within 2 days of purchase. Store spinach in damp paper towels in the fridge.", "tags": ["spinach", "tomatoes", "vegetables"]},
    {"id": "k002", "text": "Mushrooms near expiry should be sautéed and frozen or made into a mushroom soup or risotto. They lose moisture rapidly.", "tags": ["mushrooms", "vegetables"]},
    {"id": "k003", "text": "Paneer can be frozen for up to 3 months. Use expiring paneer for paneer tikka, palak paneer, or matar paneer.", "tags": ["paneer", "dairy"]},
    {"id": "k004", "text": "Overripe bananas are ideal for banana bread, smoothies, or banana lassi. Never discard overripe bananas.", "tags": ["bananas", "fruits"]},
    {"id": "k005", "text": "Chicken near expiry must be cooked immediately. Use for butter chicken, biryani, or chicken soup. Never refreeze thawed chicken.", "tags": ["chicken", "meat"]},
    {"id": "k006", "text": "Bell peppers nearing expiry are excellent in stir-fries, stuffed peppers, or frozen after roasting. Store cut peppers in water.", "tags": ["bell peppers", "vegetables"]},
    {"id": "k007", "text": "Cream approaching expiry can be turned into whipped cream, used in pasta sauces, or frozen in ice cube trays for future use.", "tags": ["cream", "dairy"]},
    {"id": "k008", "text": "Bread about to expire makes excellent French toast, bread pudding, croutons, or breadcrumbs for coating.", "tags": ["bread", "bakery"]},
    {"id": "k009", "text": "Onions and potatoes should be stored separately — onions release ethylene that accelerates potato sprouting.", "tags": ["onions", "potatoes", "storage"]},
    {"id": "k010", "text": "Fish must be used within 24 hours of thawing. Ideal for fish curry, grilled preparations, or fish tacos.", "tags": ["fish", "seafood"]},
    {"id": "k011", "text": "Yogurt/curd past best-before can be used in marinades, raita, kadhi, or lassi. Slightly sour curd enhances flavor in many dishes.", "tags": ["curd", "yogurt", "dairy"]},
    {"id": "k012", "text": "Carrots nearing expiry work well in carrot halwa, soups, stir-fries, or can be blanched and frozen.", "tags": ["carrots", "vegetables"]},
    {"id": "k013", "text": "Cauliflower and broccoli should be blanched and frozen before they yellow. Excellent in aloo-gobi or cream soups.", "tags": ["cauliflower", "broccoli", "vegetables"]},
    {"id": "k014", "text": "Leftover rice should be used within 24 hours. Ideal for fried rice, khichdi, or rice pakoras. Never reheat rice more than once.", "tags": ["rice", "grains"]},
    {"id": "k015", "text": "Butter near expiry can be clarified into ghee for longer shelf life (months vs weeks).", "tags": ["butter", "dairy", "ghee"]},
    {"id": "k016", "text": "Tomatoes about to turn can be roasted and stored in olive oil for weeks, or pureed and frozen as base sauce.", "tags": ["tomatoes", "vegetables"]},
    {"id": "k017", "text": "Excess herbs like coriander and mint can be blended into chutneys, frozen in ice cubes, or used in herb-infused oils.", "tags": ["herbs", "coriander", "mint"]},
    {"id": "k018", "text": "Prawns/shrimp near expiry must be cooked within hours. Ideal for prawn masala, biryani, or grilled preparation.", "tags": ["prawns", "seafood"]},
    {"id": "k019", "text": "Orange juice and fruit juices near expiry can be used for marinades, dressings, cocktail mixers, or frozen as popsicles.", "tags": ["juice", "beverages"]},
    {"id": "k020", "text": "FIFO (First In, First Out) is the most important storage principle — always move older stock to the front of shelves.", "tags": ["storage", "fifo", "general"]},
    {"id": "k021", "text": "Temperature danger zone is 5°C to 60°C. Food left in this range for over 2 hours should be discarded.", "tags": ["food_safety", "temperature"]},
    {"id": "k022", "text": "Batch cooking reduces per-unit waste — cook large quantities and portion/freeze for later service.", "tags": ["batch_cooking", "strategy"]},
    {"id": "k023", "text": "Eggplant/brinjal darkens quickly once cut. Use lemon juice or salt water to prevent oxidation, or cook immediately.", "tags": ["eggplant", "vegetables"]},
    {"id": "k024", "text": "Lettuce and leafy greens can be revived in ice water if wilted. Store with a paper towel to absorb excess moisture.", "tags": ["lettuce", "vegetables", "storage"]},
    {"id": "k025", "text": "Zucchini near expiry is excellent in zucchini bread, fritters, stuffed preparations, or added to soups and stews.", "tags": ["zucchini", "vegetables"]},
]


class RAGRecommendationEngine:
    """
    FAISS-powered retrieval engine for ingredient knowledge.
    Given a query (e.g., list of ingredients), retrieves relevant
    storage tips, dish ideas, and waste reduction strategies.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = embedding_model
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.knowledge: List[Dict[str, Any]] = []
        self.is_built: bool = False
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _load_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading sentence encoder: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)

    def build_index(self, knowledge_base: Optional[List[Dict]] = None) -> None:
        """Build FAISS index from knowledge base."""
        self._load_encoder()
        self.knowledge = knowledge_base or _KNOWLEDGE_BASE

        texts = [k["text"] for k in self.knowledge]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.is_built = True
        logger.info(f"RAG index built with {len(self.knowledge)} knowledge entries")

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant knowledge entries for a query."""
        if not self.is_built:
            self.build_index()

        self._load_encoder()
        q_emb = self.encoder.encode([query], show_progress_bar=False).astype(np.float32)
        distances, indices = self.index.search(q_emb, min(top_k, len(self.knowledge)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.knowledge):
                entry = self.knowledge[idx].copy()
                entry["relevance_score"] = round(float(1 / (1 + dist)), 4)
                results.append(entry)
        return results

    def get_ingredient_knowledge(
        self, ingredients: List[str], top_k_per_ingredient: int = 2
    ) -> List[Dict[str, Any]]:
        """Get knowledge for a list of ingredients, deduplicated."""
        seen_ids = set()
        all_results = []

        for ingredient in ingredients[:10]:
            results = self.retrieve(ingredient, top_k=top_k_per_ingredient + 2)
            for r in results[:top_k_per_ingredient]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_results.append(r)

        return sorted(all_results, key=lambda x: x["relevance_score"], reverse=True)

    def save(self) -> None:
        faiss.write_index(self.index, str(self.model_dir / "rag_index.faiss"))
        joblib.dump(self.knowledge, self.model_dir / "rag_knowledge.pkl")
        logger.info("RAG index saved")

    def load(self) -> bool:
        try:
            self.index = faiss.read_index(str(self.model_dir / "rag_index.faiss"))
            self.knowledge = joblib.load(self.model_dir / "rag_knowledge.pkl")
            self.is_built = True
            logger.info(f"RAG index loaded ({len(self.knowledge)} entries)")
            return True
        except FileNotFoundError:
            logger.info("No saved RAG index — will build on first use")
            return False
        except Exception as e:
            logger.error(f"Error loading RAG index: {e}")
            return False
