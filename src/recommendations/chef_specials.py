"""
AI-powered Chef Specials generator.
Supports multiple LLM backends: Groq (free), Google Gemini (free), Anthropic Claude, Ollama (local).
Falls back to rule-based suggestions if no provider is configured.
"""

import json
from typing import List, Optional, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()


_SYSTEM_PROMPT = """You are an expert executive chef and food waste consultant for a restaurant.
Your role is to:
1. Create innovative, practical Chef Special dishes using ingredients nearing their expiry.
2. Maximize ingredient utilization to reduce food waste.
3. Ensure dishes are commercially viable for a restaurant menu.
4. Provide actionable storage and prep tips.

Always respond with valid JSON exactly matching the requested schema. Be specific, creative, and practical."""


def _build_user_prompt(
    ingredients: List[str],
    cuisine_preference: Optional[str],
    dietary_restrictions: Optional[List[str]],
    n_suggestions: int,
    context: Optional[str] = None,
) -> str:
    diet_str = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
    cuisine_str = cuisine_preference or "any cuisine"
    ctx_str = f"\nAdditional context: {context}" if context else ""

    return f"""I have the following ingredients nearing expiry that need to be used urgently:
{chr(10).join(f"- {ing}" for ing in ingredients)}

Cuisine preference: {cuisine_str}
Dietary restrictions: {diet_str}
{ctx_str}

Please generate {n_suggestions} Chef Special dish suggestions. For each dish provide:
1. A creative dish name
2. Which ingredients it uses (from the list above)
3. Brief description (2-3 sentences)
4. Estimated prep time
5. Why this dish helps reduce waste
6. Storage tip for remaining ingredients

Respond ONLY with this JSON structure:
{{
  "chef_specials": [
    {{
      "name": "Dish Name",
      "ingredients_used": ["ingredient1", "ingredient2"],
      "description": "...",
      "prep_time_minutes": 30,
      "waste_reduction_rationale": "...",
      "storage_tip": "...",
      "urgency": "use_today|use_within_2_days|use_this_week"
    }}
  ],
  "general_recommendation": "Overall strategy for these ingredients...",
  "estimated_waste_reduction_pct": 75
}}"""


def _extract_json(raw: str) -> dict:
    """Extract JSON from response, handling markdown fences."""
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    result = json.loads(raw)
    result.setdefault("chef_specials", [])
    result.setdefault("general_recommendation", "Use highlighted ingredients as soon as possible.")
    result.setdefault("estimated_waste_reduction_pct", 60)
    return result


class ChefSpecialsEngine:
    """
    Multi-provider LLM engine for Chef Specials.
    Provider priority: Groq → Gemini → Claude → Ollama → fallback
    """

    def __init__(self):
        self.provider, self.client = self._init_provider()
        logger.info(f"Chef Specials engine using provider: {self.provider}")

    def _init_provider(self):
        # 1. Groq (free, fast — sign up at console.groq.com)
        groq_key = settings.groq_api_key
        if groq_key:
            try:
                from groq import Groq
                return "groq", Groq(api_key=groq_key)
            except ImportError:
                logger.warning("groq package not installed, trying next provider")

        # 2. Google Gemini (free tier — get key at aistudio.google.com)
        gemini_key = settings.gemini_api_key
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                return "gemini", genai
            except ImportError:
                logger.warning("google-generativeai package not installed, trying next provider")

        # 3. Anthropic Claude (paid)
        anthropic_key = settings.anthropic_api_key
        if anthropic_key:
            try:
                import anthropic
                return "anthropic", anthropic.Anthropic(api_key=anthropic_key)
            except ImportError:
                logger.warning("anthropic package not installed")

        # 4. Ollama (local, no key needed — install from ollama.ai)
        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                return "ollama", httpx.Client(base_url="http://localhost:11434")
        except Exception:
            pass

        # 5. No provider — rule-based fallback
        return "fallback", None

    def generate(
        self,
        ingredients: List[str],
        cuisine_preference: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        n_suggestions: int = 3,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not ingredients:
            return {"chef_specials": [], "general_recommendation": "No ingredients provided.", "estimated_waste_reduction_pct": 0}

        ingredients = [str(i).strip() for i in ingredients[:15]]

        if self.provider == "fallback":
            return self._fallback_suggestions(ingredients, n_suggestions)

        prompt = _build_user_prompt(ingredients, cuisine_preference, dietary_restrictions, n_suggestions, context)

        try:
            if self.provider == "groq":
                return self._call_groq(prompt)
            elif self.provider == "gemini":
                return self._call_gemini(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            elif self.provider == "ollama":
                return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}")
            return self._fallback_suggestions(ingredients, n_suggestions)

    def _call_groq(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.7,
        )
        raw = response.choices[0].message.content
        result = _extract_json(raw)
        logger.info(f"Groq generated {len(result['chef_specials'])} chef specials")
        return result

    def _call_gemini(self, prompt: str) -> dict:
        model = self.client.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        result = _extract_json(response.text)
        logger.info(f"Gemini generated {len(result['chef_specials'])} chef specials")
        return result

    def _call_anthropic(self, prompt: str) -> dict:
        import anthropic
        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text)
        logger.info(f"Claude generated {len(result['chef_specials'])} chef specials")
        return result

    def _call_ollama(self, prompt: str) -> dict:
        import json as _json
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        r = self.client.post("/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        result = _extract_json(raw)
        logger.info(f"Ollama generated {len(result['chef_specials'])} chef specials")
        return result

    def _fallback_suggestions(self, ingredients: List[str], n_suggestions: int) -> Dict[str, Any]:
        logger.info("Using rule-based chef specials fallback")
        templates = [
            ("Stir Fry", "A quick high-heat stir fry preserving nutrients while using multiple expiring ingredients.", 20, "use_today"),
            ("Soup", "A hearty soup incorporating multiple expiring ingredients with minimal waste.", 35, "use_within_2_days"),
            ("Curry", "A flavorful curry that uses expiring ingredients while delivering bold taste.", 45, "use_within_2_days"),
            ("Bowl", "A customizable bowl combining multiple ingredients into a balanced meal.", 25, "use_this_week"),
        ]
        specials = []
        for i, (style, desc, prep, urgency) in enumerate(templates[:n_suggestions]):
            ing = ingredients[i % len(ingredients)]
            specials.append({
                "name": f"{ing} {style}",
                "ingredients_used": ingredients[:min(3, len(ingredients))],
                "description": desc,
                "prep_time_minutes": prep,
                "waste_reduction_rationale": f"Uses {ing} and other expiring items before they spoil.",
                "storage_tip": "Store leftovers in airtight containers at 4°C for up to 2 days.",
                "urgency": urgency,
            })
        return {
            "chef_specials": specials,
            "general_recommendation": "Prioritize perishable ingredients today. Consider batch cooking to extend usable life via freezing.",
            "estimated_waste_reduction_pct": 55,
            "_source": "fallback",
        }

    def generate_inventory_advice(self, high_risk_items: list, overstock_items: list) -> str:
        if self.provider == "fallback":
            return _fallback_inventory_advice(high_risk_items, overstock_items)

        items_summary = "\n".join(
            f"- {it.get('ingredient_name','?')}: {it.get('quantity',0):.1f} {it.get('unit','kg')}, "
            f"expires in {it.get('days_to_expiry',0)} days, risk={it.get('risk_level_pred','?')}"
            for it in high_risk_items[:10]
        )
        overstock_summary = "\n".join(
            f"- {it.get('ingredient_name','?')}: excess {it.get('overstock_quantity',0):.1f}"
            for it in overstock_items[:5]
        )
        prompt = f"""As a kitchen operations expert, provide 5 specific actionable recommendations for:

HIGH RISK (expiring soon):
{items_summary or 'None'}

OVERSTOCK:
{overstock_summary or 'None'}

Format as a numbered list. Be specific and prioritize by urgency."""

        try:
            if self.provider == "groq":
                r = self.client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                )
                return r.choices[0].message.content.strip()
            elif self.provider == "gemini":
                m = self.client.GenerativeModel(settings.gemini_model)
                return m.generate_content(prompt).text.strip()
            elif self.provider == "anthropic":
                r = self.client.messages.create(
                    model=settings.claude_model, max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text.strip()
            elif self.provider == "ollama":
                resp = self.client.post("/api/chat", json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }, timeout=60)
                return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Inventory advice failed: {e}")
        return _fallback_inventory_advice(high_risk_items, overstock_items)


def _fallback_inventory_advice(high_risk: list, overstock: list) -> str:
    lines = ["Inventory Recommendations:"]
    if high_risk:
        lines.append(f"1. Use {high_risk[0].get('ingredient_name','items')} immediately — critical waste risk.")
    if len(high_risk) > 1:
        lines.append(f"2. Plan menu specials featuring {len(high_risk)} expiring items.")
    if overstock:
        lines.append(f"3. Pause ordering {overstock[0].get('ingredient_name','items')} — exceeds forecast demand.")
    lines.append("4. Review FIFO rotation — ensure older stock is used first.")
    lines.append("5. Set daily consumption targets for high-risk categories.")
    return "\n".join(lines)
