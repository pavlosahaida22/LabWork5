"""5.2.2, 5.2.9 Підключення мовної моделі (Gemini / Anthropic / OpenAI)."""
import os
from . import config

PROVIDER_LABELS = {
    "gemini": "Google Gemini (є безкоштовний рівень)",
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "none": "Без LLM (лише пошук)",
}
ENV_KEYS = {"gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"], "openai": ["OPENAI_API_KEY"]}


def get_api_key(provider: str, secrets=None) -> str | None:
    """Ключ береться зі змінних середовища або Streamlit secrets (ніколи не зберігається в коді)."""
    for name in ENV_KEYS.get(provider, []):
        if os.getenv(name):
            return os.getenv(name)
        if secrets is not None:
            try:
                if name in secrets:
                    return secrets[name]
            except Exception:
                pass
    return None


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None):
        self.provider, self.api_key = provider, api_key
        self.model = model or config.DEFAULT_MODELS[provider]

    def generate(self, system: str, user: str, max_tokens: int = 1500) -> str:
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                                   max_output_tokens=max_tokens))
            return (resp.text or "").strip()
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0.2,
                                          system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return (resp.choices[0].message.content or "").strip()
        raise ValueError(f"Невідомий провайдер: {self.provider}")
