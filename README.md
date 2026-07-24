# 🎌 Lua — AI Anime Chatbot (& friends)

A portfolio of clean, genuinely-working AI chatbots by **Lukshya Supyal** ("Shadow_lu").

## Projects

### 🌸 Lua — your anime buddy  ·  [`portfolio/anime-sensei`](portfolio/anime-sensei)
An LLM-powered anime chatbot that actually knows its stuff:
- **Live, real data** from the [AniList](https://anilist.co) API + [Fandom](https://www.fandom.com) & Wikipedia — plot, episodes, ratings, characters, relationships, lore.
- **Grounded, no hallucination** — answers come from verified sources; if it's not there, Lua says so instead of making things up.
- **Real LLM brain** via OpenRouter/Groq with **automatic multi-provider failover**.
- **Remembers the conversation** per session, with clean answers (tables, bullets), a friendly personality, poster cards, and a soft, interactive UI (light / cool / dark modes).

Run it locally:
```bash
cd portfolio/anime-sensei
cp .env.example .env          # add a free API key (openrouter.ai/keys or console.groq.com)
pip install -r requirements.txt
python app.py                 # → http://127.0.0.1:5000
```

### 💆 Serene Spa Concierge  ·  [`portfolio/serene-spa-concierge`](portfolio/serene-spa-concierge)
A calm, self-contained website chatbot demo — answers FAQs, quotes prices, books appointments, and captures leads. Pure HTML/JS (open `index.html`).

## Design
See [`DESIGN.md`](DESIGN.md) — the house style: calm, aesthetic, effortless. Every project follows it.

---
_Built to a professional standard. Need an AI chatbot for your business? Let's talk._
