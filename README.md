# 🎌 Lua — AI Anime Chatbot

### 🔗 Live demo: **https://lua-anime-chatbot.onrender.com**
_(Free host — the first load after it's been idle can take ~50 seconds to wake up.)_

An LLM-powered anime chatbot that actually knows its stuff — built by **Lukshya Supyal** ("Shadow_lu").

## What it does
- **Live, real data** from the [AniList](https://anilist.co) API + [Fandom](https://www.fandom.com) & Wikipedia — plot, episodes, ratings, characters, relationships, lore.
- **Grounded, no hallucination** — answers come from verified sources; if it's not there, Lua says so instead of making things up.
- **Real LLM brain** via OpenRouter / Groq with **automatic multi-provider failover** (never goes down when one hits its limit).
- **Remembers the conversation** per session, gives **clean answers** (tables & bullets), a friendly personality, live **poster cards**, and a soft, interactive UI with **light / cool / dark** modes.

## Run it locally
```bash
cd portfolio/anime-sensei
cp .env.example .env          # add a free API key (openrouter.ai/keys or console.groq.com)
pip install -r requirements.txt
python app.py                 # → http://127.0.0.1:5000
```

## Design
See [`DESIGN.md`](DESIGN.md) — the house style: calm, aesthetic, effortless.

---
_Built to a professional standard. Need an AI chatbot for your business? Let's talk._
