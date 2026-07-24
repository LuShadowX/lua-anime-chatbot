"""
Anime Sensei — an LLM-powered anime specialist chatbot.

How it works (real tech, not hardcoded):
  • LIVE data + poster art from the AniList GraphQL API (free, no key, reliable)
  • A real LLM brain via OpenRouter (free models) for natural understanding + replies
  • Retrieval-Augmented: verified AniList data is injected as context so facts
    (episodes, scores, studios, characters) are grounded, not hallucinated
  • Conversation memory: follow-ups like "how many episodes?" know the anime
  • Shows a poster card for the anime under discussion
  • Stays an anime specialist, but answers warmly — never begs the user for data

Setup:
  cp .env.example .env         # then paste your OpenRouter key into .env
  pip install -r requirements.txt
  python app.py                # open http://127.0.0.1:5000
"""

import os
import re
import time
import requests
from flask import Flask, request, jsonify, session, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "anime-sensei-demo-key")

# Per-tab conversations, IN MEMORY ONLY. The client sends a fresh id each page load,
# so closing the tab (or restarting the server) makes the conversation vanish — and
# two tabs get two separate ids, so their contexts never bleed into each other.
CONVOS = {}


def _convo(cid):
    c = CONVOS.get(cid)
    if c is None:
        c = {"anime": None, "history": [], "persona": None}
        CONVOS[cid] = c
        if len(CONVOS) > 300:                 # simple cap so memory can't grow forever
            for k in list(CONVOS)[:60]:
                CONVOS.pop(k, None)
    return c

ANILIST_URL = "https://graphql.anilist.co"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OR_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

_cache = {}

BASE_PERSONA = (
    "You are Lua, a fun, upbeat anime-loving friend chatting with the user.\n"
    "ACCURACY (MOST IMPORTANT): For any fact — episodes, scores, studios, characters, relationships, "
    "who-ends-up-with-who, deaths, plot points — rely ONLY on the VERIFIED DATA and WEB CONTEXT blocks "
    "below. If the WEB CONTEXT is given, treat it as the truth and base your answer on it. If the "
    "answer is NOT in the provided data/context, do NOT guess or invent — say you're not 100% sure and "
    "offer to dig deeper. NEVER make up character details, ships, or events.\n"
    "STYLE RULES:\n"
    "1) Keep replies SHORT — about 2-3 sentences, or a small list. Never a wall of text.\n"
    "2) Use SIMPLE, everyday words anyone can understand.\n"
    "3) Format cleanly: markdown table (| Label | Value |) for stats/comparisons; short '- ' bullets "
    "for lists; **bold** for key terms.\n"
    "4) Answer ONLY what was asked.\n"
    "5) BE A GOOD LISTENER: reply to what the user ACTUALLY said and answer their real question. Do NOT "
    "invent stories or romance scenarios, do NOT role-play, and do NOT force comparisons between the "
    "user's messages and anime plots. Your persona flavors your TONE only — never the facts or topic.\n"
    "Be warm and friendly, show light interest in the user, use emojis sparingly. Stay on anime; "
    "playfully steer back if off-topic. Don't claim you can't browse — you have live data."
)
HARU_VOICE = (" YOUR VIBE: like Haru from 'Trillion Game' — upbeat, confident, a fun hype-man buddy. "
              "Keep it lively, but always ANSWER THE ACTUAL QUESTION clearly and first.")
_MALE = re.compile(r"\b(guy|boy|male|man|dude|bro|gentleman|masculine|he|him|his)\b", re.I)
_FEMALE = re.compile(r"\b(girl|female|woman|lady|gal|feminine|she|her|hers)\b", re.I)


def detect_persona(msg):
    if _FEMALE.search(msg):
        return "holo"
    if _MALE.search(msg):
        return "haru"
    return None

# --------------------------------------------------------------------------
# AniList (GraphQL) data client
# --------------------------------------------------------------------------
_QUERY = """
query ($s: String) {
  Media(search: $s, type: ANIME) {
    title { romaji english }
    description(asHtml: false)
    episodes
    status
    averageScore
    seasonYear
    genres
    coverImage { large }
    studios(isMain: true) { nodes { name } }
    characters(sort: FAVOURITES_DESC, perPage: 6) { nodes { name { full } } }
    recommendations(sort: RATING_DESC, perPage: 5) {
      nodes { mediaRecommendation { title { romaji english } } }
    }
  }
}
"""


def fetch_anime(query):
    key = query.lower().strip()
    if key in _cache and time.time() - _cache[key][0] < 900:
        return _cache[key][1]
    try:
        r = requests.post(ANILIST_URL, json={"query": _QUERY, "variables": {"s": query}}, timeout=12)
        if r.status_code != 200:
            return None
        m = (r.json().get("data") or {}).get("Media")
        if not m:
            return None
        desc = re.sub(r"<[^>]+>", "", m.get("description") or "").replace("\n", " ").strip()
        rec = {
            "title": m["title"].get("english") or m["title"].get("romaji"),
            "episodes": m.get("episodes"),
            "status": (m.get("status") or "").replace("_", " ").title(),
            "score": round(m["averageScore"] / 10, 1) if m.get("averageScore") else None,
            "year": m.get("seasonYear"),
            "genres": m.get("genres") or [],
            "studios": [s["name"] for s in m["studios"]["nodes"]],
            "characters": [c["name"]["full"] for c in m["characters"]["nodes"]],
            "recommendations": [
                (x["mediaRecommendation"]["title"].get("english")
                 or x["mediaRecommendation"]["title"].get("romaji"))
                for x in m["recommendations"]["nodes"] if x.get("mediaRecommendation")
            ],
            "cover": (m.get("coverImage") or {}).get("large"),
            "description": desc,
        }
        _cache[key] = (time.time(), rec)
        return rec
    except requests.RequestException:
        return None


_CHAR_QUERY = """
query ($s: String) {
  Character(search: $s) {
    name { full }
    media(type: ANIME, sort: POPULARITY_DESC, perPage: 1) {
      nodes {
        title { romaji english } description(asHtml: false) episodes status averageScore
        seasonYear genres coverImage { large } studios(isMain: true) { nodes { name } }
      }
    }
  }
}
"""


def fetch_character(name):
    """Look up a CHARACTER on AniList → their main anime. Single-word probes are VALIDATED
    against the returned character's name, so junk words (e.g. 'team') can't match a wrong character."""
    rec = _char_once(name.strip())            # try the full phrase first
    if rec:
        return rec
    skip = STOP | _PRO | {"he", "she", "him", "her", "his", "hers", "and", "team", "name", "anime"}
    words = sorted({w.lower() for w in name.split() if len(w) >= 4 and w.lower() not in skip},
                   key=len, reverse=True)
    for w in words:
        rec = _char_once(w)
        who = rec["characters"][0].lower() if rec and rec.get("characters") else ""
        if rec and w in who:                  # trust only if the word is truly in the character's name
            return rec
    return None


def _char_once(name):
    try:
        r = requests.post(ANILIST_URL, json={"query": _CHAR_QUERY, "variables": {"s": name}}, timeout=12)
        if r.status_code != 200:
            return None
        ch = (r.json().get("data") or {}).get("Character")
        if not ch or not ch["media"]["nodes"]:
            return None
        m = ch["media"]["nodes"][0]
        desc = re.sub(r"<[^>]+>", "", m.get("description") or "").replace("\n", " ").strip()
        return {
            "title": m["title"].get("english") or m["title"].get("romaji"),
            "episodes": m.get("episodes"),
            "status": (m.get("status") or "").replace("_", " ").title(),
            "score": round(m["averageScore"] / 10, 1) if m.get("averageScore") else None,
            "year": m.get("seasonYear"), "genres": m.get("genres") or [],
            "studios": [s["name"] for s in m["studios"]["nodes"]],
            "characters": [ch["name"]["full"]], "recommendations": [],
            "cover": (m.get("coverImage") or {}).get("large"), "description": desc,
        }
    except requests.RequestException:
        return None


def as_context(a):
    if not a:
        return ""
    return (
        "\n\n[VERIFIED DATA from AniList — use for all facts]\n"
        f"Title: {a['title']}\n"
        f"Episodes: {a['episodes']} | Status: {a['status']} | Year: {a['year']}\n"
        f"Score: {a['score']}/10 | Genres: {', '.join(a['genres'])}\n"
        f"Studio: {', '.join(a['studios']) or 'n/a'}\n"
        f"Main characters: {', '.join(a['characters']) or 'n/a'}\n"
        f"Similar anime: {', '.join(a['recommendations']) or 'n/a'}\n"
        f"Synopsis: {a['description'][:350]}\n"
    )


def card(a):
    """Compact payload the frontend renders as a poster card."""
    if not a:
        return None
    return {
        "title": a["title"], "cover": a["cover"], "score": a["score"],
        "episodes": a["episodes"], "year": a["year"], "genres": a["genres"][:3],
    }


# --------------------------------------------------------------------------
# Web knowledge layer — Fandom (deep fan content) + Wikipedia (fallback)
# --------------------------------------------------------------------------
WIKI_H = {"User-Agent": "AnimeSensei/1.0 (portfolio demo; supyallukshyadl1@gmail.com)"}
_slug_cache = {}

# Questions AniList already answers → skip the web hop (keeps it fast).
_STRUCTURED = re.compile(r"\b(how many|episodes?|seasons?|score|rating|rank|studio|genre|"
                         r"recommend|similar|suggest|what year|when did|air|release)\b", re.I)
# Fan questions that need prose knowledge → do a web lookup.
_DETAIL = re.compile(r"\b(who|whose|character|team|cast|member|love|crush|girlfriend|boyfriend|"
                     r"relationship|ship|dating|married|wife|husband|backstory|origin|power|abilit|"
                     r"special|arc|happen|die|death|end|ending|spoiler|villain|antagonist|fight|"
                     r"defeat|strongest|weak|personality|voice actor|quote|why|how does|explain|meaning)\b", re.I)


def _slugs(title):
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return [base, base.replace("-", "")]


def _extract(api, page_title):
    try:
        r = requests.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                         "redirects": 1, "exchars": 2200, "format": "json", "titles": page_title},
                         headers=WIKI_H, timeout=8)
        txt = list(r.json()["query"]["pages"].values())[0].get("extract", "")
        if txt:
            return txt
    except Exception:
        pass
    try:  # some Fandom wikis lack the extracts extension → parse wikitext & strip
        r = requests.get(api, params={"action": "parse", "page": page_title, "prop": "wikitext",
                         "redirects": 1, "format": "json"}, headers=WIKI_H, timeout=8)
        wt = r.json()["parse"]["wikitext"]["*"]
        wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
        wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", wt)
        wt = re.sub(r"<[^>]+>", " ", wt).replace("'''", "").replace("''", "")
        return re.sub(r"\s+", " ", wt).strip()[:2200]
    except Exception:
        return ""


def _search_wiki(api, query):
    try:
        r = requests.get(api, params={"action": "query", "list": "search", "srsearch": query,
                         "srlimit": 1, "format": "json"}, headers=WIKI_H, timeout=8)
        hits = r.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception:
        return None


def web_lookup(query, anime_title):
    """Return (text, source). Tries the anime's Fandom wiki first, then Wikipedia."""
    if anime_title:
        tries = [_slug_cache[anime_title]] if anime_title in _slug_cache else _slugs(anime_title)
        for s in tries:
            api = f"https://{s}.fandom.com/api.php"
            page = _search_wiki(api, query)
            if page:
                txt = _extract(api, page)
                if txt:
                    _slug_cache[anime_title] = s
                    return txt, f"Fandom ({s}.fandom.com)"
    api = "https://en.wikipedia.org/w/api.php"
    page = _search_wiki(api, f"{query} {anime_title}".strip())
    if page:
        txt = _extract(api, page)
        if txt:
            return txt, "Wikipedia"
    return "", ""


# --------------------------------------------------------------------------
# Lightweight anime-title extraction (+ pronoun resolution via memory)
# --------------------------------------------------------------------------
STOP = set("""a an the is are was were do does did what which who whom whose how many much about tell
me give show of in on for this that these those it its they them their anime manga series show watch
please can you plot summary describe rating score good bad genre genres character characters cast
episode episodes season seasons studio made produce recommend recommendation similar like when year
release aired airing finished ongoing old worth popular best so and or vs review explain data info
each main protagonist antagonist villain hero heroine strongest weakest strong weak worst
specialty speciality special powers abilities""".split())
_PRO = {"it", "its", "this", "that", "they", "them"}

_LEAD = re.compile(r"""^\s*(?:
      (?:can|could)\s+you\s+ | please\s+ | about\s+
    | (?:the\s+)?(?:name\s+of\s+(?:the\s+)?)?anime\s+(?:name\s+)?(?:of\s+)?(?:this\s+|the\s+)?characters?\s+(?:called\s+|named\s+)?
    | (?:what|which)(?:'s|\s+is)?\s+(?:the\s+)?(?:anime|show|series)\s+(?:name\s+)?(?:of\s+|is\s+|does\s+|for\s+|with\s+)?(?:the\s+)?(?:character\s+)?
    | tell\s+me\s+(?:more\s+)? | tell\s+me\s+
    | what(?:'s|\s+is|\s+are)\s+(?:the\s+)?(?:plot|synopsis|summary|story|data|info)\s+(?:of\s+)?
    | what(?:'s|\s+is)\s+
    | who\s+are\s+the\s+(?:main\s+)?characters?\s+(?:in|of|from)\s+
    | who(?:'s|\s+is|\s+are|\s+does)\s+(?:in\s+)?
    | how\s+many\s+(?:episodes?|seasons?)\s+(?:does\s+|are\s+in\s+|in\s+|are\s+there\s+in\s+)?
    | how\s+long\s+is\s+
    | recommend\s+(?:me\s+)?(?:some\s+anime|something|anime)?\s*(?:like|similar\s+to)\s+
    | (?:something|anime)\s+(?:like|similar\s+to)\s+
    | suggest\s+(?:something|anime)?\s*(?:like\s+)?
    | describe\s+ | review\s+ | rate\s+ | is\s+ | this\s+ | the\s+
  )""", re.X | re.I)
_TRAIL = re.compile(r"\s*(?:have|has|about|anime|manga|series|show|from|good|worth\s+watching|please|thanks?)\s*$", re.I)
_ANIME_OF_CHAR = re.compile(r"(what|which)\s+(anime|show|series)|anime\s+(name\s+)?(of|for|with|whose)|"
                            r"character\s+(named|called|name)|from\s+(what|which)\s+(anime|show)|"
                            r"who\s+(is|are)\b", re.I)


def _relevant(cand, title):
    cw = set(re.findall(r"[a-z0-9]+", cand.lower()))
    tw = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    if not cw:
        return False
    if cand.lower() in (title or "").lower():
        return True
    return len(cw & tw) >= max(1, len(cw) // 2)   # at least half the query words appear in the title


def extract_title(text):
    low = text.strip().lower().rstrip("?.! ")
    toks = low.split()
    if not toks or set(toks) <= (_PRO | STOP):
        return None
    s = low
    prev = None
    while prev != s:                    # strip leading command phrases repeatedly
        prev = s
        s = _LEAD.sub("", s).strip()
    prev = None
    while prev != s:                    # then trailing filler
        prev = s
        s = _TRAIL.sub("", s).strip()
    s = re.sub(r"\s+and\s+(what|which|where|who|from|is|are|does|how)\b.*$", "", s)  # drop trailing clause
    s = s.strip(" -:!?.")
    st = s.split()
    if not st or set(st) <= (_PRO | STOP):
        return None
    return s or None


# --------------------------------------------------------------------------
# LLM call (OpenRouter) with free-model fallback chain
# --------------------------------------------------------------------------
def _openai_style(url, key, model, messages, title=None):
    """Call any OpenAI-compatible chat endpoint (OpenRouter / Groq / OpenAI)."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if title:
        headers["X-Title"] = title
    r = requests.post(url, headers=headers,
                      json={"model": model, "messages": messages, "temperature": 0.55, "max_tokens": 240},
                      timeout=30)
    if r.status_code == 200:
        return (r.json()["choices"][0]["message"].get("content") or "").strip() or None
    return None


def _openrouter(messages):
    for model in OR_MODELS:                     # try each free model in turn
        try:
            out = _openai_style(OPENROUTER_URL, OPENROUTER_KEY, model, messages, title="Lua")
            if out:
                return out
        except requests.RequestException:
            pass
    return None


def _groq(messages):
    return _openai_style("https://api.groq.com/openai/v1/chat/completions", GROQ_KEY,
                         "llama-3.3-70b-versatile", messages)


def _openai(messages):
    return _openai_style("https://api.openai.com/v1/chat/completions", OPENAI_KEY,
                         "gpt-4o-mini", messages)


def _gemini(messages):
    sys = " ".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": ("model" if m["role"] == "assistant" else "user"),
                 "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
    body = {"contents": contents, "generationConfig": {"temperature": 0.55, "maxOutputTokens": 240}}
    if sys:
        body["systemInstruction"] = {"parts": [{"text": sys}]}
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}")
    r = requests.post(url, json=body, timeout=30)
    if r.status_code == 200:
        cand = r.json().get("candidates", [])
        if cand:
            txt = "".join(p.get("text", "") for p in cand[0].get("content", {}).get("parts", [])).strip()
            return txt or None
    return None


def _providers():
    """Enabled providers in failover order: FREE first, PAID last (Gemini dead-last)."""
    p = []
    if OPENROUTER_KEY: p.append(("OpenRouter", _openrouter))  # free, stronger models (founder pref)
    if GROQ_KEY:       p.append(("Groq", _groq))            # free, fast (fallback)
    if OPENAI_KEY:     p.append(("OpenAI", _openai))        # paid
    if GEMINI_KEY:     p.append(("Gemini", _gemini))        # paid (founder: use last, flash only)
    return p


def ask_llm(messages):
    """Try each configured provider until one answers — automatic failover."""
    providers = _providers()
    if not providers:
        return "No API key is set yet — add one to the .env file and restart. 🙏"
    for name, fn in providers:
        try:
            out = fn(messages)
            if out:
                return out
        except requests.RequestException:
            pass
    return "All my brains are at their daily limit right now — give it a minute and try again. 🙏"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()
    convo = _convo(data.get("cid") or "default")     # this tab's private conversation
    if not msg:
        return jsonify(reply="Ask me anything about anime! 🎌", card=None)

    title = extract_title(msg)
    anime = convo["anime"]
    if title:
        new = None
        if _ANIME_OF_CHAR.search(msg):                       # "what anime is X from" / character lookup
            new = fetch_character(title)
        if not new:                                          # else resolve as an anime title (strict)
            cand = fetch_anime(title)
            if cand and _relevant(title, cand["title"]):
                new = cand
        if new:                                              # switch context ONLY on a confident match
            anime = new
            convo["anime"] = anime
        # else: keep the current anime — never clobber context on a vague follow-up

    # Deep fan questions (characters, relationships, lore) → pull web grounding.
    web_txt, web_src = "", ""
    if _DETAIL.search(msg) and not _STRUCTURED.search(msg):
        web_txt, web_src = web_lookup(msg, anime["title"] if anime else "")

    system = BASE_PERSONA + HARU_VOICE          # single default persona — no gender prompt

    messages = [{"role": "system", "content": system}]
    messages += convo["history"][-6:]                # remember the last few turns for continuity
    grounding = as_context(anime)
    if web_txt:
        grounding += f"\n\n[WEB CONTEXT from {web_src} — use this to answer]:\n{web_txt[:1400]}\n"
    messages.append({"role": "user", "content": msg + grounding})

    reply = ask_llm(messages)

    convo["history"] = (convo["history"] + [{"role": "user", "content": msg},
                                            {"role": "assistant", "content": reply}])[-12:]
    return jsonify(reply=reply, card=card(anime) if title and anime else None)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
