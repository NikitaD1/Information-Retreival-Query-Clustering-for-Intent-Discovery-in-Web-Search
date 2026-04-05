"""
dataset.py
----------
Query corpus for Intent Discovery in Web Search.
  • 230 queries spanning 10 intent categories
  • ~13 % multilingual (Hindi, Spanish, French, German)
  • Ground-truth intent labels (used for ARI evaluation only – NOT for clustering)
  • Translation dictionary for cross-lingual normalisation
"""

# ─────────────────────────────────────────────────────────────────────────────
# RAW QUERY CORPUS
# ─────────────────────────────────────────────────────────────────────────────
QUERIES = [
    # ── WEATHER (20) ─────────────────────────────────────────────────────────
    "weather in Mumbai today",
    "temperature in Delhi",
    "will it rain in Bangalore tomorrow",
    "weather forecast Pune",
    "humidity levels in Chennai",
    "current weather Hyderabad",
    "Mumbai monsoon season",
    "weather Goa",
    "temperature in Goa today",
    "rainy season in Kerala",
    "snowfall in Shimla",
    "weather conditions Kolkata",
    "is it cold in Manali",
    "weather update Jaipur",
    "climate in Ooty hills",
    "storm alert Mumbai",
    "weather this week Ahmedabad",
    "fog in Delhi winter",
    "UV index Chennai today",
    "weather Leh Ladakh",

    # ── SPORTS / CRICKET (20) ──────────────────────────────────────────────
    "India vs Australia cricket score",
    "IPL 2024 match schedule",
    "latest cricket news India",
    "cricket live score today",
    "Virat Kohli batting average",
    "World Cup 2024 India squad",
    "IPL auction players list",
    "Rohit Sharma cricket stats",
    "who won IPL 2023",
    "India cricket team captain",
    "football match highlights today",
    "Premier League standings 2024",
    "FIFA World Cup results",
    "Messi goals this season",
    "Champions League final score",
    "ISL football schedule India",
    "PV Sindhu badminton ranking",
    "tennis Grand Slam 2024 schedule",
    "Olympics 2024 India medals",
    "Pro Kabaddi League results",

    # ── FOOD / RESTAURANTS (20) ────────────────────────────────────────────
    "best restaurants near me",
    "biryani recipe at home",
    "how to make paneer butter masala",
    "pizza delivery near me",
    "vegan restaurants Mumbai",
    "butter chicken recipe easy",
    "best street food in Delhi",
    "healthy breakfast recipes",
    "how to make dosa batter",
    "restaurants open now near me",
    "Chinese food near me",
    "South Indian thali restaurants",
    "Zomato order food online",
    "how to make chocolate cake",
    "keto diet recipes India",
    "North Indian cuisine restaurants",
    "how to make masala chai",
    "gluten free food options near me",
    "continental food restaurants Bangalore",
    "dinner recipes quick easy",

    # ── TRAVEL / TOURISM (20) ─────────────────────────────────────────────
    "best places to visit in India",
    "Goa tourism packages 2024",
    "flight tickets Mumbai to Delhi",
    "hotel booking in Manali",
    "Rajasthan tour itinerary",
    "visa requirements for Canada",
    "backpacking Southeast Asia",
    "cheapest international flights from India",
    "Kerala backwaters tour package",
    "Shimla Manali trip plan",
    "Maldives holiday packages",
    "tourist places near Bangalore",
    "India travel guide budget",
    "Varanasi temple tour guide",
    "Andaman Nicobar tour package",
    "train tickets IRCTC booking",
    "international travel insurance India",
    "Ladakh road trip guide",
    "Taj Mahal visiting hours",
    "hill stations near Mumbai",

    # ── TECH / SOFTWARE (20) ──────────────────────────────────────────────
    "how to install Python",
    "best JavaScript frameworks 2024",
    "machine learning tutorial beginners",
    "ChatGPT vs Gemini comparison",
    "how to use Git commands",
    "Python pandas tutorial",
    "best laptop for programming 2024",
    "cloud computing AWS vs Azure",
    "how to build REST API",
    "deep learning course free online",
    "SQL database tutorial beginners",
    "React JS tutorial for beginners",
    "best IDE for Python development",
    "Docker tutorial beginners",
    "Linux commands cheat sheet",
    "cybersecurity basics tutorial",
    "how to learn data science",
    "mobile app development course",
    "blockchain technology explained",
    "Kubernetes tutorial deployment",

    # ── HEALTH / MEDICAL (20) ─────────────────────────────────────────────
    "symptoms of dengue fever",
    "how to reduce blood pressure naturally",
    "diabetes diet plan India",
    "home remedies for cold cough",
    "yoga for back pain relief",
    "best hospitals in Mumbai",
    "vitamin D deficiency symptoms",
    "mental health anxiety tips",
    "how to lose weight fast",
    "thyroid symptoms and treatment",
    "best dermatologist near me",
    "COVID vaccine side effects",
    "exercises for knee pain",
    "Ayurvedic treatment arthritis",
    "daily protein intake recommendation",
    "fever home remedies",
    "insomnia sleep disorder cure",
    "heart attack warning signs",
    "migraine headache relief",
    "stress management techniques",

    # ── NEWS / CURRENT AFFAIRS (20) ───────────────────────────────────────
    "India news today headlines",
    "budget 2024 highlights India",
    "stock market today India",
    "Sensex Nifty today performance",
    "election results 2024 India",
    "RBI interest rate announcement",
    "India GDP growth rate 2024",
    "inflation news India today",
    "parliament session news today",
    "Modi government new policy",
    "startup funding news India",
    "India China border news latest",
    "Supreme Court ruling today India",
    "GST changes 2024 India",
    "gold price today India",
    "petrol diesel price today India",
    "rupee dollar exchange rate today",
    "ISRO space mission latest update",
    "India trade policy news",
    "RBI monetary policy decision",

    # ── SHOPPING / E-COMMERCE (20) ────────────────────────────────────────
    "Amazon sale today offers",
    "Flipkart Big Billion Days deals",
    "best smartphone under 15000 India",
    "Samsung phone price India 2024",
    "online shopping discount coupons",
    "Myntra fashion sale offer",
    "best headphones under 5000",
    "laptop deals online India",
    "air conditioner best brand India",
    "iPhone 15 price India",
    "smartwatch comparison 2024",
    "online grocery delivery app India",
    "furniture sale online India",
    "washing machine best buy India",
    "electric scooter price India",
    "best TV under 40000 India",
    "gaming mouse buy online",
    "refrigerator deals Flipkart",
    "camera price India buy online",
    "inverter AC best model India",

    # ── EDUCATION / LEARNING (20) ─────────────────────────────────────────
    "UPSC preparation tips strategy",
    "JEE Advanced syllabus 2024",
    "MBA entrance exam CAT preparation",
    "NEET exam pattern syllabus",
    "best online courses Coursera",
    "how to study effectively tips",
    "CBSE board exam schedule 2024",
    "scholarship for engineering students India",
    "GATE exam preparation guide",
    "IIT admission process",
    "data science certification online",
    "English speaking course free",
    "coding bootcamp India",
    "GRE preparation material",
    "IELTS exam tips strategies",
    "digital marketing course online",
    "MBA colleges India ranking",
    "CLAT law entrance exam",
    "school math homework help",
    "Python programming course free",

    # ── ENTERTAINMENT / MOVIES (20) ───────────────────────────────────────
    "Bollywood movies 2024 release",
    "new movies on Netflix India",
    "Amazon Prime latest shows India",
    "Salman Khan new movie 2024",
    "top rated movies IMDb 2024",
    "Korean drama to watch 2024",
    "Marvel movies release schedule",
    "Hindi web series 2024 new",
    "Oscar winning movies 2024",
    "comedy movies to watch",
    "Disney Plus India content",
    "Deepika Padukone upcoming film",
    "Hollywood movies releasing this week",
    "best thriller series Netflix",
    "Shah Rukh Khan new movie",
    "cartoon shows for kids",
    "stand up comedy India",
    "Spotify vs JioSaavn streaming",
    "best OTT platform India 2024",
    "Tamil movies dubbed Hindi",

    # ─────────────────────────────────────────────────────────────────────
    # MULTILINGUAL QUERIES  (~30 = 13 % of corpus)
    # ─────────────────────────────────────────────────────────────────────

    # ── Hindi / Devanagari (10) ───────────────────────────────────────────
    "\u0906\u091c \u092e\u094c\u0938\u092e \u0915\u0948\u0938\u093e \u0939\u0948",                         # How is the weather today
    "\u0926\u093f\u0932\u094d\u0932\u0940 \u092e\u0947\u0902 \u092c\u093e\u0930\u093f\u0936 \u0939\u094b\u0917\u0940 \u0915\u094d\u092f\u093e",               # Will it rain in Delhi
    "\u0915\u094d\u0930\u093f\u0915\u0947\u091f \u092e\u0948\u091a \u0915\u093e \u0938\u094d\u0915\u094b\u0930",                       # Cricket match score
    "\u092c\u093f\u0930\u092f\u093e\u0928\u0940 \u092c\u0928\u093e\u0928\u0947 \u0915\u0940 \u0935\u093f\u0927\u093f",                 # How to make biryani
    "\u092e\u0941\u0902\u092c\u0908 \u092e\u0947\u0902 \u0905\u091a\u094d\u091b\u0947 \u0930\u0947\u0938\u094d\u0924\u0930\u093e\u0902",               # Good restaurants in Mumbai
    "\u092a\u0947\u091f\u094d\u0930\u094b\u0932 \u0915\u093e \u092d\u093e\u0935 \u0906\u091c",                       # Petrol price today
    "\u0928\u0908 \u092b\u093f\u0932\u094d\u092e 2024 \u092c\u0949\u0932\u0940\u0935\u0941\u0921",                  # New Bollywood movie 2024
    "\u0938\u094d\u0935\u0938\u094d\u0925 \u0916\u093e\u0928\u093e \u0930\u0947\u0938\u093f\u092a\u0940",                       # Healthy food recipe
    "\u0926\u093f\u0932\u094d\u0932\u0940 \u092e\u0947\u0902 \u0918\u0942\u092e\u0928\u0947 \u0915\u0940 \u091c\u0917\u0939",            # Places to visit in Delhi
    "\u0936\u0947\u092f\u0930 \u092c\u093e\u091c\u093e\u0930 \u0906\u091c \u0915\u094d\u092f\u093e \u0939\u0948",              # What is stock market today

    # ── Spanish (7) ───────────────────────────────────────────────────────
    "tiempo en Barcelona hoy",
    "mejor restaurante cerca de mi",
    "c\u00f3mo aprender programaci\u00f3n Python",
    "noticias de hoy Espa\u00f1a \u00faltima hora",
    "pel\u00edculas populares Netflix 2024",
    "vuelos baratos a M\u00e9xico desde Espa\u00f1a",
    "receta de paella valenciana",

    # ── French (6) ────────────────────────────────────────────────────────
    "m\u00e9t\u00e9o Paris aujourd'hui pr\u00e9visions",
    "meilleurs restaurants \u00e0 Paris",
    "comment apprendre la programmation",
    "actualit\u00e9s France aujourd'hui",
    "films populaires sur Netflix 2024",
    "recette de ratatouille fran\u00e7aise",

    # ── German (7) ────────────────────────────────────────────────────────
    "Wetter Berlin heute Vorhersage",
    "beste Restaurants in M\u00fcnchen",
    "Nachrichten Deutschland heute aktuell",
    "g\u00fcnstige Fl\u00fcge buchen Europa",
    "Programmieren lernen f\u00fcr Anf\u00e4nger",
    "beliebte Serien Netflix 2024",
    "Bundesliga Ergebnisse heute",
]

# ─────────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH INTENT LABELS  (for ARI evaluation only; never fed to clusterer)
# ─────────────────────────────────────────────────────────────────────────────
INTENT_CATEGORIES = [
    "weather", "sports", "food", "travel", "tech",
    "health", "news", "shopping", "education", "entertainment",
]

INTENT_LABELS = (
    ["weather"]       * 20 +
    ["sports"]        * 20 +
    ["food"]          * 20 +
    ["travel"]        * 20 +
    ["tech"]          * 20 +
    ["health"]        * 20 +
    ["news"]          * 20 +
    ["shopping"]      * 20 +
    ["education"]     * 20 +
    ["entertainment"] * 20 +
    # Hindi (10)
    ["weather", "weather", "sports", "food", "food",
     "news", "entertainment", "food", "travel", "news"] +
    # Spanish (7)
    ["weather", "food", "tech", "news", "entertainment", "travel", "food"] +
    # French (6)
    ["weather", "food", "tech", "news", "entertainment", "food"] +
    # German (7)
    ["weather", "food", "news", "travel", "tech", "entertainment", "sports"]
)

# Language tag for each query
LANGUAGE_LABELS = ["en"] * 200 + ["hi"] * 10 + ["es"] * 7 + ["fr"] * 6 + ["de"] * 7

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION DICTIONARY  (non-English → English)
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATIONS = {
    # Hindi
    "\u0906\u091c \u092e\u094c\u0938\u092e \u0915\u0948\u0938\u093e \u0939\u0948":             "how is the weather today",
    "\u0926\u093f\u0932\u094d\u0932\u0940 \u092e\u0947\u0902 \u092c\u093e\u0930\u093f\u0936 \u0939\u094b\u0917\u0940 \u0915\u094d\u092f\u093e":   "will it rain in Delhi",
    "\u0915\u094d\u0930\u093f\u0915\u0947\u091f \u092e\u0948\u091a \u0915\u093e \u0938\u094d\u0915\u094b\u0930":         "cricket match score",
    "\u092c\u093f\u0930\u092f\u093e\u0928\u0940 \u092c\u0928\u093e\u0928\u0947 \u0915\u0940 \u0935\u093f\u0927\u093f":             "how to make biryani recipe",
    "\u092e\u0941\u0902\u092c\u0908 \u092e\u0947\u0902 \u0905\u091a\u094d\u091b\u0947 \u0930\u0947\u0938\u094d\u0924\u0930\u093e\u0902":           "good restaurants in Mumbai",
    "\u092a\u0947\u091f\u094d\u0930\u094b\u0932 \u0915\u093e \u092d\u093e\u0935 \u0906\u091c":                 "petrol price today",
    "\u0928\u0908 \u092b\u093f\u0932\u094d\u092e 2024 \u092c\u0949\u0932\u0940\u0935\u0941\u0921":              "new Bollywood movie 2024",
    "\u0938\u094d\u0935\u0938\u094d\u0925 \u0916\u093e\u0928\u093e \u0930\u0947\u0938\u093f\u092a\u0940":                 "healthy food recipe",
    "\u0926\u093f\u0932\u094d\u0932\u0940 \u092e\u0947\u0902 \u0918\u0942\u092e\u0928\u0947 \u0915\u0940 \u091c\u0917\u0939":        "places to visit in Delhi",
    "\u0936\u0947\u092f\u0930 \u092c\u093e\u091c\u093e\u0930 \u0906\u091c \u0915\u094d\u092f\u093e \u0939\u0948":          "stock market today",
    # Spanish
    "tiempo en Barcelona hoy":                "weather in Barcelona today",
    "mejor restaurante cerca de mi":          "best restaurant near me",
    "c\u00f3mo aprender programaci\u00f3n Python":       "how to learn Python programming",
    "noticias de hoy Espa\u00f1a \u00faltima hora":       "news today Spain",
    "pel\u00edculas populares Netflix 2024":             "popular movies Netflix 2024",
    "vuelos baratos a M\u00e9xico desde Espa\u00f1a":    "cheap flights to Mexico",
    "receta de paella valenciana":            "Valencian paella recipe",
    # French
    "m\u00e9t\u00e9o Paris aujourd'hui pr\u00e9visions":  "Paris weather today forecast",
    "meilleurs restaurants \u00e0 Paris":              "best restaurants in Paris",
    "comment apprendre la programmation":    "how to learn programming",
    "actualit\u00e9s France aujourd'hui":             "France news today",
    "films populaires sur Netflix 2024":     "popular movies on Netflix 2024",
    "recette de ratatouille fran\u00e7aise":           "French ratatouille recipe",
    # German
    "Wetter Berlin heute Vorhersage":        "Berlin weather today forecast",
    "beste Restaurants in M\u00fcnchen":              "best restaurants in Munich",
    "Nachrichten Deutschland heute aktuell": "current Germany news today",
    "g\u00fcnstige Fl\u00fcge buchen Europa":          "book cheap flights Europe",
    "Programmieren lernen f\u00fcr Anf\u00e4nger":        "learn programming for beginners",
    "beliebte Serien Netflix 2024":          "popular Netflix series 2024",
    "Bundesliga Ergebnisse heute":           "Bundesliga results today",
}

N_TRUE_INTENTS = len(INTENT_CATEGORIES)   # 10

assert len(QUERIES) == len(INTENT_LABELS) == len(LANGUAGE_LABELS), \
    "Length mismatch in dataset arrays"
