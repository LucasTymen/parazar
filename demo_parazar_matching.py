from calculator import ParazarMatcher
import pandas as pd
import random

# === Personnages One Piece & Dragon Ball ===
male_characters = [
    ("Luffy", 19), ("Zoro", 21), ("Sanji", 21), ("Usopp", 19), ("Brook", 90), ("Franky", 36),
    ("Shanks", 39), ("Ace", 20), ("Sabo", 22), ("Law", 26), ("Smoker", 34), ("Crocodile", 46),
    ("Buggy", 39), ("Kizaru", 50), ("Enel", 30), ("Jinbei", 45), ("Kaido", 59), ("Katakuri", 48),
    ("Bartolomeo", 24), ("Koby", 18),
    ("Goku", 35), ("Vegeta", 37), ("Piccolo", 45), ("Krillin", 33), ("Gohan", 24), ("Yamcha", 34),
    ("Tenshinhan", 38), ("Trunks", 18), ("Goten", 17), ("Mr Satan", 50), ("Buu", 999),
    ("Beerus", 1000), ("Whis", 1500), ("Cell", 5), ("Freezer", 100), ("Broly", 30), ("Raditz", 35),
    ("Nappa", 50), ("Jiren", 40), ("Hit", 37),
]

female_characters = [
    ("Nami", 20), ("Robin", 30), ("Vivi", 18), ("Boa Hancock", 29), ("Yamato", 28),
    ("Tashigi", 23), ("Reiju", 24), ("Perona", 25), ("Shirahoshi", 16), ("Jewelry Bonney", 22),
    ("Bulma", 35), ("Chi-Chi", 34), ("Videl", 20), ("Android 18", 28), ("Pan", 16),
    ("Lunch", 25), ("Mai", 30), ("Kale", 19), ("Caulifla", 20), ("Cheelai", 22),
]

topics_pool = [
    "combat", "aventures", "famille", "stratégie", "piraterie", "justice",
    "entraide", "voyage", "cuisine", "romance", "pouvoirs", "technologie", "sport", "méditation"
]

def generate_participant(name: str, age: int, gender: str) -> dict:
    return {
        'email': f"{name.lower().replace(' ', '_')}@anime.com",
        'first_name': name,
        'age': age,
        'gender': gender,
        'experience_name': 'Battle of Universes',
        'experience_date': '2025-09-01',
        'experience_city': 'Paris',
        'topics_conversations': ",".join(random.sample(topics_pool, k=random.randint(2, 4))),
        'relationship_status': random.choice(['célibataire', 'en couple']),
        'introverted_degree': round(random.uniform(0.1, 0.9), 2),
        'parazar_partner_id': f"PZ{random.randint(1000, 9999)}",
        'telephone': f"+33{random.randint(600000000, 699999999)}"
    }

# Génération équilibrée pour créer des groupes valides
panel_data = []

# Prendre 24 hommes et 24 femmes pour garantir 8 groupes de 6 personnes
selected_males = random.sample(male_characters, 24)
selected_females = random.sample(female_characters, 24)

for name, age in selected_males:
    panel_data.append(generate_participant(name, age, "m"))
for name, age in selected_females:
    panel_data.append(generate_participant(name, age, "f"))

# Créer le DataFrame
df_panel = pd.DataFrame(panel_data)

# Lancement du matcher
matcher = ParazarMatcher()
participants = matcher.load_from_dataframe(df_panel)

print(f"Total générés : {len(df_panel)}")
print(f"Participants valides : {len(participants)}")

status, results = matcher.create_optimal_groups(participants)

# Résultats
print("\n=== RÉSULTAT DU MATCHING PARAZAR ===")
print(f"Statut global : {status.value}")
print(f"Groupes créés : {len(results['groups'])}")
print(f"Non appariés : {len(results['unmatched'])}")
print(f"Stats : {results.get('stats')}")

for i, group in enumerate(results["groups"], 1):
    print(f"\n🔹 Groupe {i} - {group.id} - Score: {group.compatibility_score}")
    for p in group.participants:
        print(f"  - {p.first_name} ({p.gender.upper()}, {p.age} ans) | introversion={p.introverted_degree:.2f} | topics: {', '.join(p.topics_conversations)}")
