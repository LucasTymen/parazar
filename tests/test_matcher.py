import pytest
import pandas as pd
from calculator import ParazarMatcher

# === Fixtures ===

@pytest.fixture
def sample_dataframe():
    import random

    topics_pool = [
        "combat", "aventures", "famille", "stratégie", "piraterie", "justice",
        "entraide", "voyage", "cuisine", "romance", "pouvoirs", "technologie", "sport", "méditation"
    ]

    def generate_participant(name, age, gender):
        return {
            'email': f"{name.lower()}@anime.com",
            'first_name': name,
            'age': age,
            'gender': gender,
            'experience_name': 'Battle of Universes',
            'experience_date': '2025-09-01',
            'experience_city': 'Paris',
            'topics_conversations': ",".join(random.sample(topics_pool, 3)),
            'relationship_status': random.choice(['célibataire', 'en couple']),
            'introverted_degree': round(random.uniform(0.1, 0.9), 2),
            'parazar_partner_id': f"PZ_{random.randint(1000,9999)}",
            'telephone': f"06{random.randint(10000000,99999999)}"
        }

    names = [('Luffy', 19, 'm'), ('Nami', 20, 'f'), ('Zoro', 21, 'm'), ('Robin', 30, 'f')]
    data = [generate_participant(n, a, g) for n, a, g in names]
    return pd.DataFrame(data)

# === Tests ===

def test_load_from_dataframe(sample_dataframe):
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(sample_dataframe)
    assert len(participants) == len(sample_dataframe)

def test_create_optimal_groups(sample_dataframe):
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(sample_dataframe)
    status, results = matcher.create_optimal_groups(participants)

    assert status.value in ["SUCCESS", "PARTIAL", "EMPTY"]
    assert "groups" in results
    assert "unmatched" in results
