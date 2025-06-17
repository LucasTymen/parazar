import pytest
from calculator import (
    ParazarMatcher,
    MatchingStatus,
    Group,
    Participant,
    load_parazar_data
)
import pandas as pd
import random
from datetime import datetime
from unittest.mock import patch

male_characters = [("Luffy", 19), ("Brook", 90)]
female_characters = [("Nami", 20)]

topics_pool = ["combat", "famille", "voyage", "technologie"]

def generate_participant(name: str, age: int, gender: str) -> dict:
    return {
        'email': f"{name.lower()}@anime.com",
        'first_name': name,
        'age': age,
        'gender': gender,
        'experience_name': 'Battle of Universes',
        'experience_date': '2025-09-01',
        'experience_city': 'Paris',
        'topics_conversations': ",".join(random.sample(topics_pool, 2)),
        'relationship_status': 'célibataire',
        'introverted_degree': 0.5,
        'parazar_partner_id': 'PZ_1234',
        'telephone': '0601020304'
    }

def test_full_parazar_flow():
    """Test du flux complet de Parazar avec des données de test"""
    panel_data = [
        generate_participant(name, age, "m") for name, age in male_characters
    ] + [
        generate_participant(name, age, "f") for name, age in female_characters
    ]
    df = pd.DataFrame(panel_data)
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    
    assert len(participants) == len(panel_data)

    status, results = matcher.create_optimal_groups(participants)

    assert status is not None
    assert isinstance(results["groups"], list)
    assert isinstance(results["unmatched"], list)

def test_create_optimal_groups_success():
    """Test de création de groupes optimaux avec des données valides"""
    matcher = ParazarMatcher()
    
    # Créer un ensemble de participants avec des âges et genres compatibles
    participants_data = [
        generate_participant(f"F{i}", 25, "f") for i in range(4)
    ] + [
        generate_participant(f"M{i}", 27, "m") for i in range(4)
    ]
    
    df = pd.DataFrame(participants_data)
    participants = matcher.load_from_dataframe(df)
    
    status, results = matcher.create_optimal_groups(participants)
    
    assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
    assert len(results["groups"]) > 0
    
    # Vérifier les contraintes des groupes
    for group in results["groups"]:
        assert len(group.participants) >= matcher.min_group_size
        assert len(group.participants) <= matcher.max_group_size
        assert group.gender_balance.get('F', 0) >= matcher.min_females_per_group
        assert group.age_spread <= matcher.max_age_spread

def test_create_optimal_groups_large_dataset():
    """Test avec un grand ensemble de données"""
    matcher = ParazarMatcher()
    
    # Créer un grand ensemble de participants
    participants_data = []
    for i in range(20):
        age = 25 + (i % 10)  # Âges entre 25 et 34
        gender = "f" if i % 2 == 0 else "m"
        participants_data.append(generate_participant(f"P{i}", age, gender))
    
    df = pd.DataFrame(participants_data)
    participants = matcher.load_from_dataframe(df)
    
    status, results = matcher.create_optimal_groups(participants)
    
    assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
    assert len(results["groups"]) > 0

def test_determine_status():
    """Test de la détermination du statut"""
    matcher = ParazarMatcher()
    
    # Test avec des groupes valides
    groups = [
        Group(
            id="test1",
            participants=[
                Participant(email="f1@test.com", first_name="F1", gender="F", age=25),
                Participant(email="f2@test.com", first_name="F2", gender="F", age=26),
                Participant(email="m1@test.com", first_name="M1", gender="M", age=27),
                Participant(email="m2@test.com", first_name="M2", gender="M", age=28)
            ]
        )
    ]
    unmatched = []
    
    status = matcher._determine_status(groups, unmatched)
    assert status == MatchingStatus.SUCCESS
    
    # Test avec des groupes invalides
    invalid_groups = [
        Group(
            id="test2",
            participants=[
                Participant(email="f1@test.com", first_name="F1", gender="F", age=25),
                Participant(email="m1@test.com", first_name="M1", gender="M", age=27)
            ]
        )
    ]
    status = matcher._determine_status(invalid_groups, unmatched)
    assert status == MatchingStatus.FAILED_CONSTRAINTS

def test_gender_normalization():
    """Test de la normalisation des genres"""
    matcher = ParazarMatcher()
    
    # Test avec différents formats de genre
    test_cases = [
        ("f", "F"),
        ("m", "M"),
        ("F", "F"),
        ("M", "M"),
        ("féminin", "F"),
        ("masculin", "M"),
        ("x", "X")
    ]
    
    for input_gender, expected_gender in test_cases:
        df = pd.DataFrame([{
            'email': 'test@example.com',
            'first_name': 'Test',
            'age': 25,
            'gender': input_gender,
            'experience_name': 'Test',
            'experience_date': '2024-01-01',
            'experience_city': 'Paris'
        }])
        
        participants = matcher.load_from_dataframe(df)
        assert len(participants) == 1
        assert participants[0].gender == expected_gender

def test_age_vs_birth_year_coherence():
    """Test de la cohérence entre l'âge et l'année de naissance"""
    matcher = ParazarMatcher()
    current_year = datetime.now().year
    
    # Test avec des données cohérentes
    df_coherent = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'birth_year': current_year - 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    
    participants = matcher.load_from_dataframe(df_coherent)
    assert len(participants) == 1
    
    # Test avec des données incohérentes (devrait être accepté avec un warning)
    df_incoherent = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'birth_year': current_year - 30,  # Incohérence de 5 ans
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    
    participants = matcher.load_from_dataframe(df_incoherent)
    assert len(participants) == 1  # Devrait être accepté malgré l'incohérence

def test_invalid_date_format_handling():
    """Test de la gestion des formats de date invalides"""
    matcher = ParazarMatcher()
    
    # Test avec une date invalide
    df_invalid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': 'INVALID',
        'experience_city': 'Paris'
    }])
    
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df_invalid)

def test_load_parazar_data_file_error():
    """Test de la gestion des erreurs de fichier"""
    with patch('pandas.read_csv', side_effect=FileNotFoundError("Fichier non trouvé")):
        with pytest.raises(FileNotFoundError):
            load_parazar_data("nonexistent.csv")

def test_missing_gender():
    """Test de la gestion des genres manquants"""
    matcher = ParazarMatcher()
    
    df_missing = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': '',  # Genre manquant
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df_missing)

def test_create_optimal_groups_returns_structure():
    """Test de la structure de retour de create_optimal_groups"""
    matcher = ParazarMatcher()
    
    # Créer un ensemble minimal de participants
    participants_data = [
        generate_participant(f"F{i}", 25, "f") for i in range(2)
    ] + [
        generate_participant(f"M{i}", 27, "m") for i in range(2)
    ]
    
    df = pd.DataFrame(participants_data)
    participants = matcher.load_from_dataframe(df)
    
    status, results = matcher.create_optimal_groups(participants)
    
    assert isinstance(status, MatchingStatus)
    assert isinstance(results, dict)
    assert "groups" in results
    assert "unmatched" in results
    assert "stats" in results

def test_age_validation():
    """Test de la validation des âges"""
    matcher = ParazarMatcher()
    
    # Test avec un âge valide
    df_valid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    participants = matcher.load_from_dataframe(df_valid)
    assert len(participants) == 1
    assert participants[0].age == 25

    # Test avec un âge invalide
    df_invalid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 'invalid',
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df_invalid)

def test_gender_validation():
    """Test de la validation des genres"""
    matcher = ParazarMatcher()
    
    # Test avec un genre valide
    df_valid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    participants = matcher.load_from_dataframe(df_valid)
    assert len(participants) == 1
    assert participants[0].gender == 'M'

    # Test avec un genre invalide
    df_invalid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': 'invalid',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df_invalid)

def test_email_validation():
    """Test de la validation des emails"""
    matcher = ParazarMatcher()
    
    # Test avec un email valide
    df_valid = pd.DataFrame([{
        'email': 'test@example.com',
        'first_name': 'Test',
        'age': 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    participants = matcher.load_from_dataframe(df_valid)
    assert len(participants) == 1
    assert participants[0].email == 'test@example.com'

    # Test avec un email invalide
    df_invalid = pd.DataFrame([{
        'email': 'invalid-email',
        'first_name': 'Test',
        'age': 25,
        'gender': 'm',
        'experience_name': 'Test',
        'experience_date': '2024-01-01',
        'experience_city': 'Paris'
    }])
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df_invalid)

def test_group_creation():
    """Test de la création de groupes"""
    matcher = ParazarMatcher()
    
    # Créer un ensemble de participants pour former un groupe valide
    participants_data = [
        generate_participant(f"F{i}", 25, "f") for i in range(3)
    ] + [
        generate_participant(f"M{i}", 27, "m") for i in range(3)
    ]
    
    df = pd.DataFrame(participants_data)
    participants = matcher.load_from_dataframe(df)
    
    status, results = matcher.create_optimal_groups(participants)
    
    assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
    assert len(results["groups"]) > 0
    
    # Vérifier les contraintes des groupes
    for group in results["groups"]:
        assert len(group.participants) >= matcher.min_group_size
        assert len(group.participants) <= matcher.max_group_size
        assert group.gender_balance.get('F', 0) >= matcher.min_females_per_group
        assert group.age_spread <= matcher.max_age_spread

def test_duplicate_email_handling():
    """Test de la gestion des emails en doublon"""
    matcher = ParazarMatcher()
    
    # Créer des données avec un email en doublon
    df = pd.DataFrame([
        {
            'email': 'test@example.com',
            'first_name': 'Test1',
            'age': 25,
            'gender': 'm',
            'experience_name': 'Test',
            'experience_date': '2024-01-01',
            'experience_city': 'Paris'
        },
        {
            'email': 'test@example.com',
            'first_name': 'Test2',
            'age': 26,
            'gender': 'm',
            'experience_name': 'Test',
            'experience_date': '2024-01-01',
            'experience_city': 'Paris'
        }
    ])
    
    participants = matcher.load_from_dataframe(df)
    assert len(participants) == 1  # Seul le premier participant devrait être conservé 