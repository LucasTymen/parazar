import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from calculator import (
    ParazarMatcher, Participant, Group, MatchingStatus,
    load_parazar_data, export_groups_to_json
)

# ============================================================================
# FIXTURES DE DONNÉES
# ============================================================================

@pytest.fixture
def basic_participant():
    """Participant de base valide"""
    return Participant(
        email="test@parazar.com",
        first_name="Test",
        gender="F",
        age=25,
        topics_conversations=["voyage", "cuisine"],
        introverted_degree=0.5
    )

@pytest.fixture
def valid_group_data():
    """Données pour créer un groupe valide (2F + 2M)"""
    return pd.DataFrame([
        {
            'email': 'alice@test.com', 'first_name': 'Alice', 'age': 28, 'gender': 'f',
            'topics_conversations': 'voyage,cuisine,art', 'introverted_degree': 0.3,
            'experience_name': 'Afterwork Paris', 'experience_date': '2025-08-15', 
            'experience_city': 'Paris', 'job_field': 'Marketing'
        },
        {
            'email': 'bob@test.com', 'first_name': 'Bob', 'age': 30, 'gender': 'm',
            'topics_conversations': 'voyage,tech,sport', 'introverted_degree': 0.6,
            'experience_name': 'Afterwork Paris', 'experience_date': '2025-08-15',
            'experience_city': 'Paris', 'job_field': 'Tech'
        },
        {
            'email': 'clara@test.com', 'first_name': 'Clara', 'age': 27, 'gender': 'f',
            'topics_conversations': 'art,cuisine,musique', 'introverted_degree': 0.2,
            'experience_name': 'Afterwork Paris', 'experience_date': '2025-08-15',
            'experience_city': 'Paris', 'job_field': 'Design'
        },
        {
            'email': 'david@test.com', 'first_name': 'David', 'age': 31, 'gender': 'm',
            'topics_conversations': 'tech,sport,finance', 'introverted_degree': 0.5,
            'experience_name': 'Afterwork Paris', 'experience_date': '2025-08-15',
            'experience_city': 'Paris', 'job_field': 'Finance'
        }
    ])

@pytest.fixture
def large_dataset():
    """Dataset plus large pour tester la création de plusieurs groupes"""
    data = []
    for i in range(24):  # 24 participants = 4 groupes de 6
        gender = 'f' if i % 2 == 0 else 'm'
        data.append({
            'email': f'user{i}@test.com',
            'first_name': f'User{i}',
            'age': 25 + (i % 15),  # Ages entre 25 et 39
            'gender': gender,
            'topics_conversations': f'topic{i%5},topic{(i+1)%5}',
            'introverted_degree': 0.1 + (i % 9) * 0.1,
            'experience_name': 'Grand Event',
            'experience_date': '2025-09-01',
            'experience_city': 'Paris',
            'job_field': f'Field{i%6}'
        })
    return pd.DataFrame(data)

@pytest.fixture
def matcher():
    """Instance de ParazarMatcher avec paramètres par défaut"""
    return ParazarMatcher()

# ============================================================================
# TESTS DE LA CLASSE PARTICIPANT
# ============================================================================

class TestParticipant:
    
    def test_participant_creation(self, basic_participant):
        """Test création d'un participant valide"""
        assert basic_participant.email == "test@parazar.com"
        assert basic_participant.first_name == "Test"
        assert basic_participant.gender == "F"
        assert basic_participant.age == 25
        assert basic_participant.topics_conversations == ["voyage", "cuisine"]
        assert basic_participant.introverted_degree == 0.5
    
    def test_social_score_calculation(self):
        """Test calcul du score social"""
        # Très introverti (0.9) -> score social bas
        introvert = Participant(
            email="intro@test.com", first_name="Intro", gender="M", age=30,
            introverted_degree=0.9
        )
        assert introvert.social_score == 1.0  # (1-0.9)*10 = 1.0
        
        # Très extraverti (0.1) -> score social élevé
        extravert = Participant(
            email="extra@test.com", first_name="Extra", gender="F", age=28,
            introverted_degree=0.1
        )
        assert extravert.social_score == 9.0  # (1-0.1)*10 = 9.0
    
    def test_compatibility_topics_post_init(self):
        """Test que les topics sont convertis en set"""
        participant = Participant(
            email="test@test.com", first_name="Test", gender="F", age=25,
            topics_conversations=["voyage", "cuisine", "voyage"]  # Doublons
        )
        assert participant.compatibility_topics == {"voyage", "cuisine"}
    
    def test_participant_with_minimal_data(self):
        """Test création avec données minimales"""
        participant = Participant(
            email="min@test.com", first_name="Min", gender="M", age=18
        )
        assert participant.social_score == 5.0  # introverted_degree=0.5 par défaut
        assert participant.compatibility_topics == set()
        assert participant.topics_conversations == []

# ============================================================================
# TESTS DE LA CLASSE GROUP
# ============================================================================

class TestGroup:
    
    def test_empty_group_creation(self):
        """Test création d'un groupe vide"""
        group = Group(id="test_group")
        assert group.id == "test_group"
        assert len(group.participants) == 0
        assert group.compatibility_score == 0.0
        assert group.age_spread == 0.0
        assert not group.is_valid  # Groupe vide invalide
    
    def test_group_with_participants(self):
        """Test groupe avec participants"""
        p1 = Participant(email="p1@test.com", first_name="P1", gender="F", age=25)
        p2 = Participant(email="p2@test.com", first_name="P2", gender="M", age=30)
        
        group = Group(id="test", participants=[p1, p2])
        assert len(group.participants) == 2
        assert group.age_spread == 5  # 30-25
        assert group.gender_balance == {'F': 1, 'M': 1}
    
    def test_group_validity_constraints(self):
        """Test contraintes de validité d'un groupe"""
        # Groupe avec 2 femmes + 2 hommes (valide)
        participants = [
            Participant(email=f"f{i}@test.com", first_name=f"F{i}", gender="F", age=25+i)
            for i in range(2)
        ] + [
            Participant(email=f"m{i}@test.com", first_name=f"M{i}", gender="M", age=25+i)
            for i in range(2)
        ]
        
        group = Group(id="valid", participants=participants)
        assert group.is_valid
        assert len(group.participants) >= 4
        assert group.gender_balance.get('F', 0) >= 2
        assert group.age_spread <= 6
    
    def test_group_invalid_too_few_participants(self):
        """Test groupe invalide - trop peu de participants"""
        participants = [
            Participant(email="p1@test.com", first_name="P1", gender="F", age=25),
            Participant(email="p2@test.com", first_name="P2", gender="M", age=26)
        ]
        group = Group(id="invalid", participants=participants)
        assert not group.is_valid  # Moins de 4 participants
    
    def test_group_invalid_too_few_females(self):
        """Test groupe invalide - pas assez de femmes"""
        participants = [
            Participant(email="f1@test.com", first_name="F1", gender="F", age=25),
            Participant(email="m1@test.com", first_name="M1", gender="M", age=26),
            Participant(email="m2@test.com", first_name="M2", gender="M", age=27),
            Participant(email="m3@test.com", first_name="M3", gender="M", age=28)
        ]
        group = Group(id="invalid", participants=participants)
        assert not group.is_valid  # Moins de 2 femmes
    
    def test_group_invalid_age_spread(self):
        """Test groupe invalide - écart d'âge trop important"""
        participants = [
            Participant(email="f1@test.com", first_name="F1", gender="F", age=20),
            Participant(email="f2@test.com", first_name="F2", gender="F", age=21),
            Participant(email="m1@test.com", first_name="M1", gender="M", age=30),
            Participant(email="m2@test.com", first_name="M2", gender="M", age=35)
        ]
        group = Group(id="invalid", participants=participants)
        assert not group.is_valid  # Écart d'âge > 6 ans
    
    def test_female_age_constraint(self):
        """Test contrainte d'âge des femmes"""
        # Cas valide: hommes plus âgés
        participants = [
            Participant(email="f1@test.com", first_name="F1", gender="F", age=25),
            Participant(email="f2@test.com", first_name="F2", gender="F", age=26),
            Participant(email="m1@test.com", first_name="M1", gender="M", age=30),
            Participant(email="m2@test.com", first_name="M2", gender="M", age=32)
        ]
        group = Group(id="valid_age", participants=participants)
        assert group.female_age_constraint_ok
        
        # Cas invalide: femme la plus âgée
        participants[0].age = 35  # Femme plus âgée que tous
        group = Group(id="invalid_age", participants=participants)
        assert not group.female_age_constraint_ok

# ============================================================================
# TESTS DE LA CLASSE PARAZARMATCHER
# ============================================================================

class TestParazarMatcher:
    
    def test_matcher_initialization(self):
        """Test initialisation du matcher"""
        matcher = ParazarMatcher(min_group_size=6, max_group_size=10)
        assert matcher.min_group_size == 6
        assert matcher.max_group_size == 10
        assert matcher.max_age_spread == 6  # Valeur par défaut
        assert matcher.min_females_per_group == 2
        assert len(matcher.groups) == 0
        assert len(matcher.unmatched) == 0
    
    def test_load_from_dataframe_valid_data(self, matcher, valid_group_data):
        """Test chargement de données valides"""
        participants = matcher.load_from_dataframe(valid_group_data)
        
        assert len(participants) == 4
        assert all(isinstance(p, Participant) for p in participants)
        assert all(p.age >= 18 and p.age <= 65 for p in participants)
        assert all(p.gender in ['M', 'F'] for p in participants)
        assert all(p.email for p in participants)
    
    def test_load_from_dataframe_invalid_data(self, matcher):
        """Test chargement avec données invalides"""
        invalid_data = pd.DataFrame([
            {'email': '', 'first_name': 'Invalid', 'age': 15, 'gender': 'x'},  # Trop jeune, genre invalide
            {'email': 'valid@test.com', 'first_name': 'Valid', 'age': 25, 'gender': 'f'},
            {'email': 'old@test.com', 'first_name': 'Old', 'age': 70, 'gender': 'm'}  # Trop âgé
        ])
        
        participants = matcher.load_from_dataframe(invalid_data)
        assert len(participants) == 1  # Seul le participant valide
        assert participants[0].email == 'valid@test.com'
    
    def test_load_from_dataframe_empty(self, matcher):
        """Test chargement DataFrame vide"""
        empty_df = pd.DataFrame()
        participants = matcher.load_from_dataframe(empty_df)
        assert len(participants) == 0
    
    def test_create_optimal_groups_insufficient_data(self, matcher):
        """Test avec données insuffisantes"""
        participants = [
            Participant(email="p1@test.com", first_name="P1", gender="F", age=25),
            Participant(email="p2@test.com", first_name="P2", gender="M", age=26)
        ]
        
        status, results = matcher.create_optimal_groups(participants)
        assert status == MatchingStatus.INSUFFICIENT_DATA
        assert "error" in results
        assert results["required"] == 4
        assert results["available"] == 2
    
    def test_create_optimal_groups_success(self, matcher, valid_group_data):
        """Test création réussie de groupes"""
        participants = matcher.load_from_dataframe(valid_group_data)
        status, results = matcher.create_optimal_groups(participants)
        
        assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
        assert "groups" in results
        assert "unmatched" in results
        assert "stats" in results
        assert results["segments_processed"] == 1
        
        if results["groups"]:
            for group in results["groups"]:
                assert group.is_valid
                assert group.female_age_constraint_ok
    
    def test_create_optimal_groups_large_dataset(self, matcher, large_dataset):
        """Test avec un grand dataset"""
        participants = matcher.load_from_dataframe(large_dataset)
        status, results = matcher.create_optimal_groups(participants)
        
        assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
        assert len(results["groups"]) >= 3  # Au moins 3 groupes attendus
        assert results["stats"]["matching_rate"] > 50  # Au moins 50% appariés
    
    def test_segmentation_by_experience(self, matcher):
        """Test segmentation par expérience/date/ville"""
        data = pd.DataFrame([
            {'email': 'p1@test.com', 'first_name': 'P1', 'age': 25, 'gender': 'f',
             'experience_name': 'Event A', 'experience_date': '2025-08-01', 'experience_city': 'Paris'},
            {'email': 'p2@test.com', 'first_name': 'P2', 'age': 26, 'gender': 'm',
             'experience_name': 'Event B', 'experience_date': '2025-08-01', 'experience_city': 'Paris'},
        ])
        
        participants = matcher.load_from_dataframe(data)
        segments = matcher._segment_participants(participants)
        
        assert len(segments) == 2  # Deux segments différents
        assert 'Event A_2025-08-01_Paris' in segments
        assert 'Event B_2025-08-01_Paris' in segments
    
    def test_find_replacement_success(self, matcher):
        """Test remplacement réussi d'un participant"""
        # Groupe existant
        group_participants = [
            Participant(email="f1@test.com", first_name="F1", gender="F", age=25,
                       experience_name="Event", experience_date="2025-08-01", experience_city="Paris"),
            Participant(email="f2@test.com", first_name="F2", gender="F", age=26,
                       experience_name="Event", experience_date="2025-08-01", experience_city="Paris"),
            Participant(email="m1@test.com", first_name="M1", gender="M", age=28,
                       experience_name="Event", experience_date="2025-08-01", experience_city="Paris"),
            Participant(email="m2@test.com", first_name="M2", gender="M", age=30,
                       experience_name="Event", experience_date="2025-08-01", experience_city="Paris")
        ]
        
        group = Group(id="test_group", participants=group_participants,
                     experience_name="Event", experience_date="2025-08-01", experience_city="Paris")
        
        # Participant qui se désiste
        leaving_participant = group_participants[2]  # M1
        
        # Pool de remplaçants
        replacement_candidates = [
            Participant(email="replacement@test.com", first_name="Replacement", gender="M", age=29,
                       experience_name="Event", experience_date="2025-08-01", experience_city="Paris"),
            Participant(email="wrong_event@test.com", first_name="Wrong", gender="M", age=29,
                       experience_name="Other Event", experience_date="2025-08-01", experience_city="Paris")
        ]
        
        replacement = matcher.find_replacement(group, leaving_participant, replacement_candidates)
        
        assert replacement is not None
        assert replacement.email == "replacement@test.com"
        assert replacement.experience_name == "Event"
    
    def test_find_replacement_no_candidates(self, matcher):
        """Test remplacement sans candidats disponibles"""
        group = Group(id="test_group", participants=[])
        leaving_participant = Participant(email="leaving@test.com", first_name="Leaving", gender="M", age=25)
        
        replacement = matcher.find_replacement(group, leaving_participant, [])
        assert replacement is None
    
    def test_calculate_global_stats(self, matcher):
        """Test calcul des statistiques globales"""
        # Créer des groupes factices
        group1 = Group(id="g1", participants=[
            Participant(email="p1@test.com", first_name="P1", gender="F", age=25),
            Participant(email="p2@test.com", first_name="P2", gender="M", age=26)
        ])
        group2 = Group(id="g2", participants=[
            Participant(email="p3@test.com", first_name="P3", gender="F", age=27),
            Participant(email="p4@test.com", first_name="P4", gender="M", age=28)
        ])
        
        unmatched = [
            Participant(email="unmatched@test.com", first_name="Unmatched", gender="F", age=29)
        ]
        
        stats = matcher._calculate_global_stats([group1, group2], unmatched)
        
        assert stats["total_participants"] == 5
        assert stats["groups_created"] == 2
        assert stats["participants_matched"] == 4
        assert stats["participants_unmatched"] == 1
        assert stats["matching_rate"] == 80.0  # 4/5 * 100
        assert stats["avg_group_size"] == 2.0
    
    def test_determine_status(self, matcher):
        """Test détermination du statut global"""
        # Test SUCCESS (>= 90%)
        results_success = {"stats": {"matching_rate": 95, "valid_groups": 3}}
        assert matcher._determine_status(results_success) == MatchingStatus.SUCCESS
        
        # Test PARTIAL_SUCCESS (>= 70%)
        results_partial = {"stats": {"matching_rate": 80, "valid_groups": 2}}
        assert matcher._determine_status(results_partial) == MatchingStatus.PARTIAL_SUCCESS
        
        # Test FAILED_CONSTRAINTS (0 groupes valides)
        results_failed = {"stats": {"matching_rate": 60, "valid_groups": 0}}
        assert matcher._determine_status(results_failed) == MatchingStatus.FAILED_CONSTRAINTS

# ============================================================================
# TESTS DES FONCTIONS UTILITAIRES
# ============================================================================

class TestUtilityFunctions:
    
    @patch("pandas.read_csv")
    def test_load_parazar_data_success(self, mock_read_csv):
        """Test chargement réussi de données CSV"""
        mock_df = pd.DataFrame([
            {'email': 'test@test.com', 'first_name': 'Test', 'age': 25, 'gender': 'f'}
        ])
        mock_read_csv.return_value = mock_df
        
        result = load_parazar_data("test.csv")
        
        mock_read_csv.assert_called_once_with("test.csv")
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch("pandas.read_csv")
    def test_load_parazar_data_file_error(self, mock_read_csv):
        """Test gestion d'erreur lors du chargement CSV"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        result = load_parazar_data("nonexistent.csv")
        
        assert result.empty
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_export_groups_to_json_success(self, mock_json_dump, mock_file):
        """Test export réussi vers JSON"""
        # Créer des groupes factices
        participants = [
            Participant(email="p1@test.com", first_name="P1", gender="F", age=25,
                       parazar_partner_id="PZ001", telephone="0123456789"),
            Participant(email="p2@test.com", first_name="P2", gender="M", age=26,
                       parazar_partner_id="PZ002", telephone="0123456790")
        ]
        
        groups = [
            Group(id="group1", participants=participants,
                 experience_name="Test Event", experience_date="2025-08-01", experience_city="Paris")
        ]
        
        export_groups_to_json(groups, "output.json")
        
        mock_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once()
        
        # Vérifier la structure des données exportées
        exported_data = mock_json_dump.call_args[0][0]
        assert len(exported_data) == 1
        assert exported_data[0]["id"] == "group1"
        assert len(exported_data[0]["participants"]) == 2
        assert exported_data[0]["participants"][0]["email"] == "p1@test.com"
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_export_groups_to_json_error(self, mock_file):
        """Test gestion d'erreur lors de l'export JSON"""
        groups = []
        
        # Ne doit pas lever d'exception
        export_groups_to_json(groups, "output.json")
        mock_file.assert_called_once()

# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    
    def test_full_workflow(self):
        """Test du workflow complet"""
        # Données de test
        data = pd.DataFrame([
            {'email': f'user{i}@test.com', 'first_name': f'User{i}', 
             'age': 25 + (i % 10), 'gender': 'f' if i % 2 == 0 else 'm',
             'topics_conversations': 'voyage,tech,sport', 'introverted_degree': 0.5,
             'experience_name': 'Integration Test', 'experience_date': '2025-08-01',
             'experience_city': 'Paris', 'job_field': 'Tech'}
            for i in range(12)  # 12 participants = 2 groupes
        ])
        
        # Workflow complet
        matcher = ParazarMatcher()
        participants = matcher.load_from_dataframe(data)
        status, results = matcher.create_optimal_groups(participants)
        
        # Vérifications
        assert len(participants) == 12
        assert status in [MatchingStatus.SUCCESS, MatchingStatus.PARTIAL_SUCCESS]
        assert len(results["groups"]) >= 1
        assert results["stats"]["matching_rate"] > 50
        
        # Test de remplacement
        if results["groups"]:
            first_group = results["groups"][0]
            if first_group.participants:
                leaving_participant = first_group.participants[0]
                replacement = matcher.find_replacement(
                    first_group, leaving_participant, results.get("unmatched", [])
                )
                # Le remplacement peut être None si pas de candidats compatibles

# ============================================================================
# TESTS DE PERFORMANCE ET EDGE CASES
# ============================================================================

class TestEdgeCases:
    
    def test_participant_with_empty_topics(self):
        """Test participant sans topics"""
        participant = Participant(
            email="empty@test.com", first_name="Empty", gender="F", age=25,
            topics_conversations=[]
        )
        assert participant.compatibility_topics == set()
        assert participant.social_score >= 0
    
    def test_group_with_extreme_ages(self):
        """Test groupe avec âges extrêmes"""
        participants = [
            Participant(email="young@test.com", first_name="Young", gender="F", age=18),
            Participant(email="old@test.com", first_name="Old", gender="M", age=65)
        ]
        group = Group(id="extreme", participants=participants)
        assert group.age_spread == 47
        assert not group.is_valid  # Écart trop important
    
    def test_matcher_with_custom_constraints(self):
        """Test matcher avec contraintes personnalisées"""
        matcher = ParazarMatcher(
            min_group_size=6, max_group_size=8, 
            max_age_spread=3, min_females_per_group=3
        )
        
        assert matcher.min_group_size == 6
        assert matcher.max_group_size == 8
        assert matcher.max_age_spread == 3
        assert matcher.min_females_per_group == 3
    
    def test_dataframe_with_nan_values(self, matcher):
        """Test DataFrame avec valeurs NaN"""
        data = pd.DataFrame([
            {'email': 'valid@test.com', 'first_name': 'Valid', 'age': 25, 'gender': 'f'},
            {'email': None, 'first_name': 'Invalid', 'age': 30, 'gender': 'm'},  # Email manquant
            {'email': 'another@test.com', 'first_name': None, 'age': 35, 'gender': 'f'}  # Nom manquant
        ])
        
        participants = matcher.load_from_dataframe(data)
        # Seul le premier participant devrait être chargé
        assert len(participants) <= 1

def test_email_case_normalization():
    row = {
        "email": "TEST@EXAMPLE.COM",
        "first_name": "Alice",
        "age": 30,
        "gender": "f"
    }
    matcher = ParazarMatcher()
    participant = matcher._create_participant_from_row(row)
    assert participant.email == "test@example.com"

def test_topics_cleaning_and_lowercase():
    row = {
        "email": "user@example.com",
        "first_name": "Bob",
        "age": 28,
        "gender": "m",
        "topics_conversations": "   Sport,  Musique  ,  Voyage  "
    }
    matcher = ParazarMatcher()
    participant = matcher._create_participant_from_row(row)
    assert participant.topics_conversations == ["sport", "musique", "voyage"]

import pytest

def test_introverted_degree_out_of_bounds():
    row = {
        "email": "edge@example.com",
        "first_name": "Edge",
        "age": 25,
        "gender": "f",
        "introverted_degree": -0.3
    }
    matcher = ParazarMatcher()
    participant = matcher._create_participant_from_row(row)
    assert 0.0 <= participant.introverted_degree <= 1.0

import pytest

def test_invalid_experience_date_raises():
    row = {
        "email": "test@invalid.com",
        "first_name": "Date",
        "age": 29,
        "gender": "f",
        "experience_date": "12-2025"  # format non supporté
    }
    matcher = ParazarMatcher()
    with pytest.raises(Exception):
        matcher._create_participant_from_row(row)

def test_gender_normalization_unexpected_value():
    row = {
        "email": "strange@example.com",
        "first_name": "Case",
        "age": 22,
        "gender": "Féminin"
    }
    matcher = ParazarMatcher()
    participant = matcher._create_participant_from_row(row)
    assert participant.gender == "FÉMININ"

def test_create_optimal_groups_returns_structure():
    from calculator import ParazarMatcher

    matcher = ParazarMatcher()
    dummy_participants = []

    status, result = matcher.create_optimal_groups(dummy_participants)

    assert status == "success"
    assert isinstance(result, dict)
    assert "groups" in result
    assert "unmatched" in result


def test_invalid_email():
    df = pd.DataFrame([{
        'email': 'invalidemail.com', 'first_name': 'BadMail', 'age': 30, 'gender': 'm',
        'topics_conversations': 'tech', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    with pytest.raises(ValueError):
        matcher.load_from_dataframe(df)

def test_empty_topics():
    df = pd.DataFrame([{
        'email': 'topic@test.com', 'first_name': 'EmptyTopic', 'age': 30, 'gender': 'f',
        'topics_conversations': '', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert participants[0].compatibility_topics == set()

def test_invalid_email(caplog):
    df = pd.DataFrame([{
        'email': 'invalidemail.com',  # pas de @ → invalide
        'first_name': 'BadMail',
        'age': 30,
        'gender': 'm',
        'topics_conversations': 'tech',
        'introverted_degree': 0.5,
        'experience_name': 'Test',
        'experience_date': '2025-08-01',
        'experience_city': 'Paris'
    }])

    matcher = ParazarMatcher()

    with caplog.at_level("WARNING"):
        participants = matcher.load_from_dataframe(df)

    # Le participant doit être ignoré → liste vide
    assert len(participants) == 0

    # Vérification du message de log explicite
    assert any("Email invalide" in message for message in caplog.text.splitlines())


if __name__ == "__main__":
    # Configuration pour pytest
    pytest.main([__file__, "-v", "--tb=short"])