import pytest
import pandas as pd
from calculator import ParazarMatcher, MatchingStatus

# Cas base invalide
def test_introverted_degree_out_of_bounds():
    df = pd.DataFrame([{
        'email': 'weird@case.com', 'first_name': 'Odd', 'age': 30, 'gender': 'm',
        'topics_conversations': 'tech', 'introverted_degree': 1.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert participants[0].introverted_degree <= 1.1

def test_invalid_date_format_handling():
    df = pd.DataFrame([{
        'email': 'bad@date.com', 'first_name': 'Bug', 'age': 25, 'gender': 'f',
        'topics_conversations': 'tech', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': 'INVALID', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    with pytest.raises(Exception):
        matcher.load_from_dataframe(df)

def test_missing_gender():
    df = pd.DataFrame([{
        'email': 'no@gender.com', 'first_name': 'Mystery', 'age': 30,
        'topics_conversations': 'tech', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert participants[0].gender in ['m', 'f', 'x']

def test_duplicate_email_detection():
    df = pd.DataFrame([
        {'email': 'dup@x.com', 'first_name': 'X1', 'age': 22, 'gender': 'f',
         'topics_conversations': 'tech', 'introverted_degree': 0.3,
         'experience_name': 'X', 'experience_date': '2025-08-01', 'experience_city': 'Paris'},
        {'email': 'dup@x.com', 'first_name': 'X2', 'age': 23, 'gender': 'm',
         'topics_conversations': 'tech', 'introverted_degree': 0.6,
         'experience_name': 'X', 'experience_date': '2025-08-01', 'experience_city': 'Paris'}
    ])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert len(set(p.email for p in participants)) == len(participants)

def test_age_vs_birth_year_coherence():
    df = pd.DataFrame([{
        'email': 'age@off.com', 'first_name': 'Time', 'age': 20, 'birth_year': 1990,
        'gender': 'f', 'topics_conversations': 'science', 'introverted_degree': 0.4,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    actual_age = 2025 - df.iloc[0]['birth_year']
    assert abs(participants[0].age - actual_age) <= 1

import pytest
import pandas as pd
from calculator import ParazarMatcher, MatchingStatus

# Cas base invalide
def test_introverted_degree_out_of_bounds():
    df = pd.DataFrame([{
        'email': 'weird@case.com', 'first_name': 'Odd', 'age': 30, 'gender': 'm',
        'topics_conversations': 'tech', 'introverted_degree': 1.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert participants[0].introverted_degree <= 1.1

def test_invalid_date_format_handling():
    df = pd.DataFrame([{
        'email': 'bad@date.com', 'first_name': 'Bug', 'age': 25, 'gender': 'f',
        'topics_conversations': 'tech', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': 'INVALID', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    with pytest.raises(Exception):
        matcher.load_from_dataframe(df)

def test_missing_gender():
    df = pd.DataFrame([{
        'email': 'no@gender.com', 'first_name': 'Mystery', 'age': 30,
        'topics_conversations': 'tech', 'introverted_degree': 0.5,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert participants[0].gender in ['m', 'f', 'x']

def test_duplicate_email_detection():
    df = pd.DataFrame([
        {'email': 'dup@x.com', 'first_name': 'X1', 'age': 22, 'gender': 'f',
         'topics_conversations': 'tech', 'introverted_degree': 0.3,
         'experience_name': 'X', 'experience_date': '2025-08-01', 'experience_city': 'Paris'},
        {'email': 'dup@x.com', 'first_name': 'X2', 'age': 23, 'gender': 'm',
         'topics_conversations': 'tech', 'introverted_degree': 0.6,
         'experience_name': 'X', 'experience_date': '2025-08-01', 'experience_city': 'Paris'}
    ])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    assert len(set(p.email for p in participants)) == len(participants)

def test_age_vs_birth_year_coherence():
    df = pd.DataFrame([{
        'email': 'age@off.com', 'first_name': 'Time', 'age': 20, 'birth_year': 1990,
        'gender': 'f', 'topics_conversations': 'science', 'introverted_degree': 0.4,
        'experience_name': 'Test', 'experience_date': '2025-08-01', 'experience_city': 'Paris'
    }])
    matcher = ParazarMatcher()
    participants = matcher.load_from_dataframe(df)
    actual_age = 2025 - df.iloc[0]['birth_year']
    assert abs(participants[0].age - actual_age) <= 1
