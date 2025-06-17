"""
Package Parazar pour la gestion des participants et des groupes.
"""
from .models import Participant, Group
from .validators import (
    ParticipantValidator,
    ValidationResult,
    EmailValidator,
    GenderValidator,
    AgeValidator,
    DateValidator
)
from .scoring import GroupScorer
from .matcher import ParazarMatcher, MatchingStatus, MatchingResult

__version__ = '0.1.0'

__all__ = [
    # Models
    'Participant',
    'Group',
    
    # Validators
    'ParticipantValidator',
    'ValidationResult',
    'EmailValidator',
    'GenderValidator',
    'AgeValidator',
    'DateValidator',
    
    # Scoring
    'GroupScorer',
    
    # Matcher
    'ParazarMatcher',
    'MatchingStatus',
    'MatchingResult'
] 