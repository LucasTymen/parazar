"""
Module principal pour le matching des participants Parazar.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .models import Participant, Group
from .validators import ParticipantValidator, ValidationResult
from .scoring import GroupScorer

logger = logging.getLogger(__name__)

class MatchingStatus(Enum):
    """Statut du processus de matching."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"

@dataclass
class MatchingResult:
    """Résultat du processus de matching."""
    status: MatchingStatus
    groups: List[Group]
    errors: List[str]
    warnings: List[str]

class ParazarMatcher:
    """Orchestrateur du processus de matching des participants."""
    
    def __init__(self):
        """Initialise le matcher."""
        self.validator = ParticipantValidator()
        self.scorer = GroupScorer()
    
    def match_participants(self, participants_data: List[Dict[str, Any]]) -> MatchingResult:
        """
        Crée des groupes optimaux à partir des participants.
        
        Args:
            participants_data: Liste des données des participants
            
        Returns:
            MatchingResult: Résultat du matching
        """
        # Valider et normaliser les participants
        validated_participants = []
        errors = []
        warnings = []
        
        for data in participants_data:
            result = self.validator.validate(data)
            if result.is_valid:
                participant = Participant.from_dict(result.normalized_data)
                validated_participants.append(participant)
            else:
                errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        if not validated_participants:
            return MatchingResult(
                status=MatchingStatus.FAILURE,
                groups=[],
                errors=errors,
                warnings=warnings
            )
        
        # Créer les groupes
        groups = self._create_groups(validated_participants)
        
        # Calculer les scores
        for group in groups:
            group.score = self.scorer.calculate_group_score(group)
        
        # Déterminer le statut
        status = self._determine_status(groups, errors)
        
        return MatchingResult(
            status=status,
            groups=groups,
            errors=errors,
            warnings=warnings
        )
    
    def _create_groups(self, participants: List[Participant]) -> List[Group]:
        """
        Crée des groupes à partir des participants validés.
        
        Args:
            participants: Liste des participants validés
            
        Returns:
            List[Group]: Liste des groupes créés
        """
        groups = []
        remaining_participants = participants.copy()
        
        while len(remaining_participants) >= 3:
            group = Group()
            
            # Ajouter des participants jusqu'à ce que le groupe soit valide
            while len(group.participants) < 5 and remaining_participants:
                participant = remaining_participants.pop(0)
                group.add_participant(participant)
                
                if group.is_valid:
                    break
            
            if group.is_valid:
                groups.append(group)
            else:
                # Remettre les participants dans la liste
                remaining_participants.extend(group.participants)
                break
        
        return groups
    
    def _determine_status(self, groups: List[Group], errors: List[str]) -> MatchingStatus:
        """
        Détermine le statut du matching.
        
        Args:
            groups: Liste des groupes créés
            errors: Liste des erreurs rencontrées
            
        Returns:
            MatchingStatus: Statut du matching
        """
        if not groups:
            return MatchingStatus.FAILURE
            
        valid_groups = [g for g in groups if g.is_valid]
        if not valid_groups:
            return MatchingStatus.FAILURE
            
        if len(valid_groups) < len(groups):
            return MatchingStatus.PARTIAL_SUCCESS
            
        return MatchingStatus.SUCCESS 