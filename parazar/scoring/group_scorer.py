"""
Module de scoring pour les groupes Parazar.
"""
from typing import List, Dict, Any
from ..models import Group, Participant

class GroupScorer:
    """Calcule les scores pour les groupes de participants."""
    
    @staticmethod
    def calculate_group_score(group: Group) -> float:
        """
        Calcule le score d'un groupe.
        
        Args:
            group: Le groupe à évaluer
            
        Returns:
            float: Score du groupe entre 0 et 1
        """
        if not group.is_valid:
            return 0.0
            
        scores = [
            GroupScorer._calculate_gender_diversity_score(group),
            GroupScorer._calculate_age_diversity_score(group),
            GroupScorer._calculate_size_score(group)
        ]
        
        return sum(scores) / len(scores)
    
    @staticmethod
    def _calculate_gender_diversity_score(group: Group) -> float:
        """
        Calcule le score de diversité des genres.
        
        Args:
            group: Le groupe à évaluer
            
        Returns:
            float: Score entre 0 et 1
        """
        genders = {p.gender for p in group.participants}
        return min(len(genders) / 3, 1.0)  # Score max si 3 genres ou plus
    
    @staticmethod
    def _calculate_age_diversity_score(group: Group) -> float:
        """
        Calcule le score de diversité des âges.
        
        Args:
            group: Le groupe à évaluer
            
        Returns:
            float: Score entre 0 et 1
        """
        age_buckets = {p.age_bucket for p in group.participants if p.age_bucket}
        return min(len(age_buckets) / 4, 1.0)  # Score max si 4 tranches d'âge ou plus
    
    @staticmethod
    def _calculate_size_score(group: Group) -> float:
        """
        Calcule le score basé sur la taille du groupe.
        
        Args:
            group: Le groupe à évaluer
            
        Returns:
            float: Score entre 0 et 1
        """
        size = len(group.participants)
        if size < 3:
            return 0.0
        if size > 5:
            return 0.5
        return 1.0  # Score max pour les groupes de 3 à 5 participants 