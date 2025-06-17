"""
Modèle Group pour Parazar.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .participant import Participant

@dataclass
class Group:
    """Représente un groupe de participants dans le système Parazar."""
    
    participants: List[Participant] = field(default_factory=list)
    score: float = 0.0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def add_participant(self, participant: Participant) -> None:
        """
        Ajoute un participant au groupe.
        
        Args:
            participant: Le participant à ajouter
        """
        self.participants.append(participant)
        self._update_validity()
    
    def remove_participant(self, participant: Participant) -> None:
        """
        Retire un participant du groupe.
        
        Args:
            participant: Le participant à retirer
        """
        if participant in self.participants:
            self.participants.remove(participant)
            self._update_validity()
    
    def _update_validity(self) -> None:
        """Met à jour la validité du groupe en fonction de ses participants."""
        self.validation_errors = []
        
        # Vérifier la taille du groupe
        if len(self.participants) < 3:
            self.validation_errors.append("Le groupe doit contenir au moins 3 participants")
        
        # Vérifier la diversité des genres
        genders = {p.gender for p in self.participants}
        if len(genders) < 2:
            self.validation_errors.append("Le groupe doit contenir au moins 2 genres différents")
        
        # Vérifier la diversité des âges
        age_buckets = {p.age_bucket for p in self.participants if p.age_bucket}
        if len(age_buckets) < 2:
            self.validation_errors.append("Le groupe doit contenir au moins 2 tranches d'âge différentes")
        
        self.is_valid = len(self.validation_errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'instance en dictionnaire.
        
        Returns:
            Dict[str, Any]: Dictionnaire représentant le groupe
        """
        return {
            'participants': [p.to_dict() for p in self.participants],
            'score': self.score,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Group':
        """
        Crée une instance de Group à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire contenant les données du groupe
            
        Returns:
            Group: Instance créée
        """
        group = cls()
        for participant_data in data.get('participants', []):
            group.add_participant(Participant.from_dict(participant_data))
        group.score = data.get('score', 0.0)
        group.is_valid = data.get('is_valid', True)
        group.validation_errors = data.get('validation_errors', [])
        return group
    
    def __str__(self) -> str:
        """Représentation en chaîne de caractères du groupe."""
        return (
            f"Group(participants={len(self.participants)}, "
            f"score={self.score}, "
            f"is_valid={self.is_valid}, "
            f"errors={self.validation_errors})"
        )
    
    def __eq__(self, other: Any) -> bool:
        """Compare deux groupes."""
        if not isinstance(other, Group):
            return False
        return (
            self.participants == other.participants and
            self.score == other.score and
            self.is_valid == other.is_valid and
            self.validation_errors == other.validation_errors
        ) 