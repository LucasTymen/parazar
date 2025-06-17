"""
Modèle Participant pour Parazar.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class Participant:
    """Représente un participant dans le système Parazar."""
    
    email: str
    gender: str
    age: int
    birth_date: Optional[str] = None
    birth_year: Optional[int] = None
    age_bucket: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Participant':
        """
        Crée une instance de Participant à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire contenant les données du participant
            
        Returns:
            Participant: Instance créée
        """
        return cls(
            email=data.get('email', '').strip(),
            gender=data.get('gender', '').strip(),
            age=int(data.get('age', 0)),
            birth_date=data.get('birth_date', '').strip() or None,
            birth_year=int(data.get('birth_year', 0)) if data.get('birth_year') else None,
            age_bucket=data.get('age_bucket')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'instance en dictionnaire.
        
        Returns:
            Dict[str, Any]: Dictionnaire représentant le participant
        """
        return {
            'email': self.email,
            'gender': self.gender,
            'age': self.age,
            'birth_date': self.birth_date,
            'birth_year': self.birth_year,
            'age_bucket': self.age_bucket
        }
    
    def __str__(self) -> str:
        """Représentation en chaîne de caractères du participant."""
        return (
            f"Participant(email='{self.email}', "
            f"gender='{self.gender}', "
            f"age={self.age}, "
            f"birth_date='{self.birth_date}', "
            f"birth_year={self.birth_year}, "
            f"age_bucket='{self.age_bucket}')"
        )
    
    def __eq__(self, other: Any) -> bool:
        """Compare deux participants."""
        if not isinstance(other, Participant):
            return False
        return (
            self.email == other.email and
            self.gender == other.gender and
            self.age == other.age and
            self.birth_date == other.birth_date and
            self.birth_year == other.birth_year and
            self.age_bucket == other.age_bucket
        ) 