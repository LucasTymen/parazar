"""Module de validation pour les données Parazar."""

import re
from datetime import datetime
from typing import Any, Optional, Union

class ParticipantValidator:
    """Classe utilitaire pour la validation des participants."""
    
    @staticmethod
    def is_valid_email(email: Any) -> bool:
        """Vérifie si l'email est valide.
        
        Args:
            email: L'email à valider
            
        Returns:
            bool: True si l'email est valide, False sinon
        """
        if not email or not isinstance(email, str):
            return False
            
        email = email.strip().lower()
        if email in ['nan', 'none', '']:
            return False
            
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_float(x: Any, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """Vérifie si la valeur est un float valide dans l'intervalle [min_val, max_val].
        
        Args:
            x: La valeur à valider
            min_val: Valeur minimale (incluse)
            max_val: Valeur maximale (incluse)
            
        Returns:
            bool: True si la valeur est valide, False sinon
        """
        try:
            if x is None:
                return False
            val = float(x)
            return min_val <= val <= max_val
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def is_valid_str(x: Any) -> bool:
        """Vérifie si la valeur est une chaîne non vide et valide.
        
        Args:
            x: La valeur à valider
            
        Returns:
            bool: True si la valeur est valide, False sinon
        """
        if not x or not isinstance(x, str):
            return False
        return x.strip().lower() not in ['nan', 'none', '']
    
    @staticmethod
    def is_valid_age(age: Any) -> bool:
        """Vérifie si l'âge est valide (entre 18 et 100).
        
        Args:
            age: L'âge à valider
            
        Returns:
            bool: True si l'âge est valide, False sinon
        """
        try:
            if age is None:
                return False
            val = int(float(age))
            return 18 <= val <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def is_valid_gender(gender: Any) -> bool:
        """Vérifie si le genre est valide (M/F).
        
        Args:
            gender: Le genre à valider
            
        Returns:
            bool: True si le genre est valide, False sinon
        """
        if not gender or not isinstance(gender, str):
            return False
        return gender.upper() in ['M', 'F']
    
    @staticmethod
    def normalize_gender(gender: str) -> str:
        """Normalise le genre (M/F).
        
        Args:
            gender: Le genre à normaliser
            
        Returns:
            str: Le genre normalisé (M/F)
            
        Raises:
            ValueError: Si le genre n'est pas valide
        """
        if not ParticipantValidator.is_valid_gender(gender):
            raise ValueError(f"Genre invalide: {gender}")
        return gender.upper()

def validate_date_format(date_str: Optional[str], format_str: str = "%Y-%m-%d") -> bool:
    """Vérifie si la date est dans le format attendu.
    
    Args:
        date_str: La date à valider
        format_str: Le format attendu
        
    Returns:
        bool: True si la date est valide, False sinon
    """
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False

def clean_topics(topics: Union[str, list, None]) -> list:
    """Nettoie et normalise la liste des topics.
    
    Args:
        topics: Les topics à nettoyer (str ou list)
        
    Returns:
        list: Liste des topics nettoyés
    """
    if not topics:
        return []
        
    if isinstance(topics, str):
        topics = topics.split(',')
        
    return [x.strip().lower() for x in topics if x and str(x).strip().lower() not in ['nan', 'none', '']] 