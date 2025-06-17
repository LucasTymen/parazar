"""
Validateur d'email pour Parazar.
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class EmailValidator:
    """Validateur d'email avec messages d'erreur explicites."""
    
    EMAIL_PATTERN = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    @staticmethod
    def validate(email: str) -> Tuple[bool, Optional[str]]:
        """
        Valide un email et retourne un tuple (est_valide, message_erreur).
        
        Args:
            email: L'email à valider
            
        Returns:
            Tuple[bool, Optional[str]]: (True, None) si valide, (False, message) sinon
        """
        if not email:
            return False, "L'email ne peut pas être vide"
            
        email = email.strip().lower()
        
        if not re.match(EmailValidator.EMAIL_PATTERN, email):
            return False, f"Format d'email invalide: {email}"
            
        return True, None
    
    @staticmethod
    def normalize(email: str) -> str:
        """
        Normalise un email (minuscules, espaces supprimés).
        
        Args:
            email: L'email à normaliser
            
        Returns:
            str: L'email normalisé
        """
        return email.strip().lower()
    
    @staticmethod
    def explain_invalid_email(email: str) -> str:
        """
        Retourne une explication détaillée si l'email est invalide.
        
        Args:
            email: L'email à analyser
            
        Returns:
            str: Message d'erreur explicatif
        """
        if not email:
            return "L'email est vide"
            
        email = email.strip()
        
        if not '@' in email:
            return "L'email doit contenir un @"
            
        if not '.' in email.split('@')[1]:
            return "Le domaine doit contenir un point"
            
        if len(email.split('@')[0]) == 0:
            return "La partie locale de l'email est vide"
            
        if len(email.split('@')[1].split('.')[0]) == 0:
            return "Le domaine est vide"
            
        if len(email.split('@')[1].split('.')[-1]) < 2:
            return "L'extension du domaine doit faire au moins 2 caractères"
            
        return "Format d'email invalide" 