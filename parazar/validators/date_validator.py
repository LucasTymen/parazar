"""
Validateur de date pour Parazar.
"""
import logging
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DateValidator:
    """Validateur de date avec messages d'erreur explicites."""
    
    ACCEPTED_FORMATS = [
        '%Y-%m-%d',  # 2024-01-01
        '%d/%m/%Y',  # 01/01/2024
        '%d-%m-%Y',  # 01-01-2024
        '%Y/%m/%d'   # 2024/01/01
    ]
    
    @staticmethod
    def validate(date_str: str) -> Tuple[bool, Optional[str]]:
        """
        Valide une date et retourne un tuple (est_valide, message_erreur).
        
        Args:
            date_str: La date à valider
            
        Returns:
            Tuple[bool, Optional[str]]: (True, None) si valide, (False, message) sinon
        """
        if not date_str:
            return True, None  # Les dates vides sont acceptées
            
        date_str = date_str.strip()
        
        # Essayer chaque format accepté
        for date_format in DateValidator.ACCEPTED_FORMATS:
            try:
                datetime.strptime(date_str, date_format)
                return True, None
            except ValueError:
                continue
                
        return False, f"Format de date invalide: {date_str}. Formats acceptés: {', '.join(DateValidator.ACCEPTED_FORMATS)}"
    
    @staticmethod
    def normalize(date_str: str) -> str:
        """
        Normalise une date au format YYYY-MM-DD.
        
        Args:
            date_str: La date à normaliser
            
        Returns:
            str: La date normalisée ou la chaîne vide si invalide
        """
        if not date_str:
            return ""
            
        date_str = date_str.strip()
        
        # Essayer chaque format accepté
        for date_format in DateValidator.ACCEPTED_FORMATS:
            try:
                date_obj = datetime.strptime(date_str, date_format)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return date_str  # Retourner la chaîne originale si aucun format ne correspond
    
    @staticmethod
    def explain_invalid_date(date_str: str) -> str:
        """
        Retourne une explication détaillée si la date est invalide.
        
        Args:
            date_str: La date à analyser
            
        Returns:
            str: Message d'erreur explicatif
        """
        if not date_str:
            return "La date est vide"
            
        date_str = date_str.strip()
        
        # Vérifier si la chaîne contient des caractères non numériques
        if not any(c.isdigit() for c in date_str):
            return "La date doit contenir des chiffres"
            
        # Vérifier le format général
        if not any(sep in date_str for sep in ['-', '/']):
            return "La date doit contenir des séparateurs (- ou /)"
            
        # Vérifier la longueur
        if len(date_str) < 8:
            return "La date est trop courte"
            
        if len(date_str) > 10:
            return "La date est trop longue"
            
        return f"Format de date invalide: {date_str}. Formats acceptés: {', '.join(DateValidator.ACCEPTED_FORMATS)}" 