"""
Validateur d'âge pour Parazar.
"""
import logging
from typing import Tuple, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class AgeValidator:
    """Validateur d'âge avec support des tranches d'âge."""
    
    # Définition des tranches d'âge
    AGE_BUCKETS = {
        'underage': (0, 17),
        'young_adult': (18, 29),
        'adult': (30, 44),
        'middle_age': (45, 59),
        'senior': (60, 74),
        'elderly': (75, float('inf'))
    }
    
    # Noms des tranches d'âge pour l'affichage
    BUCKET_NAMES = {
        'underage': 'Moins de 18 ans',
        'young_adult': '18-29 ans',
        'adult': '30-44 ans',
        'middle_age': '45-59 ans',
        'senior': '60-74 ans',
        'elderly': '75 ans et plus'
    }
    
    @staticmethod
    def validate(age: int, birth_year: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Valide un âge et retourne un tuple (est_valide, message_erreur).
        
        Args:
            age: L'âge à valider
            birth_year: Année de naissance optionnelle pour vérification
            
        Returns:
            Tuple[bool, Optional[str]]: (True, None) si valide, (False, message) sinon
        """
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            return False, f"Âge invalide: {age}"
            
        if age_int < 0:
            return False, f"L'âge ne peut pas être négatif: {age}"
            
        # Vérification de cohérence avec l'année de naissance si fournie
        if birth_year is not None:
            try:
                birth_year_int = int(birth_year)
                current_year = datetime.now().year
                calculated_age = current_year - birth_year_int
                
                if abs(calculated_age - age_int) > 1:
                    logger.warning(
                        f"Incohérence âge/année de naissance: âge={age_int}, "
                        f"année={birth_year_int}, calculé={calculated_age}"
                    )
            except (ValueError, TypeError):
                logger.warning(f"Année de naissance invalide: {birth_year}")
                
        return True, None
    
    @staticmethod
    def get_age_bucket(age: int) -> str:
        """
        Détermine la tranche d'âge d'un participant.
        
        Args:
            age: L'âge du participant
            
        Returns:
            str: La clé de la tranche d'âge
        """
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            return 'underage'
            
        for bucket, (min_age, max_age) in AgeValidator.AGE_BUCKETS.items():
            if min_age <= age_int <= max_age:
                return bucket
                
        return 'elderly'  # Par défaut pour les âges très élevés
    
    @staticmethod
    def get_bucket_name(bucket: str) -> str:
        """
        Retourne le nom lisible d'une tranche d'âge.
        
        Args:
            bucket: La clé de la tranche d'âge
            
        Returns:
            str: Le nom lisible de la tranche
        """
        return AgeValidator.BUCKET_NAMES.get(bucket, 'Tranche d\'âge inconnue')
    
    @staticmethod
    def explain_invalid_age(age: int) -> str:
        """
        Retourne une explication détaillée si l'âge est invalide.
        
        Args:
            age: L'âge à analyser
            
        Returns:
            str: Message d'erreur explicatif
        """
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            return f"L'âge doit être un nombre entier: {age}"
            
        if age_int < 0:
            return f"L'âge ne peut pas être négatif: {age}"
            
        return "Format d'âge invalide" 