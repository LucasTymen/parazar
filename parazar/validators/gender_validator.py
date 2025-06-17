"""
Validateur de genre pour Parazar.
"""
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class GenderValidator:
    """Validateur de genre avec support des genres non-binaires."""
    
    # Mapping des genres acceptés
    GENDER_MAPPING: Dict[str, str] = {
        # Masculin
        'm': 'M',
        'male': 'M',
        'homme': 'M',
        'masculin': 'M',
        'h': 'M',
        
        # Féminin
        'f': 'F',
        'female': 'F',
        'femme': 'F',
        'féminin': 'F',
        
        # Non-binaire
        'nb': 'NB',
        'non-binaire': 'NB',
        'non binaire': 'NB',
        'nonbinaire': 'NB',
        'autre': 'NB',
        'x': 'NB',
        'other': 'NB'
    }
    
    VALID_GENDERS = {'M', 'F', 'NB'}
    
    @staticmethod
    def validate(gender: str) -> Tuple[bool, Optional[str]]:
        """
        Valide un genre et retourne un tuple (est_valide, message_erreur).
        
        Args:
            gender: Le genre à valider
            
        Returns:
            Tuple[bool, Optional[str]]: (True, None) si valide, (False, message) sinon
        """
        if not gender:
            return False, "Le genre ne peut pas être vide"
            
        normalized = GenderValidator.normalize(gender)
        if normalized not in GenderValidator.VALID_GENDERS:
            return False, f"Genre invalide: {gender}. Valeurs acceptées: {', '.join(GenderValidator.VALID_GENDERS)}"
            
        return True, None
    
    @staticmethod
    def normalize(gender: str) -> str:
        """
        Normalise un genre selon le mapping défini.
        
        Args:
            gender: Le genre à normaliser
            
        Returns:
            str: Le genre normalisé ou 'NB' si non reconnu
        """
        if not gender:
            return 'NB'
            
        normalized = gender.strip().lower()
        return GenderValidator.GENDER_MAPPING.get(normalized, 'NB')
    
    @staticmethod
    def explain_invalid_gender(gender: str) -> str:
        """
        Retourne une explication détaillée si le genre est invalide.
        
        Args:
            gender: Le genre à analyser
            
        Returns:
            str: Message d'erreur explicatif
        """
        if not gender:
            return "Le genre est vide"
            
        normalized = gender.strip().lower()
        if normalized not in GenderValidator.GENDER_MAPPING:
            return f"Genre non reconnu: {gender}. Valeurs acceptées: {', '.join(GenderValidator.VALID_GENDERS)}"
            
        return "Format de genre invalide" 