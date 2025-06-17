"""
Validateur principal pour les participants Parazar.
"""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .email_validator import EmailValidator
from .gender_validator import GenderValidator
from .age_validator import AgeValidator
from .date_validator import DateValidator

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Résultat de la validation d'un participant."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_data: Dict[str, str]

class ParticipantValidator:
    """Validateur principal qui coordonne la validation des participants."""
    
    def __init__(self):
        """Initialise les validateurs."""
        self.email_validator = EmailValidator()
        self.gender_validator = GenderValidator()
        self.age_validator = AgeValidator()
        self.date_validator = DateValidator()
    
    def validate(self, data: Dict[str, str]) -> ValidationResult:
        """
        Valide les données d'un participant.
        
        Args:
            data: Dictionnaire contenant les données du participant
            
        Returns:
            ValidationResult: Résultat de la validation
        """
        errors = []
        warnings = []
        normalized_data = {}
        
        # Validation de l'email
        email = data.get('email', '').strip()
        is_valid_email, email_error = self.email_validator.validate(email)
        if not is_valid_email:
            errors.append(f"Email invalide: {email_error}")
        else:
            normalized_data['email'] = self.email_validator.normalize(email)
        
        # Validation du genre
        gender = data.get('gender', '').strip()
        is_valid_gender, gender_error = self.gender_validator.validate(gender)
        if not is_valid_gender:
            errors.append(f"Genre invalide: {gender_error}")
        else:
            normalized_data['gender'] = self.gender_validator.normalize(gender)
        
        # Validation de l'âge
        age = data.get('age', '').strip()
        birth_year = data.get('birth_year', '').strip()
        is_valid_age, age_error = self.age_validator.validate(age, birth_year)
        if not is_valid_age:
            errors.append(f"Âge invalide: {age_error}")
        else:
            normalized_data['age'] = age
            normalized_data['age_bucket'] = self.age_validator.get_age_bucket(int(age))
        
        # Validation de la date de naissance
        birth_date = data.get('birth_date', '').strip()
        is_valid_date, date_error = self.date_validator.validate(birth_date)
        if not is_valid_date:
            warnings.append(f"Date de naissance invalide: {date_error}")
        else:
            normalized_data['birth_date'] = self.date_validator.normalize(birth_date)
        
        # Vérification de la cohérence entre l'âge et la date de naissance
        if is_valid_age and is_valid_date:
            try:
                age_from_date = self._calculate_age_from_date(normalized_data['birth_date'])
                if abs(age_from_date - int(age)) > 1:
                    warnings.append(
                        f"Incohérence entre l'âge ({age}) et la date de naissance "
                        f"({normalized_data['birth_date']})"
                    )
            except ValueError:
                warnings.append("Impossible de vérifier la cohérence âge/date de naissance")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_data=normalized_data
        )
    
    def _calculate_age_from_date(self, date_str: str) -> int:
        """
        Calcule l'âge à partir d'une date de naissance.
        
        Args:
            date_str: Date de naissance au format YYYY-MM-DD
            
        Returns:
            int: Âge calculé
        """
        from datetime import datetime, date
        birth_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        today = date.today()
        age = today.year - birth_date.year
        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1
        return age 