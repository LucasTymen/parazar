"""
Modèle Participant pour Parazar.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class Participant:
    """Représente un participant dans le système Parazar."""
    email: str
    prenom: Optional[str] = None
    nom: Optional[str] = None
    age: Optional[int] = None
    genre: Optional[str] = None
    ville: Optional[str] = None
    ville_non_listée: Optional[str] = None
    sujets_aimes: Optional[List[str]] = field(default_factory=list)
    type_personnalité: Optional[str] = None
    introverti_degré: Optional[int] = None
    frequence_sorties: Optional[str] = None
    aime_sport: Optional[bool] = None
    temps_libre: Optional[str] = None
    priorites_vie: Optional[List[str]] = field(default_factory=list)
    budget_sortie: Optional[str] = None
    activites_rencontre: Optional[List[str]] = field(default_factory=list)
    jours_disponibles: Optional[List[str]] = field(default_factory=list)
    experience_name: Optional[str] = None
    experience_date: Optional[str] = None
    experience_city: Optional[str] = None
    # Champs supplémentaires du rapport structuré
    topics: Optional[List[str]] = field(default_factory=list)
    type_personnalite: Optional[str] = None
    introversion: Optional[float] = None
    sport: Optional[float] = None
    lieu_preference: Optional[str] = None
    priorites: Optional[List[str]] = field(default_factory=list)
    budget_sorties: Optional[str] = None
    activites_recherchees: Optional[str] = None
    jours_dispo: Optional[List[str]] = field(default_factory=list)
    attentes: Optional[List[str]] = field(default_factory=list)
    signe_zodiaque: Optional[str] = None
    relation: Optional[str] = None
    secteur: Optional[str] = None
    birth_date: Optional[str] = None
    birth_year: Optional[int] = None
    age_bucket: Optional[str] = None
    amis: Optional[List[Dict[str, str]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Participant':
        """
        Crée une instance de Participant à partir d'un dictionnaire.
        """
        return cls(
            email=data.get('email', '').strip(),
            prenom=data.get('prenom'),
            nom=data.get('nom'),
            age=int(data.get('age', 0)) if data.get('age') else None,
            genre=data.get('genre'),
            ville=data.get('ville'),
            ville_non_listée=data.get('ville_non_listée'),
            sujets_aimes=data.get('sujets_aimes', []) or [],
            type_personnalité=data.get('type_personnalité'),
            introverti_degré=int(data.get('introverti_degré', 1)) if data.get('introverti_degré') else None,
            frequence_sorties=data.get('frequence_sorties'),
            aime_sport=data.get('aime_sport'),
            temps_libre=data.get('temps_libre'),
            priorites_vie=data.get('priorites_vie', []) or [],
            budget_sortie=data.get('budget_sortie'),
            activites_rencontre=data.get('activites_rencontre', []) or [],
            jours_disponibles=data.get('jours_disponibles', []) or [],
            experience_name=data.get('experience_name'),
            experience_date=data.get('experience_date'),
            experience_city=data.get('experience_city'),
            topics=data.get('topics') or [],
            type_personnalite=data.get('type_personnalite'),
            introversion=float(data['introversion']) if data.get('introversion') is not None else None,
            sport=float(data['sport']) if data.get('sport') is not None else None,
            lieu_preference=data.get('lieu_preference'),
            priorites=data.get('priorites') or [],
            budget_sorties=data.get('budget_sorties'),
            activites_recherchees=data.get('activites_recherchees'),
            jours_dispo=data.get('jours_dispo', []) or [],
            attentes=data.get('attentes', []) or [],
            signe_zodiaque=data.get('signe_zodiaque'),
            relation=data.get('relation'),
            secteur=data.get('secteur'),
            birth_date=data.get('birth_date'),
            birth_year=int(data.get('birth_year', 0)) if data.get('birth_year') else None,
            age_bucket=data.get('age_bucket'),
            amis=data.get('amis', []) or []
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'instance en dictionnaire.
        """
        return self.__dict__

    def __str__(self) -> str:
        return f"Participant({self.__dict__})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Participant):
            return False
        return self.__dict__ == other.__dict__ 