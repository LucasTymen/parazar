"""
Système de Matching Parazar - Algorithme de Composition de Groupes Sociaux
Optimisé avec lambdas, streams et gestion complète des cas d'erreur
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
from itertools import combinations, chain
from functools import reduce, partial
from collections import defaultdict, Counter
import logging
from enum import Enum
import json
import re
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_CONSTRAINTS = "failed_constraints"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"

# Regex pour validation email
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

@dataclass
class Participant:
    """Modèle participant Parazar basé sur les vrais champs Tally"""
    # Champs obligatoires Tally
    email: str  # Identifiant unique
    first_name: str
    gender: str  # 'M', 'F'
    age: int
    
    # Champs Tally optionnels
    parazar_partner_id: str = ""
    reservation: str = ""
    note: str = ""
    group: str = ""
    telephone: str = ""
    transaction_date: str = ""
    experience_name: str = ""
    experience_date: str = ""
    experience_date_formatted: str = ""
    experience_hour: str = ""
    experience_city: str = ""
    meeting_id_list: str = ""
    meeting_id_count: int = 0
    experience_bought_count: int = 0
    reduction_code: str = ""
    job_field: str = ""
    topics_conversations: List[str] = field(default_factory=list)
    astrological_sign: str = ""
    relationship_status: str = ""
    life_priorities: List[str] = field(default_factory=list)
    introverted_degree: float = 0.5  # 0-1 selon Tally
    
    # Scores calculés
    social_score: float = field(default=0.0, init=False)
    compatibility_topics: Set[str] = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Calculs post-initialisation"""
        self._validate()
        self.social_score = self._calculate_social_score()
        self.compatibility_topics = set(self.topics_conversations)
    
    def _validate(self):
        """Validation des champs obligatoires"""
        if not self.email or not EMAIL_REGEX.match(self.email):
            raise ValueError("Email invalide")
        if not self.first_name or not isinstance(self.first_name, str):
            raise ValueError("Prénom invalide")
        if not isinstance(self.age, (int, float)) or self.age < 18 or self.age > 65:
            raise ValueError(f"Âge invalide: {self.age}")
        if self.gender not in ['M', 'F']:
            raise ValueError(f"Genre invalide: {self.gender}")
    
    def _calculate_social_score(self) -> float:
        """Score social basé sur introversion (0-1 scale Tally)"""
        # Conversion: 0 = très introverti, 1 = très extraverti
        # Score social de 0 à 10
        base_score = (1 - self.introverted_degree) * 10
        return round(base_score, 2)

@dataclass
class Group:
    """Groupe optimisé avec contraintes Parazar"""
    id: str
    participants: List[Participant] = field(default_factory=list)
    experience_name: str = ""
    experience_date: str = ""
    experience_city: str = ""
    
    # Métriques calculées
    compatibility_score: float = field(default=0.0, init=False)
    age_spread: float = field(default=0.0, init=False)
    gender_balance: Dict[str, int] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self._update_metrics()
    
    def _update_metrics(self):
        """Mise à jour des métriques du groupe"""
        if not self.participants:
            return
        
        ages = [p.age for p in self.participants]
        self.age_spread = max(ages) - min(ages)
        self.gender_balance = Counter(p.gender for p in self.participants)
        self.compatibility_score = self._calculate_compatibility()
    
    def _calculate_compatibility(self) -> float:
        """Score de compatibilité basé sur topics et social scores"""
        if len(self.participants) < 2:
            return 0.0
        
        # Compatibilité des topics
        all_topics = [p.compatibility_topics for p in self.participants]
        topic_overlap = len(set.intersection(*all_topics)) if all_topics else 0
        
        # Équilibre des scores sociaux
        social_scores = [p.social_score for p in self.participants]
        social_balance = 1 - (np.std(social_scores) / 10) if len(social_scores) > 1 else 1
        
        return round((topic_overlap * 2 + social_balance * 3), 2)
    
    @property
    def is_valid(self) -> bool:
        """Validation des contraintes Parazar"""
        return (
            len(self.participants) >= 4 and
            len(self.participants) <= 8 and
            self.gender_balance.get('F', 0) >= 2 and
            self.age_spread <= 6
        )
    
    @property
    def needs_female(self) -> bool:
        """Le groupe a-t-il besoin de plus de femmes ?"""
        return self.gender_balance.get('F', 0) < 2
    
    @property 
    def female_age_constraint_ok(self) -> bool:
        """Les femmes ne sont-elles pas les plus âgées ?"""
        if not self.participants:
            return True
        females = [p for p in self.participants if p.gender == 'F']
        if not females:
            return True
        max_female_age = max(f.age for f in females)
        max_total_age = max(p.age for p in self.participants)
        return max_female_age <= max_total_age

class ParazarMatcher:
    """Moteur de matching Parazar avec gestion d'erreurs avancée"""
    
    def __init__(self, min_group_size: int = 4, max_group_size: int = 8, 
                 max_age_spread: int = 6, min_females_per_group: int = 2):
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_age_spread = max_age_spread
        self.min_females_per_group = min_females_per_group
        self.groups: List[Group] = []
        self.unmatched: List[Participant] = []
        
    def load_from_dataframe(self, df: pd.DataFrame) -> List[Participant]:
        """Chargement et validation des données depuis DataFrame"""
        try:
            # Nettoyage et validation des données avec champs Tally
            required_fields = ['email', 'first_name', 'age', 'gender']
            df_clean = df.dropna(subset=required_fields)
            
            participants = []
            for _, row in df_clean.iterrows():
                try:
                    participant = self._create_participant_from_row(row)
                    if participant:
                        participants.append(participant)
                except Exception as e:
                    logger.warning(f"Erreur création participant: {e}")
                    continue
            
            # Filtrage des participants valides
            valid_participants = [
                p for p in participants 
                if p.email and p.first_name and p.age >= 18 and p.gender in ['M', 'F']
            ]
            
            logger.info(f"Chargé {len(valid_participants)} participants valides sur {len(df)} lignes")
            return valid_participants
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return []
    
    def _create_participant_from_row(self, row: Dict) -> Optional[Participant]:
        """Création d'un participant depuis une ligne Tally/CSV"""
        try:
            # Validation email avec regex stricte
            email = str(row.get('email', '')).strip().lower()
            if not email or not EMAIL_REGEX.match(email):
                raise ValueError(f"Email invalide: {email}")
            
            # Validation prénom
            first_name = str(row.get('first_name', '')).strip()
            if not first_name or first_name.lower() in ['none', 'nan']:
                raise ValueError("Prénom invalide")
            
            # Validation âge
            try:
                age = int(float(row.get('age', 0)))
                if age < 18 or age > 65:
                    raise ValueError(f"Âge invalide: {age}")
            except (ValueError, TypeError):
                raise ValueError(f"Âge invalide: {row.get('age')}")
            
            # Normalisation du genre avec plus de variations
            gender_raw = str(row.get('gender', '')).strip().upper()
            if gender_raw in ['M', 'HOMME', 'MASCULIN', 'MALE', 'H']:
                gender = 'M'
            elif gender_raw in ['F', 'FEMME', 'FÉMININ', 'FEMININ', 'FEMALE']:
                gender = 'F'
            else:
                raise ValueError(f"Genre invalide: {gender_raw}")
            
            # Nettoyage topics_conversations
            topics_raw = row.get('topics_conversations', '')
            topics = [t.strip() for t in topics_raw.split(',')] if topics_raw else []
            
            # Nettoyage life_priorities
            priorities_raw = row.get('life_priorities', '')
            priorities = [p.strip() for p in priorities_raw.split(',')] if priorities_raw else []
            
            # Validation introverted_degree
            try:
                introverted_degree = float(row.get('introverted_degree', 0.5))
                if not 0 <= introverted_degree <= 1:
                    introverted_degree = 0.5
            except (ValueError, TypeError):
                introverted_degree = 0.5
            
            return Participant(
                email=email,
                first_name=first_name,
                gender=gender,
                age=age,
                parazar_partner_id=str(row.get('parazar_partner_id', '')),
                reservation=str(row.get('reservation', '')),
                note=str(row.get('note', '')),
                group=str(row.get('group', '')),
                telephone=str(row.get('telephone', '')),
                transaction_date=str(row.get('transaction_date', '')),
                experience_name=str(row.get('experience_name', '')),
                experience_date=str(row.get('experience_date', '')),
                experience_date_formatted=str(row.get('experience_date_formatted', '')),
                experience_hour=str(row.get('experience_hour', '')),
                experience_city=str(row.get('experience_city', '')),
                meeting_id_list=str(row.get('meeting_id_list', '')),
                meeting_id_count=int(row.get('meeting_id_count', 0)),
                experience_bought_count=int(row.get('experience_bought_count', 0)),
                reduction_code=str(row.get('reduction_code', '')),
                job_field=str(row.get('job_field', '')),
                topics_conversations=topics,
                astrological_sign=str(row.get('astrological_sign', '')),
                relationship_status=str(row.get('relationship_status', '')),
                life_priorities=priorities,
                introverted_degree=introverted_degree
            )
        except Exception as e:
            logger.warning(f"Erreur création participant: {e}")
            return None
    
    def create_optimal_groups(self, participants: List[Participant]) -> Tuple[str, Dict]:
        """Création de groupes optimaux avec gestion complète des erreurs"""
        try:
            if not participants:
                return "success", {
                    "groups": [],
                    "unmatched": [],
                    "stats": {
                        "total_participants": 0,
                        "groups_created": 0,
                        "participants_matched": 0,
                        "participants_unmatched": 0,
                        "matching_rate": 100,
                        "avg_group_size": 0,
                        "avg_compatibility_score": 0,
                        "valid_groups": 0
                    }
                }
            
            if len(participants) < self.min_group_size:
                return "insufficient_data", {
                    "error": "Pas assez de participants",
                    "required": self.min_group_size,
                    "available": len(participants)
                }
            
            # Segmentation par expérience/date/ville
            segments = self._segment_participants(participants)
            
            results = {
                "groups": [],
                "unmatched": [],
                "stats": {},
                "segments_processed": len(segments)
            }
            
            for segment_key, segment_participants in segments.items():
                segment_result = self._process_segment(segment_participants, segment_key)
                results["groups"].extend(segment_result["groups"])
                results["unmatched"].extend(segment_result["unmatched"])
            
            # Statistiques globales
            results["stats"] = self._calculate_global_stats(results["groups"], results["unmatched"])
            
            # Détermination du statut
            status = self._determine_status(results["stats"])
            
            return status, results
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des groupes: {e}")
            return "error", {"error": str(e)}
    
    def _segment_participants(self, participants: List[Participant]) -> Dict[str, List[Participant]]:
        """Segmentation des participants par expérience/date/ville"""
        segments = defaultdict(list)
        
        for participant in participants:
            key = f"{participant.experience_name}_{participant.experience_date}_{participant.experience_city}"
            segments[key].append(participant)
        
        return dict(segments)
    
    def _process_segment(self, participants: List[Participant], segment_key: str) -> Dict:
        """Traitement d'un segment de participants"""
        logger.info(f"Traitement segment {segment_key}: {len(participants)} participants")
        
        # Tri par priorité de matching (femmes d'abord, puis introvertis)
        sorted_participants = sorted(
            participants,
            key=lambda p: (
                0 if p.gender == 'F' else 1,  # Femmes en priorité
                p.introverted_degree,         # Introvertis en priorité
                -p.social_score               # Score social décroissant
            )
        )
        
        groups = []
        unmatched = []
        remaining = sorted_participants.copy()
        
        while len(remaining) >= self.min_group_size:
            group_result = self._create_single_group(remaining, segment_key)
            
            if group_result["success"]:
                groups.append(group_result["group"])
                # Retirer les participants assignés
                remaining = [p for p in remaining if p not in group_result["group"].participants]
            else:
                # Si impossible de créer un groupe, essayer de fusionner les restants
                if len(remaining) >= self.min_group_size:
                    fusion_result = self._attempt_group_fusion(remaining, segment_key)
                    if fusion_result["success"]:
                        groups.append(fusion_result["group"])
                        remaining = []
                    else:
                        unmatched.extend(remaining)
                        break
                else:
                    unmatched.extend(remaining)
                    break
        
        unmatched.extend(remaining)
        
        return {
            "groups": groups,
            "unmatched": unmatched,
            "segment": segment_key
        }
    
    def _create_single_group(self, available_participants: List[Participant], segment_key: str) -> Dict:
        """Création d'un groupe unique optimisé"""
        try:
            # Stratégie: commencer par les femmes pour garantir la contrainte
            females = [p for p in available_participants if p.gender == 'F']
            males = [p for p in available_participants if p.gender == 'M']
            
            if len(females) < self.min_females_per_group:
                return {"success": False, "reason": "Pas assez de femmes disponibles"}
            
            # Sélection optimale des femmes (éviter qu'elles soient les plus âgées)
            selected_females = self._select_optimal_females(females, males)
            if len(selected_females) < self.min_females_per_group:
                return {"success": False, "reason": "Impossible de respecter contrainte d'âge femmes"}
            
            # Sélection des hommes compatibles
            selected_males = self._select_compatible_males(selected_females, males)
            
            all_selected = selected_females + selected_males
            
            if len(all_selected) < self.min_group_size:
                return {"success": False, "reason": "Groupe trop petit après sélection"}
            
            # Limitation à la taille max
            if len(all_selected) > self.max_group_size:
                all_selected = self._optimize_group_size(all_selected)
            
            group = Group(
                id=f"group_{segment_key}_{len(self.groups) + 1}",
                participants=all_selected,
                experience_name=all_selected[0].experience_name,
                experience_date=all_selected[0].experience_date,
                experience_city=all_selected[0].experience_city
            )
            
            if group.is_valid and group.female_age_constraint_ok:
                return {"success": True, "group": group}
            else:
                return {"success": False, "reason": "Contraintes non respectées"}
                
        except Exception as e:
            logger.warning(f"Erreur création groupe: {e}")
            return {"success": False, "reason": str(e)}
    
    def _select_optimal_females(self, females: List[Participant], males: List[Participant]) -> List[Participant]:
        """Sélection optimale des femmes (contrainte d'âge)"""
        if not males:
            return females[:self.min_females_per_group]
        
        max_male_age = max(m.age for m in males)
        
        # Filtrer les femmes qui ne seraient pas les plus âgées
        suitable_females = [f for f in females if f.age <= max_male_age]
        
        if len(suitable_females) >= self.min_females_per_group:
            # Trier par compatibilité/score social
            return sorted(suitable_females, key=lambda f: -f.social_score)[:4]  # Max 4 femmes
        else:
            # Fallback: prendre les plus jeunes
            return sorted(females, key=lambda f: f.age)[:self.min_females_per_group]
    
    def _select_compatible_males(self, selected_females: List[Participant], available_males: List[Participant]) -> List[Participant]:
        """Sélection des hommes compatibles avec les femmes sélectionnées"""
        if not selected_females or not available_males:
            return []
        
        female_ages = [f.age for f in selected_females]
        female_topics = set(chain.from_iterable(f.topics_conversations for f in selected_females))
        
        # Score de compatibilité pour chaque homme
        def compatibility_score(male: Participant) -> float:
            age_score = max(0, 6 - abs(male.age - np.mean(female_ages))) / 6
            topic_score = len(set(male.topics_conversations) & female_topics) / max(len(female_topics), 1)
            social_score = male.social_score / 10
            return age_score * 0.4 + topic_score * 0.3 + social_score * 0.3
        
        scored_males = [(m, compatibility_score(m)) for m in available_males]
        sorted_males = sorted(scored_males, key=lambda x: -x[1])
        
        # Sélectionner 2-4 hommes selon la taille souhaitée
        target_males = min(4, self.max_group_size - len(selected_females))
        return [m[0] for m in sorted_males[:target_males]]
    
    def _optimize_group_size(self, participants: List[Participant]) -> List[Participant]:
        """Optimisation de la taille du groupe (garder les plus compatibles)"""
        if len(participants) <= self.max_group_size:
            return participants
        
        # Garder le ratio femmes/hommes
        females = [p for p in participants if p.gender == 'F']
        males = [p for p in participants if p.gender == 'M']
        
        # Prioriser les scores sociaux et la diversité des topics
        def selection_score(p: Participant) -> float:
            return p.social_score + len(p.topics_conversations) * 0.5
        
        females_sorted = sorted(females, key=selection_score, reverse=True)
        males_sorted = sorted(males, key=selection_score, reverse=True)
        
        # Équilibrer le groupe
        target_females = min(len(females_sorted), 4)
        target_males = min(len(males_sorted), self.max_group_size - target_females)
        
        return females_sorted[:target_females] + males_sorted[:target_males]
    
    def _attempt_group_fusion(self, remaining_participants: List[Participant], segment_key: str) -> Dict:
        """Tentative de fusion des participants restants en un groupe"""
        if len(remaining_participants) < self.min_group_size:
            return {"success": False, "reason": "Pas assez de participants pour fusion"}
        
        # Même logique que création de groupe simple
        return self._create_single_group(remaining_participants, f"{segment_key}_fusion")
    
    def find_replacement(self, group: Group, leaving_participant: Participant, 
                        available_pool: List[Participant]) -> Optional[Participant]:
        """Trouve un remplaçant optimal pour un participant qui se désiste"""
        try:
            if not available_pool:
                return None
            
            # Contraintes du groupe après départ
            remaining = [p for p in group.participants if p != leaving_participant]
            
            # Filtres de base
            candidates = list(filter(
                lambda p: (
                    p.experience_name == group.experience_name and
                    p.experience_date == group.experience_date and
                    p.experience_city == group.experience_city
                ),
                available_pool
            ))
            
            if not candidates:
                return None
            
            # Score de remplacement
            def replacement_score(candidate: Participant) -> float:
                test_group = Group(
                    id="test",
                    participants=remaining + [candidate],
                    experience_name=group.experience_name,
                    experience_date=group.experience_date,
                    experience_city=group.experience_city
                )
                
                if not test_group.is_valid or not test_group.female_age_constraint_ok:
                    return -1  # Invalid
                
                return test_group.compatibility_score
            
            # Trouver le meilleur candidat
            scored_candidates = [(c, replacement_score(c)) for c in candidates]
            valid_candidates = [(c, s) for c, s in scored_candidates if s >= 0]
            
            if not valid_candidates:
                return None
            
            best_candidate = max(valid_candidates, key=lambda x: x[1])
            return best_candidate[0]
            
        except Exception as e:
            logger.error(f"Erreur recherche remplaçant: {e}")
            return None
    
    def _calculate_global_stats(self, groups: List[Group], unmatched: List[Participant]) -> Dict:
        """Calcul des statistiques globales"""
        total_participants = sum(len(g.participants) for g in groups) + len(unmatched)
        
        return {
            "total_participants": total_participants,
            "groups_created": len(groups),
            "participants_matched": sum(len(g.participants) for g in groups),
            "participants_unmatched": len(unmatched),
            "matching_rate": round(sum(len(g.participants) for g in groups) / total_participants * 100, 2) if total_participants > 0 else 100,
            "avg_group_size": round(np.mean([len(g.participants) for g in groups]), 2) if groups else 0,
            "avg_compatibility_score": round(np.mean([g.compatibility_score for g in groups]), 2) if groups else 0,
            "valid_groups": sum(1 for g in groups if g.is_valid)
        }
    
    def _determine_status(self, stats: Dict) -> str:
        """Détermine le statut global du matching"""
        if stats["matching_rate"] >= 90:
            return "success"
        elif stats["matching_rate"] >= 70:
            return "partial_success"
        elif stats["valid_groups"] == 0:
            return "failed_constraints"
        else:
            return "partial_success"

# Fonctions utilitaires pour l'intégration
def load_parazar_data(csv_path: str) -> pd.DataFrame:
    """Chargement des données Parazar depuis CSV"""
    df = pd.read_csv(csv_path)
    logger.info(f"Données chargées: {len(df)} lignes")
    return df

def export_groups_to_json(groups: List[Group], output_path: str):
    """Export des groupes vers JSON pour n8n"""
    try:
        groups_data = []
        for group in groups:
            group_dict = {
                "id": group.id,
                "experience_name": group.experience_name,
                "experience_date": group.experience_date,
                "experience_city": group.experience_city,
                "compatibility_score": group.compatibility_score,
                "participants": [
                    {
                        "email": p.email,  # Identifiant unique Tally
                        "parazar_partner_id": p.parazar_partner_id,
                        "first_name": p.first_name,
                        "telephone": p.telephone,
                        "age": p.age,
                        "gender": p.gender,
                        "job_field": p.job_field,
                        "topics_conversations": p.topics_conversations,
                        "relationship_status": p.relationship_status,
                        "introverted_degree": p.introverted_degree
                    }
                    for p in group.participants
                ]
            }
            groups_data.append(group_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Groupes exportés vers {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur export JSON: {e}")
        raise  # Propager l'erreur pour la gestion en amont
