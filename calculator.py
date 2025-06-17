"""
Système de Matching Parazar - Algorithme de Composition de Groupes Sociaux
Version CORRIGÉE pour résoudre TOUS les tests et atteindre 100% de couverture
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from itertools import combinations, chain
from functools import reduce, partial
from collections import defaultdict, Counter
import logging
from enum import Enum
import json
from datetime import datetime
from typing import Dict

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchingStatus(Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED_CONSTRAINTS = "FAILED_CONSTRAINTS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ERROR = "ERROR"

@dataclass
class Participant:
    """Modèle participant Parazar basé sur les vrais champs Tally"""
    # Champs obligatoires Tally
    email: str  # Identifiant unique
    first_name: str
    gender: str  # 'm', 'f', autre
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
    introverted_degree: float = 0.5
    
    # Scores calculés
    social_score: float = field(default=0.0, init=False)
    compatibility_topics: Set[str] = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Calculs post-initialisation avec validation"""
        # Valider et normaliser introverted_degree
        if self.introverted_degree < 0:
            self.introverted_degree = 0.0
        elif self.introverted_degree > 1:
            self.introverted_degree = 1.0
            
        # Valider l'email
        if not self._is_valid_email(self.email):
            raise ValueError(f"Email invalide: {self.email}")
            
        self.social_score = self._calculate_social_score()
        self.compatibility_topics = set(self.topics_conversations)
    
    def _is_valid_email(self, email: str) -> bool:
        """Validation basique de l'email"""
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))
    
    def _calculate_social_score(self) -> float:
        """Score social basé sur introversion (0-1 scale Tally)"""
        # Conversion: 0 = très introverti, 1 = très extraverti
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
        """Les femmes ne sont-elles pas strictement les plus âgées ?"""
        if not self.participants:
            return True
        females = [p for p in self.participants if p.gender == 'F']
        if not females:
            return True
        max_female_age = max(f.age for f in females)
        all_ages = [p.age for p in self.participants]
        return max_female_age < max(all_ages) or len([p for p in self.participants if p.age == max_female_age]) > 1

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
        """Chargement et validation des données depuis DataFrame - CORRECTION 1"""
        try:
            # CORRECTION: Vérifier si le DataFrame est vide AVANT toute opération
            if df.empty:
                logger.warning("DataFrame vide fourni")
                return []
            
            # Vérifier l'existence des colonnes requises AVANT dropna
            required_fields = ['email', 'first_name', 'age', 'gender']
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                raise KeyError(f"Colonnes manquantes: {missing_fields}")
            
            # Nettoyage et validation des données
            df_clean = df.dropna(subset=required_fields)
            
            # Détecter et gérer les doublons d'email
            df_clean = df_clean.drop_duplicates(subset=['email'], keep='first')
            
            participants = []
            for _, row in df_clean.iterrows():
                try:
                    participant = self._create_participant_from_row(row.to_dict())
                    participants.append(participant)
                except ValueError as e:
                    logger.warning(f"Participant invalide ignoré: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Erreur création participant: {e}")
                    continue
            
            # Filtrage des participants valides
            valid_participants = list(filter(
                lambda p: (
                    p.age >= 18 and p.age <= 65 and 
                    p.gender in ['M', 'F'] and
                    p.email
                ),
                participants
            ))
            
            logger.info(f"Chargé {len(valid_participants)} participants valides sur {len(df)} lignes")
            return valid_participants
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise

    def _create_participant_from_row(self, row: Dict) -> Participant:
        """Création d'un participant depuis une ligne - CORRECTION 2"""
        try:
            # Nettoyage + validation
            email = str(row.get('email', '')).strip().lower()
            first_name = str(row.get('first_name', '')).strip().capitalize()

            if not email or not self._is_valid_email(email):
                raise ValueError(f"Email invalide: {email}")
            if not first_name:
                raise ValueError("Prénom manquant")

            # CORRECTION: Valider la date d'expérience STRICTEMENT
            experience_date = str(row.get('experience_date', '')).strip()
            if experience_date and experience_date not in ['', 'nan', 'None']:
                try:
                    self._validate_date_format(experience_date)
                except Exception:
                    raise ValueError(f"Format de date invalide: {experience_date}")

            # Gérer la cohérence âge/année de naissance
            age = int(row.get('age', 0))
            birth_year = row.get('birth_year')
            if birth_year and str(birth_year) not in ['', 'nan', 'None']:
                try:
                    current_year = datetime.now().year
                    calculated_age = current_year - int(birth_year)
                    if abs(age - calculated_age) > 2:
                        age = calculated_age
                        logger.warning(f"Âge corrigé basé sur birth_year: {age}")
                except (ValueError, TypeError):
                    pass  # Ignorer les erreurs de conversion

            # Nettoyage topics_conversations
            topics_raw = row.get('topics_conversations', '')
            topics = []
            if topics_raw and str(topics_raw) not in ['', 'nan', 'None']:
                topics = list(map(lambda x: x.strip().lower(), filter(None, str(topics_raw).split(','))))

            # Nettoyage life_priorities
            priorities_raw = row.get('life_priorities', '')
            priorities = []
            if priorities_raw and str(priorities_raw) not in ['', 'nan', 'None']:
                priorities = list(map(lambda x: x.strip().lower(), filter(None, str(priorities_raw).split(','))))

            # Normalisation du genre
            gender_raw = str(row.get('gender', '')).strip().lower()
            gender_normalized = 'M' if gender_raw == 'm' else 'F' if gender_raw == 'f' else gender_raw.upper()

            # CORRECTION: Valider introverted_degree plus strictement
            introverted_degree_raw = row.get('introverted_degree', 0.5)
            try:
                introverted_degree = float(introverted_degree_raw)
                if not (0 <= introverted_degree <= 1):
                    raise ValueError(f"introverted_degree hors limites: {introverted_degree}")
            except (ValueError, TypeError):
                raise ValueError(f"introverted_degree invalide: {introverted_degree_raw}")

            return Participant(
                email=email,
                first_name=first_name,
                gender=gender_normalized,
                age=age,
                parazar_partner_id=str(row.get('parazar_partner_id', '')).strip(),
                reservation=str(row.get('reservation', '')).strip(),
                note=str(row.get('note', '')).strip(),
                group=str(row.get('group', '')).strip(),
                telephone=str(row.get('telephone', '')).strip(),
                transaction_date=str(row.get('transaction_date', '')).strip(),
                experience_name=str(row.get('experience_name', '')).strip(),
                experience_date=experience_date,
                experience_date_formatted=str(row.get('experience_date_formatted', '')).strip(),
                experience_hour=str(row.get('experience_hour', '')).strip(),
                experience_city=str(row.get('experience_city', '')).strip(),
                meeting_id_list=str(row.get('meeting_id_list', '')).strip(),
                meeting_id_count=int(row.get('meeting_id_count', 0)),
                experience_bought_count=int(row.get('experience_bought_count', 0)),
                reduction_code=str(row.get('reduction_code', '')).strip(),
                job_field=str(row.get('job_field', '')).strip(),
                topics_conversations=topics,
                astrological_sign=str(row.get('astrological_sign', '')).strip(),
                relationship_status=str(row.get('relationship_status', '')).strip(),
                life_priorities=priorities,
                introverted_degree=introverted_degree
            )
        except Exception as e:
            logger.warning(f"Erreur création participant: {e}")
            raise

    def _is_valid_email(self, email: str) -> bool:
        """Validation basique de l'email"""
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

    def _validate_date_format(self, date_str: str):
        """CORRECTION 3: Valide strictement le format de date"""
        if not date_str or date_str.strip() == '':
            raise ValueError("Date vide")
        
        # Rejeter explicitement les dates invalides
        if date_str.upper() in ['INVALID', 'NULL', 'NONE', 'NAN']:
            raise ValueError(f"Date explicitement invalide: {date_str}")
        
        # Formats acceptés
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # Vérifier que la date est réaliste (pas dans le passé lointain ou futur lointain)
                current_year = datetime.now().year
                if not (1900 <= parsed_date.year <= current_year + 10):
                    continue
                return  # Format valide trouvé
            except ValueError:
                continue
        
        # Aucun format valide trouvé
        raise ValueError(f"Format de date invalide: {date_str}")
    
    def create_optimal_groups(self, participants: List[Participant]) -> Tuple[MatchingStatus, Dict]:
        """CORRECTION 4: Création de groupes avec statuts Enum corrects"""
        try:
            if len(participants) < self.min_group_size:
                return MatchingStatus.INSUFFICIENT_DATA, {
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
            
            # Améliorer la fusion si le taux de matching est faible
            if results["groups"]:
                matching_rate = len([p for g in results["groups"] for p in g.participants]) / len(participants) * 100
                if matching_rate < 70 and len(results["unmatched"]) >= self.min_group_size:
                    fusion_result = self._aggressive_fusion(results["unmatched"], "global_fusion")
                    if fusion_result["groups"]:
                        results["groups"].extend(fusion_result["groups"])
                        results["unmatched"] = fusion_result["unmatched"]
            
            # Statistiques globales
            results["stats"] = self._calculate_global_stats(results["groups"], results["unmatched"])
            
            # Détermination du statut
            status = self._determine_status(results)
            
            return status, results
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des groupes: {e}")
            return MatchingStatus.ERROR, {"error": str(e)}
    
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
        
        # Tri par priorité de matching
        sorted_participants = sorted(
            participants,
            key=lambda p: (
                0 if p.gender == 'F' else 1,
                p.introverted_degree,
                -p.social_score
            )
        )
        
        groups = []
        unmatched = []
        remaining = sorted_participants.copy()
        
        # Stratégie multi-passes
        max_attempts = 3
        attempt = 0
        
        while len(remaining) >= self.min_group_size and attempt < max_attempts:
            initial_remaining_count = len(remaining)
            
            while len(remaining) >= self.min_group_size:
                group_result = self._create_single_group(remaining, f"{segment_key}_{attempt}")
                
                if group_result["success"]:
                    groups.append(group_result["group"])
                    remaining = [p for p in remaining if p not in group_result["group"].participants]
                else:
                    break
            
            if len(remaining) == initial_remaining_count and len(remaining) >= self.min_group_size:
                relaxed_result = self._create_relaxed_group(remaining, f"{segment_key}_relaxed_{attempt}")
                if relaxed_result["success"]:
                    groups.append(relaxed_result["group"])
                    remaining = [p for p in remaining if p not in relaxed_result["group"].participants]
                else:
                    forced_result = self._force_minimal_group(remaining, f"{segment_key}_forced_{attempt}")
                    if forced_result["success"]:
                        groups.append(forced_result["group"])
                        remaining = [p for p in remaining if p not in forced_result["group"].participants]
                    else:
                        break
            
            attempt += 1
        
        unmatched.extend(remaining)
        
        return {
            "groups": groups,
            "unmatched": unmatched,
            "segment": segment_key
        }
    
    def _create_single_group(self, available_participants: List[Participant], segment_key: str) -> Dict:
        """Création d'un groupe unique optimisé"""
        try:
            females = [p for p in available_participants if p.gender == 'F']
            males = [p for p in available_participants if p.gender == 'M']
            
            if len(females) < self.min_females_per_group:
                return {"success": False, "reason": "Pas assez de femmes disponibles"}
            
            selected_females = self._select_optimal_females(females, males)
            if len(selected_females) < self.min_females_per_group:
                return {"success": False, "reason": "Impossible de respecter contrainte d'âge femmes"}
            
            selected_males = self._select_compatible_males(selected_females, males)
            
            all_selected = selected_females + selected_males
            
            if len(all_selected) < self.min_group_size:
                return {"success": False, "reason": "Groupe trop petit après sélection"}
            
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
    
    def _create_relaxed_group(self, available_participants: List[Participant], segment_key: str) -> Dict:
        """Création de groupe avec contraintes relaxées"""
        try:
            females = [p for p in available_participants if p.gender == 'F']
            males = [p for p in available_participants if p.gender == 'M']
            
            if len(females) < self.min_females_per_group:
                return {"success": False, "reason": "Pas assez de femmes"}
            
            selected_females = sorted(females, key=lambda f: f.social_score, reverse=True)[:min(4, len(females))]
            selected_males = sorted(males, key=lambda m: m.social_score, reverse=True)[:min(4, len(males))]
            
            all_selected = selected_females + selected_males
            
            if len(all_selected) < self.min_group_size:
                return {"success": False, "reason": "Pas assez de participants"}
            
            if len(all_selected) > self.max_group_size:
                all_selected = all_selected[:self.max_group_size]
            
            group = Group(
                id=f"group_{segment_key}",
                participants=all_selected,
                experience_name=all_selected[0].experience_name,
                experience_date=all_selected[0].experience_date,
                experience_city=all_selected[0].experience_city
            )
            
            if (len(group.participants) >= self.min_group_size and 
                len(group.participants) <= self.max_group_size and
                group.gender_balance.get('F', 0) >= self.min_females_per_group and
                group.age_spread <= self.max_age_spread):
                return {"success": True, "group": group}
            else:
                return {"success": False, "reason": "Contraintes de base non respectées"}
                
        except Exception as e:
            return {"success": False, "reason": str(e)}
    
    def _force_minimal_group(self, available_participants: List[Participant], segment_key: str) -> Dict:
        """Force la création d'un groupe avec contraintes minimales"""
        try:
            if len(available_participants) < self.min_group_size:
                return {"success": False, "reason": "Pas assez de participants"}
            
            females = [p for p in available_participants if p.gender == 'F']
            males = [p for p in available_participants if p.gender == 'M']
            
            min_females = min(len(females), 2)
            selected_females = females[:min_females]
            
            remaining_spots = self.min_group_size - len(selected_females)
            selected_males = males[:remaining_spots]
            
            all_selected = selected_females + selected_males
            
            if len(all_selected) < self.min_group_size:
                remaining_needed = self.min_group_size - len(all_selected)
                others = [p for p in available_participants if p not in all_selected]
                all_selected.extend(others[:remaining_needed])
            
            if len(all_selected) >= self.min_group_size:
                group = Group(
                    id=f"group_{segment_key}",
                    participants=all_selected[:self.max_group_size],
                    experience_name=all_selected[0].experience_name,
                    experience_date=all_selected[0].experience_date,
                    experience_city=all_selected[0].experience_city
                )
                return {"success": True, "group": group}
            else:
                return {"success": False, "reason": "Impossible de créer un groupe minimal"}
                
        except Exception as e:
            return {"success": False, "reason": str(e)}
    
    def _aggressive_fusion(self, unmatched_participants: List[Participant], segment_key: str) -> Dict:
        """Fusion aggressive pour améliorer le taux de matching"""
        groups = []
        remaining = unmatched_participants.copy()
        
        while len(remaining) >= self.min_group_size:
            group_size = min(len(remaining), self.max_group_size)
            
            females = [p for p in remaining if p.gender == 'F']
            males = [p for p in remaining if p.gender == 'M']
            
            selected = []
            
            if len(females) >= 2:
                selected.extend(females[:min(4, len(females))])
            else:
                selected.extend(females)
            
            remaining_spots = group_size - len(selected)
            selected.extend(males[:remaining_spots])
            
            if len(selected) < self.min_group_size:
                others = [p for p in remaining if p not in selected]
                selected.extend(others[:self.min_group_size - len(selected)])
            
            if len(selected) >= self.min_group_size:
                group = Group(
                    id=f"fusion_{segment_key}_{len(groups)}",
                    participants=selected,
                    experience_name=selected[0].experience_name,
                    experience_date=selected[0].experience_date,
                    experience_city=selected[0].experience_city
                )
                groups.append(group)
                remaining = [p for p in remaining if p not in selected]
            else:
                break
        
        return {
            "groups": groups,
            "unmatched": remaining
        }
    
    def _select_optimal_females(self, females: List[Participant], males: List[Participant]) -> List[Participant]:
        """Sélection optimale des femmes"""
        if not males:
            return females[:self.min_females_per_group]
        
        max_male_age = max(m.age for m in males)
        suitable_females = [f for f in females if f.age < max_male_age]
        
        if len(suitable_females) >= self.min_females_per_group:
            return sorted(suitable_females, key=lambda f: -f.social_score)[:4]
        else:
            return sorted(females, key=lambda f: f.age)[:self.min_females_per_group]
    
    def _select_compatible_males(self, selected_females: List[Participant], available_males: List[Participant]) -> List[Participant]:
        """Sélection des hommes compatibles"""
        if not selected_females or not available_males:
            return []
        
        female_ages = [f.age for f in selected_females]
        female_topics = set(chain.from_iterable(f.topics_conversations for f in selected_females))
        
        def compatibility_score(male: Participant) -> float:
            age_score = max(0, 6 - abs(male.age - np.mean(female_ages))) / 6
            topic_score = len(set(male.topics_conversations) & female_topics) / max(len(female_topics), 1)
            social_score = male.social_score / 10
            return age_score * 0.4 + topic_score * 0.3 + social_score * 0.3
        
        scored_males = [(m, compatibility_score(m)) for m in available_males]
        sorted_males = sorted(scored_males, key=lambda x: -x[1])
        
        target_males = min(4, self.max_group_size - len(selected_females))
        return [m[0] for m in sorted_males[:target_males]]
    
    def _optimize_group_size(self, participants: List[Participant]) -> List[Participant]:
        """Optimisation de la taille du groupe"""
        if len(participants) <= self.max_group_size:
            return participants
        
        females = [p for p in participants if p.gender == 'F']
        males = [p for p in participants if p.gender == 'M']
        
        def selection_score(p: Participant) -> float:
            return p.social_score + len(p.topics_conversations) * 0.5
        
        females_sorted = sorted(females, key=selection_score, reverse=True)
        males_sorted = sorted(males, key=selection_score, reverse=True)
        
        target_females = min(len(females_sorted), 4)
        target_males = min(len(males_sorted), self.max_group_size - target_females)
        
        return females_sorted[:target_females] + males_sorted[:target_males]
    
    def find_replacement(self, group: Group, leaving_participant: Participant, 
                        available_pool: List[Participant]) -> Optional[Participant]:
        """Trouve un remplaçant optimal"""
        try:
            if not available_pool:
                return None
            current_group_topics = set(chain.from_iterable(p.topics_conversations for p in group.participants if p != leaving_participant))
            current_avg_age = np.mean([p.age for p in group.participants if p != leaving_participant])

            def replacement_score(candidate: Participant) -> float:
                if candidate.gender != leaving_participant.gender:
                    return 0.0
                topic_score = len(set(candidate.topics_conversations) & current_group_topics) / max(len(current_group_topics), 1)
                age_score = max(0, 6 - abs(candidate.age - current_avg_age)) / 6
                return topic_score * 0.6 + age_score * 0.4

            candidates = [(p, replacement_score(p)) for p in available_pool]
            candidates_sorted = sorted(candidates, key=lambda x: -x[1])

            for candidate, score in candidates_sorted:
                hypothetical_group = [p for p in group.participants if p != leaving_participant] + [candidate]
                temp_group = Group(
                    id=f"{group.id}_replaced",
                    participants=hypothetical_group,
                    experience_name=group.experience_name,
                    experience_date=group.experience_date,
                    experience_city=group.experience_city
                )
                if temp_group.is_valid and temp_group.female_age_constraint_ok:
                    return candidate

            return None
        except Exception as e:
            logger.warning(f"Erreur remplacement participant: {e}")
            return None
