import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from itertools import combinations, chain
from functools import reduce
from collections import defaultdict, Counter
import logging
from enum import Enum
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchingStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_CONSTRAINTS = "failed_constraints"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"

@dataclass
class Participant:
    email: str
    first_name: str
    gender: str
    age: int
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
    introverted_degree: float = 0.0
    social_score: float = field(default=0.0, init=False)
    compatibility_topics: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.gender = self.gender.strip().lower()
        self.gender = 'M' if self.gender == 'm' else 'F' if self.gender == 'f' else self.gender.upper()
        self.social_score = round((1 - self.introverted_degree) * 10, 2)
        self.compatibility_topics = set(self.topics_conversations)

@dataclass
class Group:
    id: str
    participants: List[Participant] = field(default_factory=list)
    experience_name: str = ""
    experience_date: str = ""
    experience_city: str = ""
    compatibility_score: float = field(default=0.0, init=False)
    age_spread: float = field(default=0.0, init=False)
    gender_balance: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._update_metrics()

    def _update_metrics(self):
        if not self.participants:
            return
        ages = [p.age for p in self.participants]
        self.age_spread = max(ages) - min(ages)
        self.gender_balance = Counter(p.gender for p in self.participants)
        self.compatibility_score = self._calculate_compatibility()

    def _calculate_compatibility(self) -> float:
        if len(self.participants) < 2:
            return 0.0
        all_topics = [p.compatibility_topics for p in self.participants]
        topic_overlap = len(set.intersection(*all_topics)) if all_topics else 0
        social_scores = [p.social_score for p in self.participants]
        std_dev = np.std(social_scores) or 1
        social_balance = 1 - (std_dev / 10)
        return round((topic_overlap * 2 + social_balance * 3), 2)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.participants) >= 4 and
            len(self.participants) <= 8 and
            self.gender_balance.get('F', 0) >= 2 and
            self.age_spread <= 6
        )

    @property
    def needs_female(self) -> bool:
        return self.gender_balance.get('F', 0) < 2

    @property
    def female_age_constraint_ok(self) -> bool:
        if not self.participants:
            return True
        females = [p for p in self.participants if p.gender == 'F']
        if not females:
            return True
        return max(f.age for f in females) <= max(p.age for p in self.participants)
class ParazarMatcher:
    def __init__(self, min_group_size: int = 4, max_group_size: int = 8, 
                 max_age_spread: int = 6, min_females_per_group: int = 2):
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.max_age_spread = max_age_spread
        self.min_females_per_group = min_females_per_group
        self.groups: List[Group] = []
        self.unmatched: List[Participant] = []

    def load_from_dataframe(self, df: pd.DataFrame) -> List[Participant]:
        try:
            required_fields = ['email', 'first_name', 'age', 'gender']
            df_clean = df.dropna(subset=required_fields)

            participants = [self._create_participant_from_row(row) for row in df_clean.to_dict('records')]
            valid_participants = [p for p in participants if 18 <= p.age <= 65 and p.gender in ['M', 'F'] and p.email]

            logger.info(f"Chargé {len(valid_participants)} participants valides sur {len(df)} lignes")
            return valid_participants
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return []

    def _create_participant_from_row(self, row: Dict) -> Participant:
        try:
            topics = [s.strip() for s in str(row.get('topics_conversations', '')).split(',') if s.strip()]
            priorities = [s.strip() for s in str(row.get('life_priorities', '')).split(',') if s.strip()]
            return Participant(
                email=str(row.get('email', '')),
                first_name=str(row.get('first_name', '')),
                gender=str(row.get('gender', '')),
                age=int(row.get('age', 0)),
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
                introverted_degree=float(row.get('introverted_degree', 0.5))
            )
        except Exception as e:
            logger.warning(f"Erreur création participant: {e}")
            raise

    def create_optimal_groups(self, participants: List[Participant]) -> Tuple[MatchingStatus, Dict]:
        try:
            if len(participants) < self.min_group_size:
                return MatchingStatus.INSUFFICIENT_DATA, {
                    "error": "Pas assez de participants",
                    "required": self.min_group_size,
                    "available": len(participants)
                }

            segments = self._segment_participants(participants)
            results = {
                "groups": [],
                "unmatched": [],
                "stats": {},
                "segments_processed": len(segments)
            }

            for segment_key, seg_participants in segments.items():
                segment_result = self._process_segment(seg_participants, segment_key)
                results["groups"].extend(segment_result["groups"])
                results["unmatched"].extend(segment_result["unmatched"])

            results["stats"] = self._calculate_global_stats(results["groups"], results["unmatched"])
            return self._determine_status(results), results

        except Exception as e:
            logger.error(f"Erreur lors de la création des groupes: {e}")
            return MatchingStatus.ERROR, {"error": str(e)}

    def _segment_participants(self, participants: List[Participant]) -> Dict[str, List[Participant]]:
        segments = defaultdict(list)
        for p in participants:
            key = f"{p.experience_name}_{p.experience_date}_{p.experience_city}"
            segments[key].append(p)
        return dict(segments)

    def _process_segment(self, participants: List[Participant], segment_key: str) -> Dict:
        logger.info(f"Traitement segment {segment_key}: {len(participants)} participants")

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

        while len(remaining) >= self.min_group_size:
            group_result = self._create_single_group(remaining, segment_key)
            if group_result["success"]:
                groups.append(group_result["group"])
                used = set(group_result["group"].participants)
                remaining = [p for p in remaining if p not in used]
            else:
                fusion_result = self._attempt_group_fusion(remaining, segment_key)
                if fusion_result["success"]:
                    groups.append(fusion_result["group"])
                    remaining = []
                else:
                    break

        unmatched.extend(remaining)

        return {
            "groups": groups,
            "unmatched": unmatched,
            "segment": segment_key
        }
    def _create_single_group(self, available: List[Participant], segment_key: str) -> Dict:
        try:
            females = [p for p in available if p.gender == 'F']
            males = [p for p in available if p.gender == 'M']

            if len(females) < self.min_females_per_group:
                return {"success": False, "reason": "Pas assez de femmes"}

            selected_females = sorted(females, key=lambda f: -f.social_score)[:self.min_females_per_group]
            max_female_age = max(f.age for f in selected_females)
            compatible_males = [m for m in males if m.age >= max_female_age - self.max_age_spread]

            if len(compatible_males) < (self.min_group_size - len(selected_females)):
                return {"success": False, "reason": "Pas assez d'hommes compatibles"}

            selected_males = sorted(compatible_males, key=lambda m: -m.social_score)[:self.max_group_size - len(selected_females)]
            all_participants = selected_females + selected_males

            group = Group(
                id=f"group_{segment_key}_{len(self.groups) + 1}",
                participants=all_participants,
                experience_name=selected_females[0].experience_name,
                experience_date=selected_females[0].experience_date,
                experience_city=selected_females[0].experience_city
            )

            if group.is_valid and group.female_age_constraint_ok:
                return {"success": True, "group": group}
            return {"success": False, "reason": "Contraintes non respectées"}

        except Exception as e:
            logger.warning(f"Erreur création groupe: {e}")
            return {"success": False, "reason": str(e)}

    def _attempt_group_fusion(self, remaining: List[Participant], segment_key: str) -> Dict:
        if len(remaining) < self.min_group_size:
            return {"success": False}
        return self._create_single_group(remaining, f"{segment_key}_fusion")

    def _calculate_global_stats(self, groups: List[Group], unmatched: List[Participant]) -> Dict:
        total = sum(len(g.participants) for g in groups) + len(unmatched)
        matched = sum(len(g.participants) for g in groups)

        return {
            "total_participants": total,
            "groups_created": len(groups),
            "participants_matched": matched,
            "participants_unmatched": len(unmatched),
            "matching_rate": round(matched / total * 100, 2) if total else 0,
            "avg_group_size": round(np.mean([len(g.participants) for g in groups]), 2) if groups else 0,
            "avg_compatibility_score": round(np.mean([g.compatibility_score for g in groups]), 2) if groups else 0,
            "valid_groups": sum(1 for g in groups if g.is_valid)
        }

    def _determine_status(self, results: Dict) -> MatchingStatus:
        rate = results["stats"].get("matching_rate", 0)
        if rate >= 90:
            return MatchingStatus.SUCCESS
        elif rate >= 70:
            return MatchingStatus.PARTIAL_SUCCESS
        elif results["stats"].get("valid_groups", 0) == 0:
            return MatchingStatus.FAILED_CONSTRAINTS
        return MatchingStatus.PARTIAL_SUCCESS

    def find_replacement(self, group: Group, leaving: Participant, pool: List[Participant]) -> Optional[Participant]:
        try:
            filtered = [
                p for p in pool
                if p.experience_name == group.experience_name and
                   p.experience_date == group.experience_date and
                   p.experience_city == group.experience_city
            ]
            if not filtered:
                return None

            def score(p: Participant) -> float:
                new_group = Group(
                    id="test",
                    participants=[x for x in group.participants if x != leaving] + [p],
                    experience_name=group.experience_name,
                    experience_date=group.experience_date,
                    experience_city=group.experience_city
                )
                return new_group.compatibility_score if new_group.is_valid and new_group.female_age_constraint_ok else -1

            scored = [(p, score(p)) for p in filtered]
            valid = [(p, s) for p, s in scored if s >= 0]
            return max(valid, key=lambda x: x[1])[0] if valid else None

        except Exception as e:
            logger.error(f"Erreur remplacement: {e}")
            return None
