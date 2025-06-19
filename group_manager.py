import pandas as pd
import json
from typing import List, Dict, Set, Tuple
from itertools import combinations
import os
import logging
from parazar.validators.participant_validator import ParticipantValidator
from parazar.validators.email_validator import EmailValidator

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Chargement des données ---
PARTICIPANTS_CSV = "participants.csv"  # À adapter si besoin
df = pd.read_csv(PARTICIPANTS_CSV)
df['id'] = df.index

# --- Fonctions utilitaires pour JSON ---
def safe_load_json(filename, default, expected_type=None):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(f"Type inattendu dans {filename}")
        return data
    except Exception as e:
        logging.error(f"Erreur chargement {filename}: {e}")
        return default

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# --- Chargement des fichiers annexes ---
history = set(tuple(sorted(pair)) for pair in safe_load_json("history.json", []))
manual_matches = [tuple(sorted(pair)) for pair in safe_load_json("manual_matches.json", [])]
custom_groups = safe_load_json("custom_groups.json", [])

# --- Chargement des ENUMs dynamiques ---
ENUMS = safe_load_json("enums.json", {})

def validate_enum(field, value):
    if field in ENUMS:
        if isinstance(value, list):
            return all(v in ENUMS[field] for v in value)
        else:
            return value in ENUMS[field]
    return True

# --- Chargement consentements mutuels ---
consentements = safe_load_json("consentements.json", [])
consent_dict = {}
for c in consentements:
    key = tuple(sorted([c["id1"], c["id2"]]))
    consent_dict[key] = c["consent"]

def has_mutual_consent(id1, id2):
    key = tuple(sorted([id1, id2]))
    return consent_dict.get(key, None)

def compute_score(p1: Dict, p2: Dict) -> int:
    score = 0
    # Sujets préférés (intersection)
    sujets_1 = set(str(p1.get("sujets_préférés", "")).split(","))
    sujets_2 = set(str(p2.get("sujets_préférés", "")).split(","))
    score += len(sujets_1 & sujets_2)
    # Activités
    act_1 = set(str(p1.get("activités_rencontre", "")).split(","))
    act_2 = set(str(p2.get("activités_rencontre", "")).split(","))
    score += len(act_1 & act_2)
    # Jours disponibles
    jours_1 = set(str(p1.get("jours_disponibles", "")).split(","))
    jours_2 = set(str(p2.get("jours_disponibles", "")).split(","))
    score += len(jours_1 & jours_2)
    # Budget, personnalité, sport, temps libre
    for field in ["budget_sortie", "type_personnalité", "aime_sport", "temps_libre"]:
        if str(p1.get(field, "")).strip() == str(p2.get(field, "")).strip():
            score += 1
    # Age proche
    try:
        if abs(int(p1.get("age", 0)) - int(p2.get("age", 0))) <= 5:
            score += 1
    except:
        pass
    return score

# --- Initialisation des groupes ---
groups = {}
used_ids = set()

# 1. Groupes custom
for g in custom_groups:
    gid = g.get("group_id", f"custom_{len(groups)}")
    members = g["members"]
    groups[gid] = members
    used_ids.update(members)
    for pair in combinations(members, 2):
        history.add(tuple(sorted(pair)))

# 2. Paires imposées
for pair in manual_matches:
    if pair[0] in used_ids or pair[1] in used_ids:
        continue
    gid = f"manual_{len(groups)}"
    groups[gid] = list(pair)
    used_ids.update(pair)
    history.add(pair)

# 3. Groupes restants par matching
def can_group(pids: List[int], history: Set[Tuple[int, int]]) -> bool:
    for a, b in combinations(pids, 2):
        if tuple(sorted((a, b))) in history:
            logging.info(f"Refus: {a} et {b} déjà groupés (historique)")
            return False
        consent = has_mutual_consent(a, b)
        if consent is False:
            logging.info(f"Refus: {a} et {b} ont refusé d'être ensemble (consentement)")
            return False
    return True

def create_groups(df, used_ids, group_size=4):
    participants = [p for p in df.to_dict(orient="records") if p["id"] not in used_ids]
    ungrouped = set(p["id"] for p in participants)
    while len(ungrouped) >= group_size:
        seed = ungrouped.pop()
        seed_p = df.loc[df["id"] == seed].to_dict(orient="records")[0]
        scores = []
        for p in participants:
            if p["id"] in ungrouped:
                score = compute_score(seed_p, p)
                scores.append((p["id"], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_matches = []
        for pid, _ in scores:
            candidate_group = [seed] + best_matches + [pid]
            if len(candidate_group) > group_size:
                break
            if can_group(candidate_group, history):
                best_matches.append(pid)
            if len(best_matches) == group_size - 1:
                break
        group = [seed] + best_matches
        if len(group) == group_size:
            gid = f"auto_{len(groups)}"
            groups[gid] = group
            used_ids.update(group)
            for pair in combinations(group, 2):
                history.add(tuple(sorted(pair)))
            for pid in group:
                if pid in ungrouped:
                    ungrouped.remove(pid)
            logging.info(f"Groupe {gid} créé avec membres {group}")
        else:
            logging.warning(f"Impossible de créer un groupe complet à partir de {seed}")
    return groups

groups = create_groups(df, used_ids)

# --- Remplacement automatique ---
def replace_in_group(groups, group_id, leaving_id, df, history):
    group = groups[group_id]
    if leaving_id not in group:
        logging.warning(f"Participant {leaving_id} non présent dans le groupe {group_id}.")
        return
    group.remove(leaving_id)
    all_grouped = set(pid for g in groups.values() for pid in g)
    all_ids = set(df["id"])
    candidates = all_ids - all_grouped
    best_score = -1
    best_id = None
    for cid in candidates:
        if any(tuple(sorted((cid, mid))) in history for mid in group):
            continue
        if any(has_mutual_consent(cid, mid) is False for mid in group):
            continue
        scores = []
        cand = df.loc[df["id"] == cid].to_dict(orient="records")[0]
        for member_id in group:
            member = df.loc[df["id"] == member_id].to_dict(orient="records")[0]
            scores.append(compute_score(cand, member))
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_id = cid
    if best_id is not None:
        group.append(best_id)
        for mid in group:
            history.add(tuple(sorted((best_id, mid))))
        logging.info(f"Participant {best_id} remplace {leaving_id} dans le groupe {group_id}")
    else:
        logging.warning(f"Aucun remplaçant trouvé pour {leaving_id} dans le groupe {group_id}.")

# --- Export ---
save_json("output_groups.json", groups)
save_json("history.json", [list(pair) for pair in history])

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    print("\nGroupes finaux :")
    for gid, members in groups.items():
        print(f"Groupe {gid}: {members}")
    # Exemple de remplacement
    gid = list(groups.keys())[0]
    leave_id = groups[gid][0]
    print(f"\nSuppression du participant {leave_id} du groupe {gid}")
    replace_in_group(groups, gid, leave_id, df, history)
    print("\nGroupes après remplacement :")
    for gid, members in groups.items():
        print(f"Groupe {gid}: {members}")

    # Exemple de validation d'email sur un participant
    participant = groups[gid][0] if isinstance(groups[gid], list) else groups[gid]
    email = participant.get('email') if isinstance(participant, dict) else getattr(participant, 'email', None)
    valid, err = EmailValidator.validate(email or '')
    if not valid:
        raise ValueError(f"Email invalide: {email} ({err})")

    def ma_fonction():
        pass 