import pandas as pd
import json
from typing import List, Dict, Set, Tuple
from itertools import combinations
import os

# --- Chargement des données ---
PARTICIPANTS_CSV = "participants.csv"  # À adapter si besoin
df = pd.read_csv(PARTICIPANTS_CSV)
df['id'] = df.index

# --- Fonctions utilitaires pour JSON ---
def load_json(filename, default):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# --- Chargement des fichiers annexes ---
history = set(tuple(sorted(pair)) for pair in load_json("history.json", []))
manual_matches = [tuple(sorted(pair)) for pair in load_json("manual_matches.json", [])]
custom_groups = load_json("custom_groups.json", [])

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
    return all(tuple(sorted((a, b))) not in history for a, b in combinations(pids, 2))

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
    return groups

groups = create_groups(df, used_ids)

# --- Remplacement automatique ---
def replace_in_group(groups, group_id, leaving_id, df, history):
    group = groups[group_id]
    if leaving_id not in group:
        print("Participant non présent dans ce groupe.")
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
        print(f"Participant {best_id} remplace {leaving_id} dans le groupe {group_id}")
    else:
        print("Aucun remplaçant trouvé.")

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