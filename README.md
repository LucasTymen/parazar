# Parazar - Matching Intelligence

Ce projet implémente un moteur de matching pour constituer des groupes compatibles lors d’expériences sociales ou événementielles (type speed-dating, team-building…).

## 🔧 Fonctionnalités

- Matching basé sur :
  - L’âge
  - Le genre
  - Le degré d’introversion
  - Les intérêts communs
- Calcul d’un score de compatibilité
- Constitution de groupes optimaux

## 🚀 Démarrage rapide

```bash
pip install pandas numpy
python3 parazar_matching.py


🧠 Objectif de l’algorithme Parazar

Créer des groupes de participants compatibles pour des rencontres sociales, à partir des réponses à un formulaire de type Tally (test de personnalité / préférences sociales).
⚙️ Tâches que remplit l'algorithme

    Chargement des participants depuis un fichier ou formulaire (CSV ou JSON).

    Validation des données via des validateurs dédiés (âge, genre, email, etc.).

    Normalisation des réponses pour permettre le traitement des ENUMs.

    Matching par similarité selon des critères définis (cf. ci-dessous).

    Attribution automatique en groupes (nombre variable).

    Évitement de re-matching avec les mêmes personnes que par le passé.

    Capacité à gérer les absences ou reports, en remplaçant par le meilleur match suivant.

    Possibilité pour les utilisateurs de matcher ou se dématcher eux-mêmes.

    Émission d’un score d'affinité entre chaque participant du groupe.

    Création d’un rapport de matching (score, raisons, logiques de sélection).

🧱 Champs traités (issus du formulaire Tally)

{
  "ville": "string",
  "ville_non_listée": "string",
  "sujets_préférés": ["actualités", "relations", "culture", "société", "spiritualité", "autres"],
  "type_personnalité": "extraverti | introverti",
  "introverti_degré": "1-5",
  "fréquence_sorties": "jamais | parfois | souvent | tout le temps",
  "aime_sport": "oui | non",
  "temps_libre": "seul | avec amis | peu importe",
  "priorités_vie": ["famille", "carrière", "argent", "bonheur", "santé", "autre"],
  "budget_sortie": "moins de 10€ | 10-30€ | 30-50€ | plus de 50€",
  "activités_rencontre": ["sport", "jeux", "balade", "café", "culture", "autre"],
  "jours_disponibles": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
}

🔢 Catégories d’ENUMs utilisées dans le scoring
🎭 Personnalité

    "type_personnalité" : ["extraverti", "introverti"]

    "introverti_degré" : 1 à 5 (pondération)

🔄 Habitudes sociales

    "fréquence_sorties" : ["jamais", "parfois", "souvent", "tout le temps"]

    "temps_libre" : ["seul", "avec amis", "peu importe"]

    "aime_sport" : ["oui", "non"]

💬 Centres d’intérêt / Valeurs

    "sujets_préférés" : ["actualités", "relations", "culture", "société", "spiritualité", "autres"]

    "priorités_vie" : ["famille", "carrière", "argent", "bonheur", "santé", "autre"]

    "activités_rencontre" : ["sport", "jeux", "balade", "café", "culture", "autre"]

💰 Budget

    "budget_sortie" : ["moins de 10€", "10-30€", "30-50€", "plus de 50€"]

📆 Jours disponibles

    "jours_disponibles" : ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

🧩 Classes Python dans l’architecture
✅ Participant

    Données issues du formulaire

    Méthodes de normalisation

    Support des consentements et historique de matching

✅ Group

    Ensemble de Participant

    Score moyen

    Méthodes de rotation/remplacement

✅ Matcher

    Logique de matching entre participants

    Scoring par similarité pondérée

    Allocation des groupes

✅ Validators

    Fichier avec validateurs par champ (email, âge, genre, etc.)

✅ Autres fichiers ou éléments liés

    enums.json : contient tous les ENUMs réels extraits du Tally

    consentements.json : liste des relations passées pour éviter les re-matching

    test_participant.py : tests unitaires sur la structure et les comportements

🛠️ Tests Pytest & Structure

Les tests incluent :

    Chargement & validation des participants

    Fonction match_participants

    Fonction create_groups

    Résilience face à des cas incomplets

Commandes CLI :

pytest
# ou avec couverture
pytest --cov=parazar --cov-report=term-missing


Process

---

## 📦 Installation

```bash
git clone https://github.com/LucasTymen/parazar.git
cd parazar
pip install -r requirements.txt

    ⚠️ Assurez-vous d'utiliser Python 3.10+.

🚀 Exécution locale

    Ajoutez vos participants dans data/participants.csv ou data/participants.json
    (structure conforme au fichier de test fourni)

    Lancer le script principal :

python demo_parazar_matching.py

    Résultats :

        Un fichier groupes.json est généré dans le dossier output/

        Le log de matching est enregistré dans logs/rapport_matching.log

🧪 Tester l’algorithme

pytest
# ou avec couverture :
pytest --cov=parazar --cov-report=term-missing

🔄 Intégration avec React Native
Option 1 – Fichier JSON partagé

    Le fichier groupes.json peut être lu dans l’app mobile via une API ou par lecture locale (selon le mode offline/online).

    Il contient tous les participants et leurs affectations.

Option 2 – API Flask (à venir)

Un microservice sera ajouté pour exposer :

GET /api/groupes
POST /api/participants

🧠 Données traitées

    Fichier d’entrée : participants.csv ou .json basé sur le formulaire Tally

    ENUMs utilisés : enums.json

    Consentements passés : consentements.json (évite les rematchs)

    Historique des groupes : enregistré automatiquement à chaque exécution

📁 Structure des fichiers

parazar/
├── models/
│   └── participant.py, group.py
├── scoring/
│   └── group_scorer.py
├── validators/
├── tests/
├── demo_parazar_matching.py
├── consentements.json
├── enums.json
├── output/groupes.json

🧩 À venir

    Intégration temps réel avec Firebase ou Supabase

    Interface admin pour forcer des matches / visualiser les scores

    Gestion multiville et fuseaux horaires

👤 Auteurs

    Lucas Tymen