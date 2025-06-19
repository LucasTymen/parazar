import pytest
from parazar.models.participant import Participant

def test_participant_creation_full():
    participant = Participant(
        email="test@mail.com",
        prenom="Jean",
        nom="Dupont",
        age=30,
        genre="Homme",
        ville="Paris",
        ville_non_listée=None,
        sujets_aimes=["Culture", "Relations"],
        type_personnalité="Introverti",
        introverti_degré=4,
        frequence_sorties="Parfois",
        aime_sport=True,
        temps_libre="Seul",
        priorites_vie=["Bonheur", "Santé"],
        budget_sortie="10-30€",
        activites_rencontre=["Café", "Balade"],
        jours_disponibles=["Samedi", "Dimanche"],
        experience_name="Exp Café",
        experience_date="2024-05-12",
        experience_city="Paris",
        topics=["Actualités & Politique", "Art, musique & culture"],
        type_personnalite="Calme et réfléchie",
        introversion=0.7,
        sport=0.5,
        lieu_preference="En ville",
        priorites=["Mon travail", "Ma liberté"],
        budget_sorties="20-40€",
        activites_recherchees="Sortie cinéma",
        jours_dispo=["Lundi", "Mardi"],
        attentes=["Rencontrer de nouvelles personnes", "Trouver une relation long terme"],
        signe_zodiaque="Poissons",
        relation="Célibataire",
        secteur="Tech",
        birth_date="1994-01-01",
        birth_year=1994,
        age_bucket="25-34",
        amis=[{"nom": "Ami1", "téléphone": "0600000000"}]
    )
    assert participant.email == "test@mail.com"
    assert participant.prenom == "Jean"
    assert participant.age == 30
    assert participant.sujets_aimes is not None and "Culture" in participant.sujets_aimes
    assert participant.type_personnalité == "Introverti"
    assert participant.introverti_degré == 4
    assert participant.frequence_sorties == "Parfois"
    assert participant.aime_sport is True
    assert participant.temps_libre == "Seul"
    assert participant.priorites_vie is not None and "Bonheur" in participant.priorites_vie
    assert participant.budget_sortie == "10-30€"
    assert participant.activites_rencontre is not None and "Café" in participant.activites_rencontre
    assert participant.jours_disponibles is not None and "Samedi" in participant.jours_disponibles
    assert participant.experience_name == "Exp Café"
    assert participant.topics is not None and len(participant.topics) > 0 and participant.topics[0] == "Actualités & Politique"
    assert participant.type_personnalite == "Calme et réfléchie"
    assert participant.introversion == 0.7
    assert participant.sport == 0.5
    assert participant.lieu_preference == "En ville"
    assert participant.priorites is not None and "Mon travail" in participant.priorites
    assert participant.budget_sorties == "20-40€"
    assert participant.activites_recherchees == "Sortie cinéma"
    assert participant.jours_dispo is not None and "Lundi" in participant.jours_dispo
    assert participant.attentes is not None and "Rencontrer de nouvelles personnes" in participant.attentes
    assert participant.signe_zodiaque == "Poissons"
    assert participant.relation == "Célibataire"
    assert participant.secteur == "Tech"
    assert participant.birth_date == "1994-01-01"
    assert participant.birth_year == 1994
    assert participant.age_bucket == "25-34"
    assert participant.amis and participant.amis[0].get("nom") == "Ami1"