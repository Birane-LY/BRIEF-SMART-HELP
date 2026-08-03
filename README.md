# Smart Help API

API intelligente d'analyse automatique des réclamations clients e-commerce.

---

## Table des matières

1. [Contexte](#1-contexte)
2. [Stack technique](#2-stack-technique)
3. [Architecture](#3-architecture)
4. [Structure du projet](#4-structure-du-projet)
5. [Installation](#5-installation)
6. [Configuration](#6-configuration)
7. [Lancement](#7-lancement)
8. [Endpoint principal](#8-endpoint-principal)
9. [Pipeline de traitement](#9-pipeline-de-traitement)
10. [Services IA](#10-services-ia)
11. [Base de connaissances RAG](#11-base-de-connaissances-rag)
12. [Diagnostic et scoring](#12-diagnostic-et-scoring)
13. [Gestion des erreurs](#13-gestion-des-erreurs)
14. [Optimisations mémoire](#14-optimisations-mémoire)
15. [Fichiers temporaires](#15-fichiers-temporaires)
16. [Git Flow](#16-git-flow)
17. [Tests Swagger](#17-tests-swagger)
18. [Exemples complets](#18-exemples-complets)
19. [Statuts possibles](#19-statuts-possibles)
20. [Améliorations futures](#20-améliorations-futures)

---

## 1. Contexte

Le service client d'une entreprise e-commerce est submergé par les réclamations. Les clients envoient des notes vocales, des photos de produits endommagés et des messages texte via des applications de messagerie. L'équipe perd un temps précieux à tout analyser manuellement.

Smart Help automatise cette première analyse :
- Transcription automatique des messages audio
- Classification visuelle des images de produits
- Consultation du règlement interne via recherche sémantique
- Proposition de statut pour chaque ticket

---

## 2. Stack technique

| Composant | Technologie |
|---|---|
| Framework | FastAPI |
| Serveur | Uvicorn |
| Transcription | faster-whisper (CTranslate2) |
| Vision | CLIP zero-shot (openai/clip-vit-base-patch32) |
| Embeddings RAG | sentence-transformers (multilingual-MiniLM-L12-v2) |
| Index vectoriel | FAISS (faiss-cpu) |
| Validation | Pydantic v2 |
| Configuration | pydantic-settings + python-dotenv |
| Images | Pillow |
| Python | 3.12+ |

---

## 3. Architecture

```
Client (messagerie)
       |
       v
POST /api/support-ticket (multipart/form-data)
       |
       |-- 1. VALIDATION
       |      Format, taille, combinaisons
       |
       |-- 2. TRANSCRIPTION AUDIO (ASR)
       |      faster-whisper -> texte
       |
       |-- 3. CLASSIFICATION IMAGE (Vision)
       |      CLIP zero-shot -> type de defaut
       |
       |-- 4. RECHERCHE RAG
       |      Texte client -> FAISS -> regle applicable
       |
       |-- 5. DIAGNOSTIC
       |      Score + statut + avertissements
       |
       v
Reponse JSON structuree
```

---

## 4. Structure du projet

```
BRIEF SMART HELP/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Point d'entree FastAPI + exception handlers
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── support_ticket.py    # Endpoint POST /api/support-ticket
│   ├── core/
│   │   ├── config.py                # Configuration centralisee (.env)

│   ├── models/
│   │   ├── __init__.py
│   │   ├── asr_model.py             # Singleton faster-whisper

│   │   └── vision_model.py          # Singleton CLIP
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── ticket_response.py       # Modele de reponse Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_service.py         # Transcription audio
│   │   ├── vision_service.py        # Classification image
│   │   ├── rag_service.py           # Recherche documentaire
│   │   └── diagnostic_service.py    # Scoring et decision finale
│   └── utils/
│       ├── __init__.py
│       ├── file_validation.py       # Validation format et taille
│       └── temp_files.py            # Context manager fichiers temporaires
├── data/
│   ├── knowledge_base/
│   │   └── politique_support.yaml   # Reglement interne (9 regles)
│   ├── audio/

├── .env                              # Variables d'environnement (NON versionne)
├── .env.example                       # Template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Installation

### Prérequis

- Python 3.12+
- pip
- Git
- ~4 Go d'espace disque (modèles IA)

### Étapes

```bash
# Cloner
git clone <URL_DU_REPOSITORY>
cd "BRIEF SMART HELP"

# Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependances
pip install --no-cache-dir -r requirements.txt

# Configuration
cp .env.example .env

# Dossiers
mkdir -p data/cache data/tmp data/index
```

### Vérification

```bash
python -c "import fastapi, faster_whisper, faiss, sentence_transformers; print('OK')"
```

---

## 6. Configuration

Fichier `.env` (jamais committé) :

```env
APP_NAME=Smart Help API
APP_ENV=development
DEBUG=true
API_PREFIX=/api

WHISPER_MODEL_NAME=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
TEXT_MODEL_NAME=Davlan/afro-xlmr-base
VISION_MODEL_NAME=openai/clip-vit-base-patch32
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

MAX_AUDIO_SIZE_MB=10
MAX_IMAGE_SIZE_MB=5
MAX_IMAGES=3
ALLOWED_AUDIO_EXTENSIONS=.mp3,.wav
ALLOWED_IMAGE_EXTENSIONS=.jpg,.jpeg,.png

KNOWLEDGE_BASE_PATH=data/knowledge_base/politique_support.yaml
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.45
VISION_THRESHOLD=0.60

CACHE_DIR=data/cache
TEMP_DIR=data/tmp
```

Toutes les valeurs passent par `app/core/config.py`. Rien n'est codé en dur.

---

## 7. Lancement

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

- Swagger : http://127.0.0.1:8000/docs
- Health : http://127.0.0.1:8000/health

Premier appel lent (téléchargement modèles). Ensuite rapide (singleton).

---

## 8. Endpoint principal

```
POST /api/support-ticket
Content-Type: multipart/form-data
```

### Paramètres

| Champ | Type | Requis | Description |
|---|---|---|---|
| description | string (Form) | Non* | Texte de la réclamation |
| audio | fichier (File) | Non* | Message vocal (.mp3, .wav) |
| images | fichier(s) (File) | Non | Photos produit (.jpg, .jpeg, .png) max 3 |

*Minimum : description OU audio obligatoire.

### Combinaisons

| Entrée | Résultat | Code |
|---|---|---|
| Texte seul | Accepté | 200 |
| Audio seul | Accepté | 200 |
| Texte + images | Accepté | 200 |
| Audio + images | Accepté | 200 |
| Images seules | Rejeté | 422 |
| Rien | Rejeté | 422 |
| Mauvais format | Rejeté | 415 |
| Fichier trop gros | Rejeté | 413 |

---

## 9. Pipeline de traitement

1. **Validation** : format, taille, combinaisons acceptées
2. **Transcription** : faster-whisper convertit l'audio en texte
3. **Texte combiné** : description + transcription fusionnés
4. **Vision** : CLIP classe l'image (défaut, bon état, hors sujet)
5. **RAG** : le texte client est comparé au règlement via FAISS
6. **Diagnostic** : score de fiabilité + statut proposé + avertissements
7. **Réponse JSON** : tout est synthétisé dans une réponse structurée

---

## 10. Services IA

### 10.1 Transcription audio (audio_service.py)

- Modèle : faster-whisper "small" (CTranslate2)
- Chargement unique via `@lru_cache(maxsize=1)`
- Détection automatique de la langue (français, wolof, anglais)
- Fichier temporaire créé dans `data/tmp/`, supprimé dans un context manager (`finally`)
- Entrée : bytes du fichier audio
- Sortie : texte transcrit

Fonctionnement :
```python
with temporary_file(suffix=".wav") as tmp_path:
    tmp_path.write_bytes(audio_bytes)
    segments, _ = model.transcribe(str(tmp_path))
    return " ".join(seg.text for seg in segments)
# fichier supprime automatiquement ici
```

### 10.2 Classification image (vision_service.py)

- Modèle : CLIP (openai/clip-vit-base-patch32)
- Méthode : zero-shot classification
- Labels candidats en français :
  - "produit cassé ou brisé"
  - "produit fissuré ou fendu"
  - "produit rayé ou abîmé"
  - "emballage endommagé"
  - "produit en bon état"
  - "image sans rapport avec un produit"
- Seuil : score >= 0.60 = pertinent
- Entrée : bytes de l'image
- Sortie : label, score, is_relevant

### 10.3 Recherche documentaire (rag_service.py)

- Embeddings : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Index : FAISS IndexFlatIP (similarité cosinus sur vecteurs normalisés)
- Base : 9 règles structurées en YAML
- Seuil : similarité >= 0.45
- Entrée : texte combiné (description + transcription)
- Sortie : règle (id, title, status, action, source, conditions, score)

### 10.4 Analyse textuelle (text_service.py) - Réservé

- Modèle : AfroXLM-R (Davlan/afro-xlmr-base)
- Usage prévu : détection wolof/français, score de pertinence
- Statut : chargé mais non appelé dans le pipeline principal

---

## 11. Base de connaissances RAG

Fichier `data/knowledge_base/politique_support.yaml`

### Règles

| ID | Titre | Statut |
|---|---|---|
| R1.1 | Produit cassé dans les 48h | REMBOURSABLE |
| R1.2 | Produit cassé après 48h | A_VERIFIER |
| R2.1 | Mauvais article reçu | ECHANGE_GRATUIT |
| R2.2 | Pièce manquante | EXPEDITION_PIECE |
| R3.1 | Retard mineur (<=3 jours) | NON_REMBOURSABLE_RETARD_MINEUR |
| R3.2 | Retard majeur (>5 jours) | DEDOMMAGEMENT_10 |
| R3.3 | Colis perdu (>7 jours) | REMBOURSABLE_COLIS_PERDU |
| R4.1 | Usure normale / mauvaise utilisation | REFUSE |
| R4.2 | Absence de preuve | EN_ATTENTE_JUSTIFICATIFS |

### Fonctionnement

1. Premier appel : charge le YAML
2. Chaque règle transformée en texte (title + retrieval_queries)
3. Embeddings générés avec sentence-transformers
4. Index FAISS construit en mémoire
5. À chaque requête : texte client encodé et comparé à l'index
6. Règle avec le meilleur score (au-dessus du seuil) retournée

---

## 12. Diagnostic et scoring

### Score de fiabilité (0 à 1)

| Critère | Poids | Détail |
|---|---|---|
| Texte exploitable | 45% | Texte présent et > 20 caractères |
| Image pertinente | 30% | Label de défaut + score >= seuil |
| Règle RAG trouvée | 25% | Score de similarité de la règle |

### Logique de décision

Ordre de priorité :
1. Image hors sujet + pas de RAG = **REFUSE**
2. RAG a trouvé une règle = **statut de la règle**
3. Image montre un défaut mais pas de règle = **A_VERIFIER**
4. Texte seul sans image ni règle = **EN_ATTENTE_JUSTIFICATIFS**

### Avertissements

Générés quand :
- L'image ne correspond pas à un produit
- Aucune règle du règlement ne correspond
- La confiance de classification est faible

---

## 13. Gestion des erreurs

### Handlers globaux (app/core/exceptions.py)

| Handler | Intercepte | Code |
|---|---|---|
| http_exception_handler | HTTPException du code | Variable |
| validation_exception_handler | Erreurs Pydantic | 422 |
| global_exception_handler | Exceptions non prévues | 500 |

### Format de réponse erreur

```json
{
  "error": {
    "code": "MISSING_CLAIM_CONTEXT",
    "message": "Veuillez accompagner vos images d'une description."
  }
}
```

### Codes d'erreur

| Code | HTTP | Signification |
|---|---|---|
| MISSING_CLAIM_CONTEXT | 422 | Image sans texte ni audio |
| EMPTY_CLAIM | 422 | Ni texte ni audio |
| EMPTY_FILENAME | 422 | Fichier sans nom |
| EMPTY_FILE | 422 | Fichier vide |
| UNSUPPORTED_FILE_TYPE | 415 | Extension non autorisée |
| FILE_TOO_LARGE | 413 | Dépasse la limite |
| VALIDATION_ERROR | 422 | Données invalides (Pydantic) |
| INTERNAL_ERROR | 500 | Erreur serveur |

---

## 14. Optimisations mémoire

### Problème

Les modèles IA pèsent 200 Mo à 1 Go chacun. Les charger à chaque requête saturerait la RAM.

### Solution : Singleton via @lru_cache

```python
@lru_cache(maxsize=1)
def get_asr_model() -> WhisperModel:
    return WhisperModel("small", device="cpu", compute_type="int8")
```

Chaque modèle est chargé une seule fois au premier appel, réutilisé ensuite.

### Modèles

| Fichier | Modèle | RAM estimée |
|---|---|---|
| asr_model.py | faster-whisper small | ~500 Mo |
| vision_model.py | CLIP vit-base-patch32 | ~600 Mo |
| language_model.py | AfroXLM-R (réservé) | ~1.1 Go |
| rag_service.py | multilingual-MiniLM | ~120 Mo |

### compute_type int8

Réduit l'empreinte mémoire de Whisper de ~40% vs float32, perte de qualité négligeable.

---

## 15. Fichiers temporaires

### Problème

faster-whisper exige un chemin fichier sur disque. Il faut écrire l'audio puis le supprimer.

### Solution : Context manager (app/utils/temp_files.py)

```python
@contextmanager
def temporary_file(suffix=".tmp"):
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(delete=False, suffix=suffix, dir=str(temp_dir))
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
```

### Garanties

- Fichier créé dans `data/tmp/`
- Dossier créé automatiquement s'il n'existe pas
- Suppression dans `finally` : même si le modèle plante, le fichier est nettoyé
- Aucun fichier orphelin ne peut s'accumuler

---

## 16. Git Flow

### Branches

| Branche | Rôle |
|---|---|
| main | Version stable, protégée |
| develop | Intégration |
| feature/project-setup | Structure initiale |
| feature/api-ingestion | Endpoint + validation |
| feature/model-loaders | Chargement modèles IA |
| feature/ai-services | Services audio, vision, RAG, diagnostic |
| feature/hardening-documentation | Robustesse + doc |
| release/1.0.0 | Préparation livraison |

### Règles

- Aucun commit direct sur `main` ou `develop`
- Chaque fonctionnalité dans une branche `feature/`
- Pull Request obligatoire vers `develop`
- Release fusionné vers `main` ET `develop`
- Tag `v1.0.0` sur `main`

### Convention commits

```
feat(api): add support ticket ingestion
feat(asr): add audio transcription
feat(vision): add image classification
feat(rag): add policy search
feat(diagnostic): build scoring
chore: setup project structure
```

---

## 17. Tests Swagger

Ouvrir http://127.0.0.1:8000/docs

### Cas 1 : Texte seul
- description : "Mon colis n'est jamais arrivé, ça fait 10 jours"
- Attendu : RAG R3.3, statut REMBOURSABLE_COLIS_PERDU

### Cas 2 : Texte + image produit cassé
- description : "Mon téléphone est arrivé cassé"
- images : photo écran fissuré
- Attendu : vision "produit cassé", RAG R1.1, REMBOURSABLE

### Cas 3 : Image seule (rejet)
- images : photo quelconque
- Attendu : HTTP 422, MISSING_CLAIM_CONTEXT

### Cas 4 : Image hors sujet + texte
- description : "Je veux un remboursement"
- images : photo non pertinente
- Attendu : vision "image sans rapport", REFUSE, warning

### Cas 5 : Audio seul
- audio : fichier .wav
- Attendu : transcription remplie, RAG interrogé

### Cas 6 : Fichier invalide
- audio : fichier .pdf
- Attendu : HTTP 415

---

## 18. Exemples complets

### Requête réussie (texte + image défaut)

```json
{
  "message": "Reclamation recue et analysee.",
  "has_description": true,
  "has_audio": false,
  "image_count": 1,
  "transcription": null,
  "full_text_context": "Mon telephone est arrive avec l'ecran fissure",
  "vision_analysis": [
    {
      "label": "produit fissure ou fendu",
      "score": 0.7823,
      "is_relevant": true
    }
  ],
  "rag_result": {
    "id": "R1.1",
    "title": "Produit casse ou endommage signale dans les 48 heures",
    "status": "REMBOURSABLE",
    "action": "Remboursement integral ou renvoi gratuit",
    "source": "Section 1 - Regle 1.1",
    "conditions": [
      "Le client signale un produit casse, fissure ou endommage",
      "Une photo probante est fournie",
      "La reclamation est faite dans un delai de 48 heures"
    ],
    "score": 0.8456
  },
  "statut_propose": "REMBOURSABLE",
  "score_fiabilite": 0.892,
  "warnings": []
}
```

### Requête rejetée (image seule)

```json
{
  "error": {
    "code": "MISSING_CLAIM_CONTEXT",
    "message": "Veuillez accompagner vos images d'une description ecrite ou d'un message audio."
  }
}
```

### Requête avec image non pertinente

```json
{
  "message": "Reclamation recue et analysee.",
  "has_description": true,
  "has_audio": false,
  "image_count": 1,
  "transcription": null,
  "full_text_context": "Je veux un remboursement",
  "vision_analysis": [
    {
      "label": "image sans rapport avec un produit",
      "score": 0.72,
      "is_relevant": false
    }
  ],
  "rag_result": null,
  "statut_propose": "REFUSE",
  "score_fiabilite": 0.2,
  "warnings": [
    "Image non pertinente : image sans rapport avec un produit",
    "L'image fournie ne correspond pas a un produit."
  ]
}
```

---

## 19. Statuts possibles

| Statut | Signification |
|---|---|
| REMBOURSABLE | Éligible au remboursement intégral |
| A_VERIFIER | Validation manuelle du manager requise |
| REFUSE | Réclamation rejetée (hors garantie) |
| ECHANGE_GRATUIT | Échange pris en charge |
| EXPEDITION_PIECE | Envoi pièce manquante sous 3 jours |
| DEDOMMAGEMENT_10 | Bon d'achat 10% |
| REMBOURSABLE_COLIS_PERDU | Remboursement après enquête transporteur |
| EN_ATTENTE_JUSTIFICATIFS | Preuves insuffisantes |
| NON_REMBOURSABLE_RETARD_MINEUR | Retard <=3 jours, aucun dédommagement |

---

## 20. Améliorations futures

- **AfroXLM-R** : activer le filtre de pertinence avant le RAG
- **Cache disque** : persister les transcriptions entre redémarrages (diskcache)
- **Conditions temporelles** : vérification automatique du délai 48h
- **Multi-images** : agréger les résultats de plusieurs photos
- **Score cohérence** : comparer intention texte vs défaut image
- **Historique** : détecter les réclamations abusives récurrentes
- **Docker** : Dockerfile + docker-compose pour déploiement reproductible