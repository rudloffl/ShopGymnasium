# Rapport d'avancement — Atelier de production (Version DQN, environnement final)

Ce document décrit l'état **corrigé et à jour** de l'environnement d'atelier industriel utilisé pour l'apprentissage par renforcement (DQN) dans le projet.

Il est désormais **aligné** avec la spécification officielle :
`workshop_environment_specification.md`.

---

## 1. Objectif de l'environnement

L'environnement simule un **atelier industriel multitâche** dans lequel un agent doit :
- planifier la production sur deux machines,
- gérer un **carnet de commandes (backlog)** mis a jour toutes les 15 minutes,
- commander de la matière première avec délai (environ 120 minutes),
- maintenir les stocks à un niveau pertinent malgré un **vol nocturne** (11:55 pm),
- maximiser le revenu sur un horizon de **7 jours (10080 minutes)**.

L'agent sera entraîné avec un algorithme de type **DQN avec SB3** sur un environnement Gymnasium personnalisé (`WorkshopEnv`).

---

## 2. Structure globale du code

L'environnement est organisé en plusieurs modules :

```text
env/
│
├── machines.py       → logique des machines M1 et M2 (batches asynchrones)
├── stock.py          → gestion des stocks (MP, P1, P2_inter, P2)
├── delivery.py       → file d'attente de livraisons de matière première
├── market.py         → génération de la demande + calcul des ventes
└── workshop_env.py   → environnement Gymnasium complet (classe WorkshopEnv)
```


---

## 3. Horizon temporel et dynamique

- **1 step = 1 minute**
- **1 journée = 1440 minutes**
- **1 épisode = 7 jours = 10080 steps**

À chaque step :

1. L'agent choisit une action.
2. Les machines avancent d'une minute (si elles sont occupées).
3. Les éventuels batches terminés mettent à jour les stocks.
4. Les livraisons de MP prévues pour cette minute sont ajoutées au stock.
5. Toutes les **15 minutes**, une nouvelle demande est générée et les ventes sont calculées.
6. Toutes les **1440 minutes**, un vol nocturne réduit les stocks de P1 et P2.
7. Un nouvel état (13 variables) et un reward sont renvoyés à l'agent.

L'épisode se termine lorsque `time >= 10080`.

---

## 4. Composants principaux

### 4.1 Machines (machines.py)

- **M1** :
  - Produit P1 (à partir de MP) : durée d'un batch de taille k = `3 × k` minutes
  - Produit P2_inter (étape 1 de P2) : durée = `10 × k` minutes

- **M2** :
  - Produit P2 (à partir de P2_inter) : durée d'un batch de taille k = `15 × k` minutes

Les machines ont un état interne (`busy`, `time_left`, `batch_type`, `batch_k`) et sont mises à jour via une méthode `tick()` appelée chaque minute.

### 4.2 Stocks (stock.py)

Stocks gérés :
- `stock_raw` : matière première (MP)
- `stock_p1` : produit fini P1
- `stock_p2_inter` : produit intermédiaire P2
- `stock_p2` : produit fini P2

Les méthodes permettent de consommer / ajouter du stock en respectant des bornes logiques (≈ 50 unités).

### 4.3 Livraisons (delivery.py)

- Les commandes de MP sont planifiées avec un **délai d'environ 120 minutes**.
- Chaque commande est stockée comme `(quantité, time_livraison)` dans une file FIFO.
- À chaque minute, les livraisons arrivées sont retirées de la file et ajoutées à `stock_raw`.

L'observation inclut :
- le **temps restant avant la prochaine livraison**,
- la **quantité totale de MP en transit** (somme de toutes les livraisons futures).

### 4.4 Marché et demande (market.py)

La demande n'est plus ponctuelle, mais modélisée comme un **backlog** (carnet de commandes non servies).

- Une fois par quart d'heure (`time % 15 == 0`), on génère une **nouvelle demande horaire** pour P1 et P2 :
  - la demande dépend de l'heure de la journée (**profil jour/nuit**),
  - elle suit une loi de Poisson agrégée sur 60 minutes.

- Cette nouvelle demande est **ajoutée au backlog** :
  ```python
  backlog_p1 += new_d1
  backlog_p2 += new_d2
  ```

- Puis on calcule les ventes possibles à partir des stocks disponibles :
  ```python
  ventes_p1 = min(stock_p1, backlog_p1)
  ventes_p2 = min(stock_p2, backlog_p2)
  ```

- Le backlog est mis à jour :
  ```python
  backlog_p1 -= ventes_p1
  backlog_p2 -= ventes_p2
  ```

- Le reward associé aux ventes est :
  ```python
  reward += 2 * ventes_p1 + 20 * ventes_p2
  ```

La demande visible par l'agent dans l'état est donc **la demande résiduelle** (backlog), pas la demande brute de la dernière heure.

### 4.5 Vol nocturne

Toutes les **1440 minutes** (fin de journée a 11:55pm) :

```python
stock_p1 = floor(stock_p1 * 0.9)
stock_p2 = floor(stock_p2 * 0.9)
```

Seuls les stocks de P1 et P2 sont affectés.  
Ce mécanisme pénalise les stocks excessifs de produits finis et incite à une production mieux synchronisée avec le backlog.

---

## 5. Espace d'actions (WorkshopEnv)

L’espace d’actions est :
```python
self.action_space = spaces.Discrete(201)
```

Signification des actions :

| Actions     | Effet                                       |
|-------------|---------------------------------------------|
| 0 à 49      | Produire P1, avec k = action + 1            |
| 50 à 99     | Produire P2 (étape 1, M1), k = action − 49  |
| 100 à 149   | Produire P2 (étape 2, M2), k = action − 99  |
| 150 à 199   | Commander MP, quantité q = action − 149     |
| **200**     | **WAIT** (ne rien faire)                    |

Règles de validité :
- Si une machine est déjà occupée, une tentative de production sur cette machine devient **WAIT**.
- Si le stock de MP / P2_inter est insuffisant, l’action de production devient **WAIT**.
- Les actions hors `[0, 200]` lèvent une erreur.

---

## 6. Espace d’observation (WorkshopEnv)

L’observation est un vecteur de taille **13** :

1. `time` : minute courante (0 à 10080)
2. `m1_busy` : 0/1
3. `m1_time_left` : temps restant sur le batch M1
4. `m2_busy` : 0/1
5. `m2_time_left` : temps restant sur le batch M2
6. `stock_raw` : stock de MP
7. `stock_p1` : stock de P1
8. `stock_p2_inter` : stock intermédiaire P2
9. `stock_p2` : stock de P2
10. `next_delivery_countdown` : minutes restantes avant la prochaine livraison de MP (0 s’il n’y en a pas)
11. `backlog_p1` : demande résiduelle (carnet de commandes P1 non servies)
12. `backlog_p2` : demande résiduelle (carnet de commandes P2 non servies)
13. `q_total_en_route` : quantité totale de MP en transit (commandée mais non livrée)

Cet état est conçu pour être **Markovien** : l’agent dispose de toutes les informations nécessaires pour prendre des décisions optimales sans mémoire externe.

---

## 7. Reward

Le reward par step peut inclure :

- **revenu des ventes** (uniquement à l’heure pleine) :
  - `+ 2 * ventes_p1`
  - `+ 20 * ventes_p2`
- **coût des commandes de MP** :
  - `− q` lors d’une action de commande
- pénalité explicite appliquee sur le backlog 

---

## 8. Utilisation avec DQN

L’environnement `WorkshopEnv` est compatible Gymnasium et peut être utilisé avec un agent DQN (par exemple via Stable-Baselines3) :

- action space : `Discrete(201)`
- observation : vecteur `np.array` de taille 13 (`dtype=float32`)
- horizon long : 10080 steps / épisode
- reward non trivial : combinaison ventes − coûts − vol

Il est recommandé :

- de normaliser les observations (time, stocks, backlog),
- de suivre, en plus du reward :
  - l’évolution moyenne du backlog,
  - le taux de satisfaction de la demande,
  - l’utilisation des machines,
  - l’évolution des stocks de P1 / P2.

---





## 9. Vérification complète de l’environnement avec le notebook `Workshop_Test`

L’ensemble des tests décrits ci-dessous provient du fichier : Workshop_Test.py. Ils valident toutes les dynamiques essentielles de l’environnement, de la production aux livraisons, en passant par les ventes, le backlog et le vol nocturne.

### 9.1 Production P1 (k = 3)
Vérification du fonctionnement de M1 avec un batch de taille 3, durée correcte (3×k), mise à jour du stock et progression temporelle valide.

### 9.2 Production P2 (STEP1 → STEP2)
Contrôle du flux complet P2 : STEP1 sur M1 puis STEP2 sur M2, consommations et productions correctes, cohérence des états machines.

### 9.3 Production P2 multiple (k = 3)
Validation de la cohérence temporelle et du flux matière pour un batch plus large (k=3).

### 9.4 Vol nocturne
Test du mécanisme de réduction des stocks de P1 et P2 à t mod 1440 = 1435, sans impact sur MP ou P2_inter.

### 9.5 Livraisons multiples avec jitter
Tests de commandes successives de MP : délais 120±2, FIFO respectée, aucune livraison écrasée.

### 9.6 Livraison + vente simultanée
Cas critique : livraison tombant exactement à une heure pleine (t=120). Vérification que la vente ne touche jamais au stock MP, cohérence backlog/ventes.

### 9.7 Livraison + vente + vol nocturne
Superposition des événements critiques proches de minuit : livraisons, vente (t=1440), vol (t=1435). Aucune interaction indésirable.

