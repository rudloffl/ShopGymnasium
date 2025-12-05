# Spécification de l’Environnement Workshop – Version DQN (finale)

Ce document constitue la **spécification officielle** de l’environnement industriel Workshop utilisé pour l’apprentissage par renforcement (DQN, PPO, etc.).

Il intègre toutes les modifications récentes :

- 1 step = **1 minute**
- **épisode = 7 jours** (10080 minutes)
- machines véritablement **multitâches** (batches asynchrones)
- **demande = backlog résiduel** (carnet de commandes non satisfaites)
- **ventes agrégées toutes les heures**
- demande **jour / nuit** (intensité différente selon l’heure)
- production de P2 en **deux étapes** (M1 puis M2)
- **commandes de MP** avec délai et file d’attente
- **vol nocturne** sur P1 et P2 chaque nuit
- action **WAIT = 200** clairement définie
- état observé = **vecteur de 13 variables**, Markovien et DQN‑ready

---

## 1. Structure temporelle

- 1 step = **1 minute** (résolution de base)
- 1 journée = 1440 minutes
- 1 épisode = **7 jours = 10080 steps**

À chaque step, l’agent choisit **une action** même si les machines sont déjà en train d’exécuter un batch long. Les machines avancent en parallèle, minute par minute.

---

## 2. Ressources et produits

### 2.1 Types de stocks

- `stock_raw` : matière première (MP)
- `stock_p1` : produit fini P1
- `stock_p2_inter` : produit intermédiaire pour P2
- `stock_p2` : produit fini P2

Capacité logique : **50 unités** par type de stock (bornes de l’observation).

---

## 3. Machines

### 3.1 Machine M1

Rôle : transformer de la MP en P1 ou en P2_inter.

- Production de **P1** :
  - 1 MP → 1 P1
  - durée d’un batch de taille *k* : `3 × k` minutes

- Production de **P2_step1** (intermédiaire) :
  - 1 MP → 1 P2_inter
  - durée d’un batch de taille *k* : `10 × k` minutes

État interne de M1 :
- `busy` : 0 ou 1 (libre / occupée)
- `time_left` : minutes restantes avant la fin du batch courant
- `batch_type` : `"P1"` ou `"P2STEP1"` (ou None si inactif)
- `batch_k` : taille du batch en cours

### 3.2 Machine M2

Rôle : transformer P2_inter en P2.

- Production de **P2_step2** :
  - 1 P2_inter → 1 P2
  - durée d’un batch de taille *k* : `15 × k` minutes

État interne de M2 :
- `busy` : 0 ou 1
- `time_left`
- `batch_type` : `"P2STEP2"` ou None
- `batch_k`

### 3.3 Avancement multitâche

À chaque minute (step) :

1. L’agent choisit une action.
2. Les machines (M1, M2) avancent d’une minute (`tick()`).
3. Lorsqu’une machine termine un batch (`time_left` atteint 0) :
   - M1 ajoute **P1** ou **P2_inter** au stock suivant.
   - M2 ajoute **P2** au stock final.
   - Les champs internes (`busy`, `batch_type`, `batch_k`, `time_left`) sont réinitialisés.

Les machines peuvent être occupées pendant de longues durées, mais l’agent agit **chaque minute**.

---

## 4. Commandes de MP et livraisons

### 4.1 Action de commande de MP

Les actions **150 à 199** correspondent à une commande de MP :

- `q = action − 149`  → quantité de MP commandée
- coût immédiat dans le reward : `reward -= q`
- la livraison est programmée pour `time + 120` minutes

Plusieurs commandes peuvent être passées avant la livraison :  
elles s’empilent dans une file (file FIFO) avec leurs propres dates de livraison.

### 4.2 Livraison

La structure `DeliveryQueue` :

- contient une liste de `(quantité, time_livraison)`
- à chaque minute, elle vérifie si certaines livraisons sont arrivées
- les quantités livrées sont ajoutées à `stock_raw`

L’état inclut le **temps restant avant la prochaine livraison** et la **quantité totale de MP en transit** (voir section Observation).

---

## 5. Demande, backlog et ventes

### 5.1 Demande = backlog

Les variables `demande_p1` et `demande_p2` dans l’environnement représentent désormais **le backlog** :

> **backlog = carnet de commandes non satisfaites**  
> (nombre d’unités de P1 / P2 que le système “doit” encore produire pour honorer la demande)

Le backlog évolue **cumulativement** :

1. À chaque période de vente (1 fois par heure), on **ajoute** une nouvelle demande.
2. À chaque vente, on **soustrait** ce qui a pu être servi avec le stock disponible.
3. Le backlog ne revient jamais artificiellement à 0, sauf si les ventes l’absorbent complètement.

### 5.2 Périodicité des ventes : toutes les heures

- Les ventes **ne se produisent qu’une fois par heure**, lorsque `time % 60 == 0` et `time > 0`.
- Entre deux ventes, il n’y a aucune nouvelle demande et le backlog reste constant.

### 5.3 Demande horaire jour / nuit

La demande est modélisée par un **processus de Poisson agrégé sur la période** :

- On distingue deux régimes :
  - **Jour** : de 8h à 20h (minutes 480 à 1199)
  - **Nuit** : le reste du temps

Soit :

- `lambda_p1_day`, `lambda_p2_day` : intensités moyennes par minute en journée
- `lambda_p1_night`, `lambda_p2_night` : intensités moyennes par minute la nuit

Pour une période de 60 minutes, à une heure donnée :

```text
new_d1 ~ Poisson(lambda_p1_(day/night) × 60)
new_d2 ~ Poisson(lambda_p2_(day/night) × 60)
```

Ces valeurs représentent la **nouvelle demande brute** sur l’heure, qui vient s’ajouter au backlog.

### 5.4 Mise à jour du backlog et ventes

À chaque heure pleine (`time % 60 == 0` et `time > 0`) :

1. Génération d’une nouvelle demande horaire :
   ```python
   new_d1, new_d2 = sample_demand(time, period_minutes=60)
   ```
2. Ajout au backlog :
   ```python
   demande_p1 += new_d1
   demande_p2 += new_d2
   ```
3. Calcul des ventes possibles à partir du stock et du backlog :
   ```python
   ventes_p1 = min(stock_p1, demande_p1)
   ventes_p2 = min(stock_p2, demande_p2)
   ```
4. Mise à jour des stocks :
   ```python
   stock_p1 -= ventes_p1
   stock_p2 -= ventes_p2
   ```
5. Mise à jour du backlog (demande résiduelle) :
   ```python
   demande_p1 -= ventes_p1
   demande_p2 -= ventes_p2
   ```
6. Reward de vente :
   ```python
   reward += 2 * ventes_p1 + 20 * ventes_p2
   ```

Ainsi :

- si le stock est suffisant, le backlog peut retomber à 0
- si le stock est insuffisant, le backlog augmente dans le temps

L’agent voit donc **en temps réel la pression de la demande** qu’il n’a pas encore satisfaite.

---

## 6. Vol nocturne

À la fin de chaque journée (toutes les 1440 minutes, pour `time % 1440 == 0` et `time > 0`) :

```python
stock_p1 = floor(stock_p1 * 0.9)
stock_p2 = floor(stock_p2 * 0.9)
```

Le vol porte uniquement sur les stocks de produits finis P1 et P2, pas sur la MP ni l’intermédiaire P2_inter.

Ce mécanisme incite l’agent à :

- éviter de conserver des stocks trop élevés sur P2 (produit à forte valeur)
- mieux planifier sa production en fonction des ventes et du backlog

---

## 7. Reward

Le reward total à chaque minute peut combiner :

- **revenus de vente** (uniquement si minute de vente) :
  - `+ 2 * ventes_p1`
  - `+ 20 * ventes_p2`
- **coût de commande de MP** :
  - `− q` lorsqu’une action de commande est effectuée
- aucune pénalité explicite sur le backlog dans la version de base
  - mais on peut en ajouter une (par ex. `− alpha * backlog_total`) si nécessaire

Il n’y a pas de coût de stockage explicite dans cette spécification (option propre à ajouter si besoin).

---

## 8. Espace d’actions (201 actions)

L’espace d’action est :

```text
Discrete(201)  → actions entières de 0 à 200
```

La signification est :

| Intervalle d’actions | Signification |
|----------------------|---------------|
| 0–49                 | Produire P1 (taille de batch k = action + 1) |
| 50–99                | Produire P2 étape 1 (M1, k = action − 49) |
| 100–149              | Produire P2 étape 2 (M2, k = action − 99) |
| 150–199              | Commander q MP (q = action − 149) |
| **200**              | **WAIT (ne rien faire)** |

### 8.1 Règles de validité

Avant de lancer une action de production, l’environnement vérifie :

- pour P1 / P2_step1 :
  - M1 doit être libre (`busy == 0`)
  - `stock_raw >= k`

- pour P2_step2 :
  - M2 doit être libre
  - `stock_p2_inter >= k`

Si une condition n’est pas satisfaite, l’action est automatiquement transformée en **WAIT**.  
Les actions hors `[0, 200]` sont invalides et lèvent une erreur.

---

## 9. Espace d’observation (13 variables)

L’état retourné à chaque step est un vecteur de dimension 13 :

| Index | Nom                       | Description                                             |
|-------|---------------------------|---------------------------------------------------------|
| 0     | `time`                    | minute courante (0 à 10080)                             |
| 1     | `m1_busy`                 | 1 si M1 travaille, 0 sinon                              |
| 2     | `m1_time_left`            | minutes restantes pour M1                               |
| 3     | `m2_busy`                 | 1 si M2 travaille                                       |
| 4     | `m2_time_left`            | minutes restantes pour M2                               |
| 5     | `stock_raw`               | stock de matière première                               |
| 6     | `stock_p1`                | stock P1                                                |
| 7     | `stock_p2_inter`          | stock intermédiaire P2                                  |
| 8     | `stock_p2`                | stock P2 fini                                           |
| 9     | `next_delivery_countdown` | temps avant la prochaine livraison de MP (0 si aucune)  |
| 10    | `backlog_p1`              | demande résiduelle P1 (carnet de commandes non servies) |
| 11    | `backlog_p2`              | demande résiduelle P2                                   |
| 12    | `q_total_en_route`        | quantité totale de MP commandée mais non encore livrée  |

Cet état est conçu pour être **Markovien** : il contient tout ce qui est nécessaire pour que l’agent prenne des décisions optimales sans mémoire supplémentaire.

---

## 10. Boucle d’un step

À chaque step (minute) :

1. L’agent propose une action `a_t`.
2. L’environnement décode l’action et vérifie les contraintes.
3. Éventuellement, un batch est lancé sur M1 ou M2, ou une commande de MP est enregistrée.
4. Le temps avance de 1 minute : `time += 1`.
5. Les machines M1 et M2 avancent (`tick()`), et les batches terminés mettent à jour les stocks.
6. Les livraisons de MP arrivant à cette minute sont ajoutées au stock.
7. Si `time % 60 == 0` : on met à jour le backlog avec une nouvelle demande horaire, puis on calcule les ventes et le revenu.
8. Si `time % 1440 == 0` : on applique le vol nocturne sur P1 et P2.
9. On calcule le reward total de la minute.
10. On renvoie `(observation, reward, done, truncated, info)`.

---

## 11. Fin d’épisode

- Un épisode se termine lorsque :

```text
time >= 10080
```

- Il n’y a pas de condition de fin alternative (pas de faillite explicite, etc.).

---

## 12. Résumé global

L’environnement Workshop final modélise :

- un atelier de production avec deux machines et plusieurs types de stocks,
- un marché avec demande jour/nuit,
- des ventes agrégées par heure,
- un carnet de commandes résiduel (backlog) visible,
- des commandes de MP avec délai et pipeline visible,
- un vol nocturne sur les produits finis,
- un espace d’actions discret simple mais riche,
- un vecteur d’état compact mais complet.

Cet environnement est **réaliste**, **multitâche** et **adapté à l’apprentissage par renforcement profond (DQN, PPO, etc.)**.
