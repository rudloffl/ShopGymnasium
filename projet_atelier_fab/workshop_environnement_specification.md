# Spécification de l’Environnement Workshop – Version DQN 

Ce document constitue la **spécification officielle** de l’environnement industriel Workshop utilisé pour l’apprentissage par renforcement (DQN, PPO, etc.).

Il intègre toutes les modifications récentes :

- 1 step = **1 minute**
- **épisode = 7 jours** (10080 minutes)
- machines véritablement **multitâches** (batches asynchrones)
- **demande = backlog résiduel** (carnet de commandes non satisfaites)
- **ventes toutes les 15 minutes** 
- demande **jour / nuit**
- production de P2 en **deux étapes**
- **commandes de MP** avec délai aléatoire autour de 120 minutes (120 ± 2 minutes, jitter uniforme)
- **vol nocturne**
- action **WAIT = 200**
- observation Markovienne (13 variables)

---

## 1. Structure temporelle

- 1 step = **1 minute**
- 1 journée = 1440 minutes
- 1 épisode = 7 jours = **10080 steps**

À chaque step, l’agent choisit une action. Les machines avancent en parallèle.

---

## 2. Ressources et produits

### 2.1 Types de stocks

- `stock_raw` : matière première
- `stock_p1` : produit fini P1
- `stock_p2_inter` : intermédiaire P2
- `stock_p2` : produit fini P2

Capacité logique : 50 unités par stock.

---

## 3. Machines

### 3.1 Machine M1

Transforme la MP en P1 ou P2_inter.

- **P1** : 1 MP → 1 P1, durée = `3 × k`
- **P2_step1** : 1 MP → 1 P2_inter, durée = `10 × k`

État interne :  
`busy`, `time_left`, `batch_type`, `batch_k`.

### 3.2 Machine M2

Transforme P2_inter en P2.

- **P2_step2** : 1 P2_inter → 1 P2, durée = `15 × k`

État interne similaire à M1.

### 3.3 Avancement multitâche

Chaque minute, les machines décrémentent `time_left`. Lorsqu’un batch termine :

- M1 ajoute P1 ou P2_inter
- M2 ajoute P2

---

## 4. Commandes de MP

Actions **150 à 199** :

```
q = action − 149
reward -= q
livraison dans 120 minutes
livraison dans un délai aléatoire autour de 120 minutes (entre 118 et 122 minutes, i.e. 120 ± 2)

File FIFO des commandes.

Livraison ajoute à `stock_raw`.

---

## 5. Demande, backlog et ventes (60 minutes)

### 5.1 Définition du backlog

`demande_p1` et `demande_p2` = carnet de commandes non servies.

### 5.2 Fréquence des ventes  
**Une vente a lieu toutes les 15 minutes** :

```
time > 0 and time % 15 == 0
```

Aucune autre vente dans l’intervalle.

### 5.3 Demande jour/nuit

- Jour : 8h → 20h
- Nuit : reste

Génération :

```
new_d1 ~ Poisson(lambda_p1 × 60)
new_d2 ~ Poisson(lambda_p2 × 60)
```

### 5.4 Ventes (backlog + stock)

À chaque heure pleine :

```
demande_p1 += new_d1
demande_p2 += new_d2

ventes_p1 = min(stock_p1, demande_p1)
ventes_p2 = min(stock_p2, demande_p2)

stock_p1 -= ventes_p1
stock_p2 -= ventes_p2

demande_p1 -= ventes_p1
demande_p2 -= ventes_p2

reward += 2 * ventes_p1 + 20 * ventes_p2
```

---

## 6. Vol nocturne

Toutes les 1440 minutes :
Chaque jour, 5 minutes avant minuit (quand `time % 1440 == 1435`, après l'incrément de `time` dans `step()`):
```
stock_p1 = floor(stock_p1 * 0.9)
stock_p2 = floor(stock_p2 * 0.9)
```

---

## 7. Reward

- ventes : +2/P1 et +20/P2
- commande MP : coût `−q`
- WAIT : -0.2/action
- action impossible : -1
- lancement de production : +0.5*k pour P1, +5*k pour P2STEP1, +15*k pour P2STEP2

---

## 8. Espace d’actions (0–200)

| Actions | Signification |
|--------|---------------|
| 0–49 | Produire P1 (k = action + 1) |
| 50–99 | Produire P2_step1 |
| 100–149 | Produire P2_step2 |
| 150–199 | Commander MP |
| 200 | WAIT |

Si non réalisable → WAIT.

---

## 9. Observation (13 variables)

```
[ time,
  m1_busy, m1_time_left,
  m2_busy, m2_time_left,
  stock_raw, stock_p1, stock_p2_inter, stock_p2,
  next_delivery_countdown,
  backlog_p1, backlog_p2,
  q_total_en_route ]
```

---

## 10. Step

action → tick machines → livraisons → ventes (+reward) → time += 1 → vol → observation.

---

## 11. Fin d’épisode

```
time >= 10080
```

---

## 12. Résumé

L’environnement modélise :

- production multitâche,
- backlog cumulatif,
- **ventes toutes les 15 minutes**,
- carnet de commandes dynamique toutes les 15 minutes, juste avant les ventes,
- commandes MP avec délai,
- commandes MP avec délai aléatoire autour de 120 minutes,
- actions discrètes,
- état Markovien compact.