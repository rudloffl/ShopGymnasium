
# Modélisation 4 — Gestion d’atelier avec Q-Learning et Production en Lots

Ce notebook implémente une version avancée d’un atelier industriel simulé, entraîné par **Q-Learning**, avec :

- production en lots de Produit 1 (P1) ou Produit 2 (P2),
- contraintes réelles de stock et capacité,
- pénalités, coûts et durées,
- décision optimale dépendante de l’état `(stock_raw, stock_sell)`.

Il s’agit de la quatrième itération du modèle, après les versions introduisant progressivement la gestion du stock, les coûts, et la prise en compte du temps.

---

##  1. Règles de l’environnement

L’atelier fonctionne sous deux stocks :

- **stock_raw** : matière première (0 à 10)
- **stock_sell** : produits finis (0 à 10)

L’agent peut effectuer **22 actions** :

| Action | Description                                  |
|--------|----------------------------------------------|
| 0      | Attendre                                     |
| 1..10  | Produire k unités de Produit 1 (P1)          |
| 11..20 | Produire k unités de Produit 2 (P2)          |
| 21     | Commander +5 unités de matière première (MP) |

**Les bornes sont strictes** : les deux stocks restent dans `[0, 10]`.

---

## 2. Coûts et récompenses

###  Produit 1 (P1)
- Coût MP : **1 unité de matière première**
- Marge : **+2 par unité produite**
- Durée : **1 unité de temps par produit**

###  Produit 2 (P2)
- Coût MP : **2 unités de MP**
- Marge : **+20 par unité produite**
- Durée : **3 unités de temps par produit**

###  Commande de MP (+5 MP)
- Récompense : **–5**
- Durée : **1 unité de temps**
- Règle historique du modèle : **immuable**

###  Attendre
- Si stock_raw = 0 → **–1**
- Sinon → 0
- Durée : **1**

###  Coût de stockage
Après chaque action :

```
reward -= 0.5 * stock_sell
```

---

##  3. Prise en compte du temps

Le temps influence la récompense **via la durée des actions** :

| Action      | Durée |
|-------------|-------|
| Attendre    | 1     |
| Commander   | 1     |
| Produire P1 | k     |
| Produire P2 | 3k    |

L’épisode dure **50 unités de temps maximum**.

Conséquences :

- P2 rapporte beaucoup, mais consomme plus de temps.  
- À stock_sell élevé, produire peut devenir coûteux (pénalité).  
- Dans certains états, **Attendre** devient optimal.

---

## 4. Entraînement par Q-Learning

Hyperparamètres :

- `alpha = 0.1`
- `gamma = 0.95`
- `epsilon_decay = 0.995`
- `n_actions = 22`
- Q-table : `Q[stock_raw][stock_sell][action]`

---

##  5. Politique optimale (résultat final)

Issue de la cellule 7 du notebook.

###  Lecture :
- P2 est massivement privilégié quand `stock_sell` est bas  
- P1 devient un régulateur quand `stock_sell` augmente  
- Commander est quasiment exclusivement choisi quand `stock_raw = 0`  
- Attendre apparaît dans les zones dangereuses (stock_sell élevé)

---

##  6. Synthèse métier finale

###  1. Politique de commande
L’agent commande **uniquement** lorsque `stock_raw = 0`.  
→ Politique rationnelle et réaliste.

###  2. Produit 2 (P2) : moteur de profit
Lorsque `stock_sell ≤ 3` :

- P2 domine largement  
- Longs lots : P2-2, P2-3, P2-4 selon MP  
- L’agent maximise la rentabilité en début de cycle

*Zone “profit massif”*

###  3. Produit 1 (P1) : régulation fine
Lorsque `stock_sell ≥ 4` :

- P1 prend le relais  
- Production en petits ou moyens lots  
- Permet d’éviter une explosion du coût de stockage

*Zone “pilotage fin”*

###  4. Attente
L’action **Attendre** apparaît quand :

- `stock_sell` est élevé,
- la production augmenterait trop la pénalité,
- ou la durée restante est trop courte (risque de dépasser 50 steps).

*Zone “prudence / stabilité”*

###  5. Résumé stratégique global

- **P2 dès que possible** en début de cycle  
- **P1 pour contrôler la pénalité** quand stock_sell augmente  
- **Commander exclusivement en cas de pénurie de MP**  
- **Attendre** pour éviter d’aggraver le surstock  

---

##  7. Fichier source du notebook

Le code Python complet utilisé dans cette modélisation est disponible dans :

- `04_modelisation4.ipynb`

