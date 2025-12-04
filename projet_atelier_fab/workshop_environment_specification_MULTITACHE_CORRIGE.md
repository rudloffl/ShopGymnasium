# Spécification de l’Environnement Workshop – Version DQN  
### ***Mise à jour : 1 step = 1 minute, multitâche complet***

Ce document constitue la **spécification officielle** de ton environnement industriel RL.  
Il est maintenant **corrigé** pour intégrer clairement la règle essentielle suivante :

# **1 step = 1 minute, même lorsqu'une machine exécute un batch long.**  
→ L’agent peut prendre **une nouvelle décision chaque minute**, même si un batch dure 500 minutes.  
→ Cela permet le **multitâche**, comme dans une vraie usine.

---

# 1. Structure temporelle

- **1 minute = 1 step**  
- **1 épisode = 1440 steps = 1440 minutes = 1 journée**
- À chaque step :
  - l’agent choisit une action,
  - le temps avance **d’une minute**, **jamais plus**,
  - les machines continuent leur travail en arrière‑plan.

**Très important :**  
Le temps ne saute jamais directement de “maintenant” à “fin du batch”.  
Les machines avancent leur `time_left` d’une minute à chaque step.

---

# 2. Machines et multitâche (corrigé)

## Machines
- **M1** : produit P1 et P2 Étape 1  
- **M2** : produit P2 Étape 2  

## Multitâche réel
Lorsqu’un batch est lancé :

- La machine devient `busy = 1`.
- Un compteur est initialisé :
  ```
  m1_time_left = durée_batch
  ```
- À chaque step :
  ```
  m1_time_left -= 1
  m2_time_left -= 1
  ```

### Et surtout :
**L’agent peut agir pendant que les machines tournent.**

Il peut :
- commander de la MP,
- lancer une production sur l’autre machine,
- attendre,
- laisser faire les ventes automatiques,
- optimiser ses prises de décision.

Une machine se libère automatiquement quand :

```
time_left <= 0
```

Au moment de la libération :
- P1 produit → `stock_p1 += k`
- P2 étape 1 → `stock_p2_inter += k`
- P2 étape 2 → `stock_p2 += k`

---

# 3. Production

### Produit P1
- 1 MP → 1 P1  
- Machine : M1  
- Durée : **3 × k minutes**  
- Fonctionne en tâche de fond minute après minute  

### Produit P2
#### Étape 1 — M1 :
- 1 MP → 1 P2_inter  
- Durée : **10 × k minutes**

#### Étape 2 — M2 :
- 1 P2_inter → 1 P2  
- Durée : **15 × k minutes**

### Taille des batchs
**k de 1 à 50 unités.**

---

# 4. Stocks

- `stock_raw` (MP)  
- `stock_p1`  
- `stock_p2_inter`  
- `stock_p2`  
- Capacité **50 unités** chacun.

---

# 5. Commande de MP (avec coût)

- L’agent peut commander **1 à 50 unités**.  1 MP coute 1 donc la commande de q MP coute q
- La livraison arrive après un délai fixe (120 min).  
- Lors de la commande :
  ```
  reward -= q
  ```
- Lors de la livraison :
  ```
  stock_raw = min(stock_raw + q, 50)
  ```

---

# 6. Demande & ventes automatiques

Chaque minute :
- une demande est générée,
- les ventes se font automatiquement :

```
ventes_p1 = min(stock_p1, demande_p1)
ventes_p2 = min(stock_p2, demande_p2)
```

Revenu ajouté :
```
reward += 2 * ventes_p1 + 20 * ventes_p2
```

---

# 7. Vol nocturne

À `time >= 1440` :
```
stock_p1 = floor(stock_p1 * 0.9)
stock_p2 = floor(stock_p2 * 0.9)
```

---

# 8. Récompense complète

À chaque minute (step), le reward peut inclure :

1. **+ revenus des ventes**
2. **– coût d'achat MP (q)**
3. (optionnel) coûts de stockage
4. (optionnel) pénalités de rupture ou d'inactivité

---

# 9. Actions

201 actions :

- **0–49** : produire P1 (k = action + 1)  
- **50–99** : produire P2_step1 (k = action − 49)  
- **100–149** : produire P2_step2 (k = action − 99)  
- **150–199** : commander MP (q = action − 149)  
- **200** : attendre

---

# 10. Observation

12 variables :
- time  
- état & timers M1/M2  
- 4 stocks  
- prochaine livraison  
- demande minute courante  

---

# 11. Transition d’un step (corrigé)

Chaque step fait **toujours** :

1. Décodage de l’action  
2. Lancement éventuel d’un batch  
3. **Avancement du temps = 1 minute**  
4. Décrément des compteurs machines  
5. Libération automatique + ajout au stock  
6. Livraisons éventuelles  
7. Demande & ventes  
8. Application du coût d’achat  
9. Vol si fin de journée  
10. Retour du nouvel état + reward + done

---

# Résumé

- **Multitâche complet** : les machines tournent pendant que l’agent agit  
- **1 step = 1 minute** (jamais plus)  
- Production ajoutée seulement lorsque `time_left <= 0`  
- Spécification désormais **cohérente, réaliste et DQN‑ready**

