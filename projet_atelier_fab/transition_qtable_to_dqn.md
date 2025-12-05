# Transition from Q-Table to DQN: Rationale and Technical Justification

## 1. Introduction

This document explains **why we are transitioning from a tabular
Q-Learning approach to a Deep Q-Network (DQN)** for modeling our
industrial workshop simulation. It summarizes the reasoning, technical
constraints, and limits of the Q-table approach when modeling a
realistic environment based on Laurent's specifications.

------------------------------------------------------------------------

## 2. Initial Approach: Tabular Q-Learning

We started with a classical **Q-table** approach:

-   **State space**:\
    (stock_raw, stock_P1, stock_P2_inter, stock_P2), each in \[0..10\]\
    → 11⁴ = **14,641 states**

-   **Action space**:\
    32 actions (produce P1 batches, produce P2 batches, order material,
    wait)

-   **Q-table size**:\
    14,641 × 32 = **468,512 Q-values**

This size is manageable: - low memory usage, - straightforward
implementation, - convergence possible within \~100k--300k steps, -
useful for validating early concepts.

------------------------------------------------------------------------

## 3. Added Realism in the Environment

To remain faithful to Laurent's intended workshop, we progressively
introduced realistic industrial features:

### ✔ Time modeled in minutes

1 unit = 1 minute\
1 episode = 1440 minutes (1 working day)\
Batch durations:\
- P1: 3 minutes × k\
- P2 step 1: 10 minutes × k\
- P2 step 2: 15 minutes × k

### ✔ Nightly stock loss

Every night (each 1440 minutes), 10% of external stock is stolen.

### ✔ Supply lead time

Material orders arrive after a delay (e.g., 120 minutes).

### ✔ Multistage production

P1 is simple; P2 requires M1 then M2.

These modifications made the simulation more realistic **without
increasing Q-table size**, because the state representation remained
compact.

------------------------------------------------------------------------

## 4. The Fundamental Limitation: No Parallelism

As realism increased, a major limitation appeared.

### In a real workshop:

-   Machines run **in parallel**.
-   An operator launches a job and moves on.
-   M1 and M2 can be busy simultaneously.
-   Each machine has its own **remaining processing time**.
-   The agent must make decisions **while machines are working**.

### In the tabular model:

-   One action = one entire batch.
-   Agent cannot act while production runs.
-   No machine timers.
-   No machine concurrency.
-   No operator behavior modeling.

To fix this, we must add machine states:

-   M1 busy/free + remaining time\
-   M2 busy/free + remaining time\
-   possibly queues or operator states

But adding these to the state **explodes** the Q-table.

------------------------------------------------------------------------

## 5. Why Tabular Q-Learning Cannot Scale

Adding a machine timer (0..150 minutes):

14,641 × 151 = **2.2 million states**

Two machines:

14,641 × 151 × 151 ≈ **334 million states**

This is infeasible: - gigantic memory requirement, - impossible
exploration, - unrealistic training time, - no generalization.

This is the classical **curse of dimensionality**.

------------------------------------------------------------------------

## 6. The Solution: Deep Q-Network (DQN)

DQN replaces the Q-table with a **neural network approximation** of
Q(s,a).

### Benefits:

### ✔ Handles large/continuous state spaces

Machine timers, busy flags, queues, lead times, stock ranges...

### ✔ Allows parallel machine modeling

Agent can make decisions every minute while machines run.

### ✔ Supports realistic features

-   operator availability\
-   maintenance\
-   machine failures\
-   stochastic processing times\
-   real-time scheduling

### ✔ More expressive and scalable

DQN generalizes and handles unseen states, unlike Q-tables.

------------------------------------------------------------------------

## 7. Final Decision

We switch to DQN because:

### **1. Q-tables cannot model concurrent machine operations.**

### **2. Realistic industrial behavior requires machine timers and parallel processes.**

### **3. Adding these to a Q-table explodes its size to hundreds of millions of states.**

### **4. DQN naturally handles high-dimensional states.**

### **5. It matches Laurent's original intent: a SimPy-like multi-process workshop.**

------------------------------------------------------------------------

## 8. Conclusion

The Q-table approach successfully validated our initial workshop logic.\
But to achieve industrial realism --- parallel machines, timers, delays,
operator-free machine cycles --- the tabular method is fundamentally
insufficient.

**DQN is the necessary next step.**

It preserves what we've built, and unlocks the ability to develop a
realistic industrial simulation matching Laurent's vision.

------------------------------------------------------------------------

*End of report.*
