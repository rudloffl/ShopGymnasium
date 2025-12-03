# Workshop Environment Specification (DQN-Ready Version)

This document defines **all rules** of the industrial environment used
for the Deep Q-Network (DQN) model.\
It consolidates every design decision taken so far and serves as the
**official reference specification** for implementing, training, and
validating the future DQN agent.

The environment models a realistic industrial workshop with autonomous
machines, multistage production, demand-driven sales, inventory
constraints, delays, and time-based events.

------------------------------------------------------------------------

# 1. Global Time Model

## 1.1 Fundamental Time Units

-   **1 unit of time = 1 minute**
-   This is the universal temporal reference for:
    -   machine cycles\
    -   sales\
    -   lead times\
    -   night events\
    -   step duration accumulation

## 1.2 Episode Duration

-   **1 episode = 1440 minutes** (a full 24-hour industrial day)

-   The episode terminates when:

        time >= 1440

## 1.3 Step Definition

-   **1 step = 1 action chosen by the RL agent.**
-   A step does *not* correspond to 1 minute.
-   Each action may advance time by **a variable number of minutes**
    depending on:
    -   batch size\
    -   process duration\
    -   machine availability

Example: launching a batch of 10 P2 units (total duration 250 minutes)
advances time by 250 minutes.

------------------------------------------------------------------------

# 2. Machines and Parallelism

## 2.1 Machines Overview

The workshop contains two autonomous machines:

  Machine   Uses                       Purpose
  --------- -------------------------- ------------------------
  **M1**    P1 production, P2 step 1   Processes raw material
  **M2**    P2 step 2                  Finalizes P2

## 2.2 Autonomy and Parallel Operation

-   Machines **run independently** once a job is launched.
-   A job has a **remaining processing time** managed by the
    environment.
-   While a machine is busy, the RL agent **can still act** and launch:
    -   another job on a free machine\
    -   a material order\
    -   a different production batch\
    -   administrative actions\
-   The agent **cannot interrupt** a job already in progress.

This creates a **true multitasking environment**, closer to a
SimPy-style industrial simulation.

## 2.3 Machine States

Each machine has: - `busy : bool` - `time_remaining : int` (in
minutes) - `current_batch_size : int` (optional, for tracking)

Machines update their states **every time the global clock moves
forward**.

------------------------------------------------------------------------

# 3. Products and Production Rules

## 3.1 Product Definitions

### P1 (simple product)

-   Produced on **M1**
-   Requires **1 unit of raw material per unit**
-   Production time per unit: **3 minutes**
-   Value per unit sold: **2**

### P2 (complex product, 2 steps)

**Step 1**\
- Machine: M1\
- Converts raw material → semi-finished P2\
- Time per unit: **10 minutes**

**Step 2**\
- Machine: M2\
- Converts semi-finished P2 → finished P2\
- Time per unit: **15 minutes**

-   Total time per P2: **25 minutes**\
-   Value per unit sold: **20**

## 3.2 Batch Production

The agent may choose a batch size `k` for: - P1 production\
- P2 step 1\
- P2 step 2

Production durations are:

  Action                  Duration
  ----------------------- ------------------
  Produce `k` P1          `3 * k` minutes
  Produce `k` P2 step 1   `10 * k` minutes
  Produce `k` P2 step 2   `15 * k` minutes

Batch sizes typically range from 1 to 10 units.

------------------------------------------------------------------------

# 4. Inventory Rules

The environment tracks:

-   `stock_raw` (raw material)
-   `stock_p1` (finished P1)
-   `stock_p2_inter` (semi-finished P2)
-   `stock_p2` (finished P2)

### All stocks are capped at a maximum of **10 units** for stability.

------------------------------------------------------------------------

# 5. Material Ordering and Lead Time

## 5.1 Ordering Rule

-   The agent may place a material order.
-   A typical order quantity is **5 units** of raw material.
-   Orders are appended to a **pending deliveries queue**.

## 5.2 Lead Time

-   Delivery arrives after a deterministic delay, e.g.:

        lead_time = 120 minutes

-   When:

        delivery_time <= current_time

    stock is increased accordingly.

Raw material stock is capped (max 10).

------------------------------------------------------------------------

# 6. Sales Model (Option 1: Demand-Driven Sales)

This is the **realistic sales simulation** selected for the DQN
environment.

## 6.1 Demand Streams

At fixed intervals (e.g., every minute or every 10 minutes), the
environment generates **external customer demand**:

    demand_p1(t)
    demand_p2(t)

Demand may be: - deterministic\
- random (Poisson, Normal, Uniform)\
- hybrid (baseline + noise)

## 6.2 Automatic Sales Processing

The RL agent **does NOT decide when to sell**.

Sales occur automatically as:

    sales_p1 = min(stock_p1, demand_p1)
    sales_p2 = min(stock_p2, demand_p2)

Inventory is reduced accordingly.

## 6.3 Revenue

The agent receives:

    reward += 2 * sales_p1 + 20 * sales_p2

## 6.4 Lost Demand

Unmet demand is **lost** and does not carry over.

This models real industrial sales behavior: - customers buy what is
available\
- shortages reduce potential revenue\
- production must match fluctuating demand

------------------------------------------------------------------------

# 7. Night Events

## 7.1 Timing

At the end of each day:

    time >= 1440

A **night event** occurs before the episode ends.

## 7.2 Theft

As specified by Laurent:

-   **10 %** of P1 and P2 stock is stolen:

```{=html}
<!-- -->
```
    stock_p1 = floor(stock_p1 * 0.9)
    stock_p2 = floor(stock_p2 * 0.9)

Raw material and semi-finished P2 are **not** affected.

------------------------------------------------------------------------

# 8. Rewards

The total reward in one step includes:

### 1. Sales Revenue

From section 6:

    + 2 * sales_p1 + 20 * sales_p2

### 2. Holding Costs

Optional but realistic:

    - holding_cost * (stock_p1 + stock_p2_inter + stock_p2)

### 3. Production Costs

Optional depending on realism:

    - cost_p1 * produced_p1
    - cost_p2 * produced_p2

### 4. Penalties

Possible penalties: - machine idle time\
- stockouts\
- unfinished jobs at end of day

For now: **none are included**, but DQN can support them.

------------------------------------------------------------------------

# 9. Actions Summary

The action set includes:

1.  **Launch P1 batch (k units)**\
2.  **Launch P2 Step 1 batch (k units)**\
3.  **Launch P2 Step 2 batch (k units)**\
4.  **Place material order**\
5.  **Do nothing / wait**

All actions: - are instantaneous in decision time\
- may advance the clock depending on job duration or 1 minute for
administrative actions\
- may be forbidden if the target machine is busy

------------------------------------------------------------------------

# 10. State Representation (DQN Input)

The state given to the DQN includes:

-   Current time in minutes (scaled)\
-   Machine M1:
    -   busy flag
    -   time remaining
-   Machine M2:
    -   busy flag
    -   time remaining
-   Inventory:
    -   stock_raw\
    -   stock_p1\
    -   stock_p2_inter\
    -   stock_p2\
-   Pending deliveries (time until arrival)
-   Optional: last demand values (for partially observable demand)

This state is **too large** for tabular Q-learning → hence the move to
**DQN**.

------------------------------------------------------------------------

# 11. Transition Dynamics

Each step:

1.  Agent chooses an action\
2.  Environment validates feasibility\
3.  If production begins → machine becomes busy\
4.  Time advances by:
    -   production duration, or\
    -   1 minute for administrative actions\
5.  Machine timers decrease\
6.  Deliveries may arrive\
7.  Sales occur automatically\
8.  Night theft may occur\
9.  New state is returned

------------------------------------------------------------------------

# 12. Episode Termination

An episode ends when:

    time >= 1440

Night event triggers once before termination.

------------------------------------------------------------------------

# End of Environment Specification
