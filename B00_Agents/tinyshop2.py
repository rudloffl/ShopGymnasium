from typing import Optional
import numpy as np
import gymnasium as gym
import simpy
import random


class ShopEnv(gym.Env):
    def __init__(self,
                 operatorcounts: int = 2,
                 machine_count: int = 2,
                 product_count: int = 2,
                 step: int = 1,  # Hour
                 duration_max=7  # Days
                 ):
        self.operatorcounts = operatorcounts
        self.product_count = product_count
        self.machine_count = machine_count

        self.forced_stop = np.zeros((self.product_count, self.machine_count))
        
        self.order_log = []
        self.sales_log = []
        self.cash_flow = []
        self.stealing_log = []
        self.stock_utilization = []
        self.maq_utilization = []

        self.prod_assignment = np.zeros((self.product_count, self.machine_count))
        # Prod A only needs machine 1 for 6 unit of time (in minutes)
        self.prod_assignment[0, 0] = 6
        self.prod_assignment[0, 1] = 0
        # Prod B needs machine 1 for 15 unit of time, and 20 for machine 2 (in minutes)
        self.prod_assignment[1, 0] = 15
        self.prod_assignment[1, 1] = 20
        self.duration_max = duration_max

        self.prod_dict = {'0': {'Name': 'A', 'Cost': 3},
                          '1': {'Name': 'B', 'Cost': 20},
                          }

        # Where products need to go - Intermediary stock
        self.to_stock_prod = np.zeros((self.product_count, self.machine_count))
        self.to_stock_prod[0, 0] = 0
        self.to_stock_prod[0, 1] = 0
        self.to_stock_prod[1, 0] = 1
        self.to_stock_prod[1, 1] = 0

        # Selling Stock
        self.to_sell_prod = np.zeros((self.product_count, self.machine_count))
        self.to_sell_prod[0, 0] = 1
        self.to_sell_prod[0, 1] = 0
        self.to_sell_prod[1, 0] = 0
        self.to_sell_prod[1, 1] = 1
        
        self.pending_raw = 0
        
        # OBSERVATION SPACE - Dict is supported for observations
        self.observation_space = gym.spaces.Dict({
            # Stocks 
            "stockraw_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "stockraw_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "stockint_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "stockint_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "stock2sell_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "stock2sell_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),

            # What is in progress and how many to do next
            'current_batch_remaining': gym.spaces.Box(low=0, high=100, 
                                                     shape=(self.product_count, self.machine_count), 
                                                     dtype=np.float32),
            'next_batch': gym.spaces.Box(low=0, high=100, 
                                        shape=(self.product_count, self.machine_count), 
                                        dtype=np.float32),
            'ranking_next': gym.spaces.Box(low=0, high=1, 
                                          shape=(self.product_count, self.machine_count), 
                                          dtype=np.float32),

            # How does the staffing look like
            "count_operators_busy": gym.spaces.Box(low=0, high=self.operatorcounts, 
                                                   shape=(1,), dtype=np.float32),
            "count_operators_free": gym.spaces.Box(low=0, high=self.operatorcounts, 
                                                   shape=(1,), dtype=np.float32),

            # Where is made what with the cycle time
            "prod_assignment": gym.spaces.Box(low=0, high=500, 
                                             shape=(self.product_count, self.machine_count), 
                                             dtype=np.float32),

            # Pending Reception
            "pending_reception": gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),

            # Time of day
            "timeday": gym.spaces.Box(low=0, high=23, shape=(1,), dtype=np.float32)
        })

        # ACTION SPACE - Flattened Box (PPO doesn't support Dict actions)
        # Calculate flattened action space dimensions
        self.action_dim_current_batch = self.product_count * self.machine_count
        self.action_dim_force = self.product_count * self.machine_count
        self.action_dim_next_batch = self.product_count * self.machine_count
        self.action_dim_ranking = self.product_count * self.machine_count
        self.action_dim_order = 1
        
        total_action_dim = (self.action_dim_current_batch + 
                           self.action_dim_force + 
                           self.action_dim_next_batch + 
                           self.action_dim_ranking + 
                           self.action_dim_order)
        
        # All actions normalized to 0-1 range
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(total_action_dim,), 
            dtype=np.float32
        )
        
        # Store indices for easier parsing
        self.action_indices = {
            'current_batch': (0, self.action_dim_current_batch),
            'force_current_batch': (self.action_dim_current_batch, 
                                    self.action_dim_current_batch + self.action_dim_force),
            'next_batch': (self.action_dim_current_batch + self.action_dim_force,
                          self.action_dim_current_batch + self.action_dim_force + self.action_dim_next_batch),
            'ranking_next': (self.action_dim_current_batch + self.action_dim_force + self.action_dim_next_batch,
                            self.action_dim_current_batch + self.action_dim_force + self.action_dim_next_batch + self.action_dim_ranking),
            'order_raw_prod': (self.action_dim_current_batch + self.action_dim_force + self.action_dim_next_batch + self.action_dim_ranking,
                              total_action_dim)
        }

        # State variables
        self.current_batch = np.zeros((self.product_count, self.machine_count))
        self.force_current_batch = np.zeros((self.product_count, self.machine_count))
        self.next_batch = np.zeros((self.product_count, self.machine_count))
        self.ranking_next = np.zeros((self.product_count, self.machine_count))

        # Unit of time for each step
        self.step_size = step  # Decisions are made every hour
        self.episode_end = self.duration_max * 24  # End after 7 days
        self.salesrewards = 0
        self.poormanagementpenality = 0

        self.prod_log = np.zeros((self.product_count, self.machine_count))
        self.sell_log = np.zeros(self.product_count)

        self._make_simpy_env()

    def _make_simpy_env(self):
        self.shopsim = simpy.Environment()
        
        self.operators = simpy.Resource(self.shopsim, capacity=self.operatorcounts)
        
        self.stockraw = simpy.Container(self.shopsim, capacity=100, init=random.randint(1, 75))
        
        self.stockint = simpy.FilterStore(self.shopsim, capacity=50)
        self.stocksell = simpy.FilterStore(self.shopsim, capacity=50)

        for machine_num in range(self.machine_count):
            setattr(self, f'machine_{machine_num}', simpy.Resource(self.shopsim, capacity=1))  

        self.now_sim = 0

        self.shopsim.process(self.steal_product_at_night())
        self.shopsim.process(self.sell_products())
        self.shopsim.process(self.get_operators_to_work())

    def _parse_action(self, action_flat):
        """
        Parse flattened action array into structured components.
        
        Args:
            action_flat: Flattened numpy array from the policy (values between 0 and 1)
        
        Returns:
            Dictionary with parsed action components
        """
        action = {}
        
        # Extract current_batch (scale to 0-100)
        start, end = self.action_indices['current_batch']
        action['current_batch'] = (action_flat[start:end] * 100).reshape(
            self.product_count, self.machine_count
        )
        
        # Extract force_current_batch (threshold at 0.5 to get binary)
        start, end = self.action_indices['force_current_batch']
        action['force_current_batch'] = (action_flat[start:end] > 0.5).astype(int).reshape(
            self.product_count, self.machine_count
        )
        
        # Extract next_batch (scale to 0-100)
        start, end = self.action_indices['next_batch']
        action['next_batch'] = (action_flat[start:end] * 100).reshape(
            self.product_count, self.machine_count
        )
        
        # Extract ranking_next (already 0-1)
        start, end = self.action_indices['ranking_next']
        action['ranking_next'] = action_flat[start:end].reshape(
            self.product_count, self.machine_count
        )
        
        # Extract order_raw_prod (scale to 0-100)
        start, end = self.action_indices['order_raw_prod']
        action['order_raw_prod'] = action_flat[start:end][0] * 100
        
        return action
    
    def _get_obs(self):
        """Return observations as numpy arrays matching observation_space."""
        return {
            # Convert all floats to numpy arrays with proper shape
            "stockraw_used": np.array([self.stockraw.level], dtype=np.float32),
            "stockraw_free": np.array([self.stockraw.capacity - self.stockraw.level], dtype=np.float32),
            "stockint_used": np.array([len(self.stockint.items)], dtype=np.float32),
            "stockint_free": np.array([self.stockint.capacity - len(self.stockint.items)], dtype=np.float32),
            "stock2sell_used": np.array([len(self.stocksell.items)], dtype=np.float32),
            "stock2sell_free": np.array([self.stocksell.capacity - len(self.stocksell.items)], dtype=np.float32),
            
            # These are already numpy arrays, ensure they're float32
            "current_batch_remaining": self.current_batch.astype(np.float32),
            "next_batch": self.next_batch.astype(np.float32),
            "ranking_next": self.ranking_next.astype(np.float32),
            
            # Convert operator counts to numpy arrays
            "count_operators_busy": np.array([self.operators.count], dtype=np.float32),
            "count_operators_free": np.array([self.operators.capacity - self.operators.count], dtype=np.float32),
            
            # Already numpy array, ensure float32
            "prod_assignment": self.prod_assignment.astype(np.float32),
            
            # Convert pending reception to numpy array
            "pending_reception": np.array([self.pending_raw], dtype=np.float32),
            
            # Convert time to numpy array
            "timeday": np.array([self.shopsim.now % 24], dtype=np.float32)
        }

    def _get_info(self):
        """Only for debugging purpose."""
        return {
            "production_log": self.prod_log,
            "sell_log": self.sell_log
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.order_log = []
        self.sales_log = []
        self.cash_flow = []
        self.stock_utilization = []
        self.stealing_log = []
        self.maq_utilization = []

        self._make_simpy_env()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action_flat):
        """
        Execute one step in the environment.
        
        Args:
            action_flat: Flattened action array from the policy
        """
        # Parse the flattened action
        action = self._parse_action(action_flat)
        
        reward = 0
        truncated = False
        terminated = True if self.shopsim.now >= int(self.duration_max * 24) else False
        
        # Extract action components
        current_batch = action['current_batch']
        force_current_batch = action['force_current_batch']
        next_batch = action['next_batch']
        ranking_next = action['ranking_next']
        order_raw_prod = action['order_raw_prod']
        
        # Does he want to enforce a new batch size:
        for i, j in np.ndindex(force_current_batch.shape):
            if force_current_batch[i, j] == 1:
                self.forced_stop[i, j] = 1
                self.current_batch[i, j] = int(current_batch[i, j])
                reward -= 5
        
        # We set what should be done next and with what urgency
        for i, j in np.ndindex(next_batch.shape):
            if (self.prod_assignment[i, j] == 0) and (next_batch[i, j] != 0):
                # Can't assign something when not planned to
                reward -= 50
                continue
            if self.prod_assignment[i, j] > 0:
                self.next_batch[i, j] = int(next_batch[i, j])
                self.ranking_next[i, j] = ranking_next[i, j]
        
        # We order raw product
        if order_raw_prod > 0:
            self.shopsim.process(self.order_raw_product(int(order_raw_prod)))
            reward = reward - int(order_raw_prod) * 1
        
        # Execute the Simpy
        self.now_sim += self.step_size
        self.shopsim.run(until=self.now_sim)
        
        reward += self.salesrewards
        reward -= self.poormanagementpenality
        
        self.salesrewards = 0
        self.poormanagementpenality = 0
        
        self.record_log_step()

        observation = self._get_obs()
        info = self._get_info()
        

        return observation, reward, terminated, truncated, info

    def order_raw_product(self, qty):
        qty = int(qty)
        self.pending_raw += qty
        self.order_log.append((self.shopsim.now, qty))
        yield self.shopsim.timeout(1)
        roominstock = min(self.stockraw.capacity - self.stockraw.level, qty)
        if (roominstock > 0) and (roominstock <= self.stockraw.capacity):
            yield self.stockraw.put(roominstock)
        self.pending_raw -= qty

    def steal_product_at_night(self, threshold_stock=10, day_hour=6, probability=.1):
        """At night for 12 hours each product not in the shop is subject to be stolen at a 10% chance per cycle time."""
        while True:
            if self.shopsim.now % day_hour > day_hour / 2:
                # Filter Stores
                stolen = 0
                for stock in (self.stockint, self.stocksell):
                    if len(stock.items) > threshold_stock:
                        at_risk = (np.random.random(len(stock.items) - threshold_stock) < probability).astype(int)
                        
                        for product_index in at_risk:
                            if product_index == 1:
                                stolen += 1
                                yield stock.get()
                if stolen != 0:
                    self.stealing_log.append((self.shopsim.now, 'products', -stolen))
                # Containers
                stolen = 0
                for stock in (self.stockraw,):
                    if stock.level > threshold_stock:
                        at_risk = (np.random.random(int(stock.level) - threshold_stock) < probability).astype(int)
                        for product_index in at_risk:
                            if product_index == 1:
                                stolen += 1
                                yield stock.get(1)
                if stolen != 0:
                    self.stealing_log.append((self.shopsim.now, 'rawproducts', -stolen))
            yield self.shopsim.timeout(self.step_size)

    def make_products(self, machine_j=0, prod_i=0, patience=60):
        cycletime = self.prod_assignment[prod_i, machine_j]  # Minutes
        machine = getattr(self, f'machine_{machine_j}')

        # Determine stock sources
        stockin = self.stockraw
        filterstore = False
        if prod_i == 1 and machine_j == 1:
            stockin = self.stockint
            filterstore = True
        
        if self.to_stock_prod[prod_i, machine_j] == 1:
            stockout = self.stockint
        if self.to_sell_prod[prod_i, machine_j] == 1:
            stockout = self.stocksell
        
        with self.operators.request() as op_req, machine.request() as machine_req:
            # Wait for both resources to be available
            yield op_req
            yield machine_req
            start = self.shopsim.now
            while self.current_batch[prod_i, machine_j] != 0:
                if self.forced_stop[prod_i, machine_j] == 1:
                    # We stop what we were doing immediately
                    self.forced_stop[prod_i, machine_j] = 0
                    break
                if filterstore:
                    event = stockin.get(lambda x: x == str(prod_i))
                else:
                    event = stockin.get(1)

                timeout_event = self.shopsim.timeout(patience / 60)

                result = yield event | timeout_event

                if event in result:
                    self.current_batch[prod_i, machine_j] -= 1
                    self.prod_log[prod_i, machine_j] += 1
                    self.salesrewards += .1
                    yield stockout.put(str(prod_i))
                    yield self.shopsim.timeout(cycletime / 60)
                else:
                    break
            end = self.shopsim.now
            self.maq_utilization.append((start, end, prod_i, f'machine_{machine_j}'))

    def get_operators_to_work(self):
        while True:
            if (self.operators.count < self.operators.capacity) and (self.next_batch.sum() != 0):
                max_idx = np.argmax(self.ranking_next)
                i, j = np.unravel_index(max_idx, self.ranking_next.shape)
                self.ranking_next[i, j] = 0
                self.current_batch[i, j] = self.next_batch[i, j]
                self.next_batch[i, j] = 0
                machine = getattr(self, f'machine_{j}')
                if machine.count != 0:
                    # Only pass the task if the machine is available
                    self.ranking_next[i, j] = 0
                    self.current_batch[i, j] = self.next_batch[i, j]
                    self.next_batch[i, j] = 0
                if (self.current_batch[i, j] > 0) and (machine.count == 0) and (self.prod_assignment[i, j] != 0):
                    self.shopsim.process(self.make_products(prod_i=i, machine_j=j))
                
            yield self.shopsim.timeout(self.step_size / 60)

    def sell_products(self, freq=1):
        while True:
            if self.shopsim.now % freq == 0:
                stepreward = 0
                allprods = [x for x in self.stocksell.items]
                for prod in allprods:
                    sold = yield self.stocksell.get(lambda x: x == prod)
                    self.sell_log[int(prod)] += 1
                    self.salesrewards += self.prod_dict.get(prod)['Cost']
                    stepreward += self.prod_dict.get(prod)['Cost']

                self.sales_log.append((self.shopsim.now, stepreward))
            yield self.shopsim.timeout(self.step_size)

    def record_log_step(self):

        self.stock_utilization.append((self.shopsim.now, 'Stockint', len(self.stockint.items)))
        self.stock_utilization.append((self.shopsim.now, 'stocksell', len(self.stocksell.items)))
        self.stock_utilization.append((self.shopsim.now, 'stockraw', self.stockraw.level))

    def render(self):
        return {'order_log': self.order_log,
                # 'cash_flow': self.cash_flow,
                'sales_log': self.sales_log,
                'stock_utilization': self.stock_utilization,
                'stealing_log': self.stealing_log,
                'maq_utilization': self.maq_utilization}

if __name__ == "__main__":
    env = ShopEnv(duration_max=1/24)
    observation, info = env.reset()
    print("Initial observation keys:", observation.keys())
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space: {env.action_space}")
    
    terminated = False
    rewards = []
    step_count = 0
    
    pc = env.product_count
    mc = env.machine_count
    
    print("\n" + "="*60)
    print("MANUAL CONTROL MODE - Enter actions for the shop")
    print("="*60)
    
    while not terminated:
        print(f"\n{'='*60}")
        print(f"STEP {step_count + 1}")
        print(f"{'='*60}")
        print(f"Current Time: {observation['timeday'][0]:.1f} hours")
        print(f"Raw Stock: {observation['stockraw_used'][0]:.0f}/{observation['stockraw_used'][0] + observation['stockraw_free'][0]:.0f}")
        print(f"Intermediate Stock: {observation['stockint_used'][0]:.0f}/{observation['stockint_used'][0] + observation['stockint_free'][0]:.0f}")
        print(f"Sell Stock: {observation['stock2sell_used'][0]:.0f}/{observation['stock2sell_used'][0] + observation['stock2sell_free'][0]:.0f}")
        print(f"Operators: {observation['count_operators_busy'][0]:.0f} busy, {observation['count_operators_free'][0]:.0f} free")
        
        # Raw product order (0-100, will be scaled from 0-1)
        rawproductorder = float(input("\nRaw Product Order (0-100): "))
        rawproductorder_normalized = rawproductorder / 100.0
        
        # Current batch
        print(f"\nCurrent Batch ({pc}x{mc} values, 0-100):")
        current_batch = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                if env.prod_assignment[i, j] > 0:
                    val = float(input(f"  current_batch[Prod {i}, Machine {j}] (0-100): "))
                    current_batch[i, j] = val / 100.0  # Normalize to 0-1
        
        # Force current batch
        print(f"\nForce Current Batch ({pc}x{mc} values, 0 or 1):")
        force_current_batch = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                if env.prod_assignment[i, j] > 0:
                    val = float(input(f"  force_current_batch[Prod {i}, Machine {j}] (0 or 1): "))
                    force_current_batch[i, j] = val  # Already 0 or 1
        
        # Next batch
        print(f"\nNext Batch ({pc}x{mc} values, 0-100):")
        next_batch = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                if env.prod_assignment[i, j] > 0:
                    val = float(input(f"  next_batch[Prod {i}, Machine {j}] (0-100): "))
                    next_batch[i, j] = val / 100.0  # Normalize to 0-1
        
        # Ranking next
        print(f"\nRanking Next ({pc}x{mc} values, 0-1):")
        ranking_next = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                if env.prod_assignment[i, j] > 0:
                    val = float(input(f"  ranking_next[Prod {i}, Machine {j}] (0-1): "))
                    ranking_next[i, j] = val  # Already 0-1
        
        # Flatten the action into a single array
        action_flat = np.concatenate([
            current_batch.flatten(),
            force_current_batch.flatten(),
            next_batch.flatten(),
            ranking_next.flatten(),
            np.array([rawproductorder_normalized])
        ]).astype(np.float32)
        
        # Take the step
        observation, reward, terminated, truncated, info = env.step(action_flat)
        rewards.append(reward)
        step_count += 1
        
        print(f"\n{'='*60}")
        print(f"STEP {step_count} RESULTS")
        print(f"{'='*60}")
        print(f"Reward: {reward:.2f}")
        print(f"Cumulative Reward: {sum(rewards):.2f}")
        print(f"Production Log:\n{info['production_log']}")
        print(f"Sell Log: {info['sell_log']}")
        
        if truncated:
            terminated = True
    
    print(f"\n{'='*60}")
    print("EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f'Final Total Reward: {sum(rewards):.2f}')
    print(f'Total Steps: {step_count}')
    print(f'Average Reward per Step: {sum(rewards)/step_count:.2f}')
    print(f'Final Production Log:\n{info["production_log"]}')
    print(f'Final Sell Log: {info["sell_log"]}')

    print(env.render())