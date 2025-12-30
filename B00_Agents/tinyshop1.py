from typing import Optional
import numpy as np

import gymnasium as gym
import simpy
from collections import namedtuple
gym.__version__

import gymnasium as gym
import simpy
import numpy as np

import random


Action = namedtuple('Action', [
    'current_batch',
    'force_current_batch', 
    'next_batch',
    'ranking_next',
    'order_raw_prod'
])

class ShopEnv(gym.Env):
    def __init__(self,
                 operatorcounts: int = 2,
                 machine_count: int = 2,
                 product_count: int = 2,
                 step: int = 1, #Hour
                 duration_max = 7 # Days
                ):
        self.operatorcounts = operatorcounts
        self.product_count = product_count
        self.machine_count = machine_count

        self.forced_stop = np.zeros((self.product_count, self.machine_count))
        
        self.prod_assignment = np.zeros((self.product_count, self.machine_count))
        # Prod A only needs machine 1 for 3 unit of time (in minutes)
        self.prod_assignment[0, 0] = 6
        self.prod_assignment[0, 1] = 0
        # Prod B needs machine 1 for 10 unit of time, and 15 for machine 2 (in minutes)
        self.prod_assignment[1, 0] = 15
        self.prod_assignment[1, 1] = 20
        self.duration_max = duration_max

        self.prod_dict = {'0': {'Name': 'A', 'Cost': 3},
                          '1': {'Name': 'B', 'Cost': 20},
                          }

        # Where products need to go
        # Intermediary stock
        self.to_stock_prod = np.zeros((self.product_count, self.machine_count))
        self.to_stock_prod[0,0] = 0
        self.to_stock_prod[0,1] = 0
        # Prod B needs machine 1 for 10 unit of time, and 15 for machine 2 (in minutes)
        self.to_stock_prod[1,0] = 1
        self.to_stock_prod[1,1] = 0

        # Selling Stock
        self.to_sell_prod = np.zeros((self.product_count, self.machine_count))
        self.to_sell_prod[0,0] = 1
        self.to_sell_prod[0,1] = 0
        # Prod B needs machine 1 for 10 unit of time, and 15 for machine 2 (in minutes)
        self.to_sell_prod[1,0] = 0
        self.to_sell_prod[1,1] = 1
        
        self.pending_raw = 0
        
        self.observation_space = gym.spaces.Dict(
            {
                # Stocks 
                "stockraw_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "stockraw_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "stockint_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "stockint_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "stock2sell_used": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "stock2sell_free": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),

                # What is in progress and how many to do next
                'current_batch_remaining': gym.spaces.Box(low=0, high=100, shape=(self.product_count, self.machine_count)),
                'next_batch': gym.spaces.Box(low=0, high=100, shape=(self.product_count, self.machine_count)),
                'ranking_next': gym.spaces.Box(low=np.zeros((self.product_count, self.machine_count)),
                                               high=np.ones((self.product_count, self.machine_count)), dtype=np.float32),

                # How does the staffing look like
                "count_operators_busy" : gym.spaces.Box(low=0, high=self.operatorcounts),
                "count_operators_free" : gym.spaces.Box(low=0, high=self.operatorcounts),

                # Where is made what with the cycle time
                "prod_assignment": gym.spaces.Box(low=0, high=500, shape=(self.product_count, self.machine_count), dtype=np.float32),

                # Pending Reception
                "pending_reception": gym.spaces.Box(low=0, high=1000),

                # Time of day
                "timeday": gym.spaces.Box(low=0, high=23)
            }
        )

        
        
        self.action_space = gym.spaces.Dict({
            # Allows to propose an alternative campaign size remaining
            'current_batch': gym.spaces.Box(low=0, high=100, shape=(self.product_count, self.machine_count)),
            # Enforce the campaign
            'force_current_batch': gym.spaces.MultiBinary((self.product_count, self.machine_count)),
            # Next Campaign batch size
            'next_batch': gym.spaces.Box(low=0, high=100, shape=(self.product_count, self.machine_count)),
            # Ranking for which products are active, higher better
            'ranking_next': gym.spaces.Box(low=np.zeros((self.product_count, self.machine_count)),
                                           high=np.ones((self.product_count, self.machine_count)), dtype=np.float32),
            # Do we need to order anything
            'order_raw_prod': gym.spaces.Box(low=0, high=100)
        })

        self.current_batch = np.zeros((self.product_count, self.machine_count))
        self.force_current_batch = np.zeros((self.product_count, self.machine_count))
        self.next_batch = np.zeros((self.product_count, self.machine_count))
        self.ranking_next = np.zeros((self.product_count, self.machine_count))

        # Unit of time for each step
        self.step_size = step # Decisions are made every hour
        self.episode_end = 7 * 24 # End after 7 days
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

        # self.shopsim.run(until=1/60)
    
    
    def _get_obs(self):
        return {
            "stockraw_used": np.array([self.stockraw.level], dtype=np.float32),
            "stockraw_free": np.array([self.stockraw.capacity - self.stockraw.level], dtype=np.float32),
            "stockint_used": np.array([len(self.stockint.items)], dtype=np.float32),
            "stockint_free": np.array([self.stockint.capacity - len(self.stockint.items)], dtype=np.float32),
            "stock2sell_used": np.array([len(self.stocksell.items)], dtype=np.float32),
            "stock2sell_free": np.array([self.stocksell.capacity - len(self.stocksell.items)], dtype=np.float32),
            
            "current_batch_remaining": self.current_batch.astype(np.float32),
            "next_batch": self.next_batch.astype(np.float32),
            "ranking_next": self.ranking_next.astype(np.float32),
            
            "count_operators_busy": np.array([self.operators.count], dtype=np.float32),
            "count_operators_free": np.array([self.operators.capacity - self.operators.count], dtype=np.float32),
            
            "prod_assignment": self.prod_assignment.astype(np.float32),
            
            "pending_reception": np.array([self.pending_raw], dtype=np.float32),
            
            "timeday": np.array([self.shopsim.now % 24], dtype=np.float32)
        }

    def _get_info(self):
        # Only for debugging purpose
        return {
            "production_log": self.prod_log,
            "sell_log": self.sell_log
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._make_simpy_env()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def order_raw_product(self, qty):
        qty = int(qty)
        self.pending_raw += qty
        yield self.shopsim.timeout(1)
        roominstock = min(self.stockraw.capacity - self.stockraw.level, qty)
        if (roominstock>0) and (roominstock<=self.stockraw.capacity):
            yield self.stockraw.put(roominstock)
        self.pending_raw -= qty

    def steal_product_at_night(self, threshold_stock=10, day_hour=6, probability=.1):
        # at night for 12 hours each product not in the shop is subject to be stollen at a 10 % chance per cycle time
        while True:
            if self.shopsim.now % day_hour > day_hour/2:
                # print('We are in Night mode, get your products in !!')
                # Filter Stores
                for stock in (self.stockint, self.stocksell):
                    if len(stock.items) > threshold_stock:
                        at_risk = (np.random.random(len(stock.items)-threshold_stock) < probability).astype(int)
                        for product_index in at_risk:
                            if product_index == 1:
                                yield stock.get()
                                # print('product Stollen from Store')
                # Containers 
                for stock in (self.stockraw, ):
                    if stock.level > threshold_stock:
                        at_risk = (np.random.random(stock.level-threshold_stock) < probability).astype(int)
                        for product_index in at_risk:
                            if product_index == 1:
                                yield stock.get(1)
                                # print('product Stollen fron Container')
            else:
                # print('We are in Day mode')
                pass
            yield self.shopsim.timeout(self.step_size)

    def make_products(self, machine_j=0, prod_i=0, patience = 60):
        cycletime = self.prod_assignment[prod_i, machine_j] #Minutes
        machine = getattr(self, f'machine_{machine_j}')

        # That needs to be improved / automated
        # if prod_i == 0:
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
            while self.current_batch[prod_i, machine_j] != 0:
                if self.forced_stop[prod_i, machine_j] == 1:
                    # We stop what we were doing immediately
                    self.forced_stop[prod_i, machine_j] = 0
                    break
                if filterstore:
                    # print(stockin.items , machine_j)
                    event = stockin.get(lambda x: x==str(prod_i))
                else:
                    event = stockin.get(1)

                timeout_event = self.shopsim.timeout(patience/60)

                result = yield event | timeout_event

                if event in result:
                    self.current_batch[prod_i, machine_j] -= 1
                    self.prod_log[prod_i, machine_j] +=1
                    self.salesrewards += .1
                    yield stockout.put(str(prod_i))
                    # print(f'Just made a product {self.prod_dict.get(str(prod_i))['Name']} - Machine {machine_j}')
                    yield self.shopsim.timeout(cycletime/60)
                else:
                    # print(f'Operator Gave up after {patience} minutes wait at {self.shopsim.now} - Machine {machine_j} - Product {prod_i}')
                    # self.poormanagementpenality +=5
                    break
        # if prod_made == self.current_batch[prod_i, machine_j]:          
        #     self.current_batch[prod_i, machine_j] = self.next_batch[prod_i, machine_j]
        #     self.next_batch[prod_i, machine_j] = 0

    def get_operators_to_work(self):
        while True:
            # if self.current_batch.sum() == 0:
            #     # If no current task, pass next to current
            #     for i, j in np.ndindex(self.next_batch.shape):
            #         if self.current_batch[i, j] == 0:
            #             self.current_batch[i, j] = self.next_batch[i, j]
            #             self.next_batch[i, j] = 0
            #             self.ranking_next[i, j] = 0
            if (self.operators.count < self.operators.capacity) and (self.next_batch.sum() != 0):
                max_idx = np.argmax(self.ranking_next)
                i, j = np.unravel_index(max_idx, self.ranking_next.shape)
                self.ranking_next[i, j] = 0
                self.current_batch[i, j] = self.next_batch[i, j]
                self.next_batch[i, j] = 0
                machine = getattr(self, f'machine_{j}')
                if machine.count!=0:
                    # Only pass the task if the machime is available
                    self.ranking_next[i, j] = 0
                    self.current_batch[i, j] = self.next_batch[i, j]
                    self.next_batch[i, j] = 0
                if (self.current_batch[i, j] > 0) and (machine.count==0) and (self.prod_assignment[i, j] != 0):
                    # print(f"Assigning an operator - Machine {j} - Product {i}")
                    self.shopsim.process(self.make_products(prod_i=i, machine_j=j))
                
            # Can try an else: here
            yield self.shopsim.timeout(self.step_size/60)

    def sell_products(self, freq=1):
        while True:
            # print(self.stocksell.items)
            if self.shopsim.now % freq == 0:
                allprods = [x for x in self.stocksell.items]
                for prod in allprods:
                    sold = yield self.stocksell.get(lambda x: x==prod)
                    # print(f'Congrats, you just sold a product {self.prod_dict.get(prod)['Name']} - Value {self.prod_dict.get(prod)['Cost']}')
                    self.sell_log[int(prod)] +=1
                    self.salesrewards += self.prod_dict.get(prod)['Cost']
            yield self.shopsim.timeout(self.step_size)

    def step(self, action):
        # Does he want to enforce a new batch size:
        reward = 0
        truncated = False
        terminated = True if self.shopsim.now == int(self.duration_max*24) else False

        current_batch = action['current_batch']
        force_current_batch = action['force_current_batch']
        next_batch = action['next_batch']
        ranking_next = action['ranking_next']
        order_raw_prod = action['order_raw_prod']

        for i, j in np.ndindex(force_current_batch.shape):
            if force_current_batch[i,j]==1:
                self.forced_stop[i,j] = 1
                self.current_batch[i,j] = int(current_batch[i,j])
                reward -= 5

        # We set what should be done next and with what urgency
        for i, j in np.ndindex(next_batch.shape):
            if (self.prod_assignment[i, j] == 0) and (next_batch[i, j] != 0):
                # Cant assign something when not planned too
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
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":
    env = ShopEnv(duration_max=2/24)
    observation, info = env.reset()
    print(observation)
    terminated = False
    rewards = []

    pc = env.product_count
    mc = env.machine_count

    while not terminated:
        # Raw product order
        rawproductorder = float(input("Raw Product Order: "))
        
        # Current batch
        print(f"\nCurrent Batch ({pc}x{mc} values):")
        current_batch = np.zeros((pc, mc), dtype=np.float32)
        # for i in range(pc):
        #     for j in range(mc):
        #         current_batch[i, j] = float(input(f"  current_batch[{i},{j}]: "))
        
        # Force current batch
        print(f"\nForce Current Batch ({pc}x{mc} values, 0 or 1):")
        force_current_batch = np.zeros((pc, mc), dtype=np.float32)
        # for i in range(pc):
        #     for j in range(mc):
        #         force_current_batch[i, j] = float(input(f"  force_current_batch[{i},{j}]: "))
        
        # Next batch
        print(f"\nNext Batch ({pc}x{mc} values):")
        next_batch = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                next_batch[i, j] = float(input(f"  next_batch[{i},{j}]: "))
        
        # Ranking next
        print(f"\nRanking Next ({pc}x{mc} values, 0-1):")
        ranking_next = np.zeros((pc, mc), dtype=np.float32)
        for i in range(pc):
            for j in range(mc):
                ranking_next[i, j] = float(input(f"  ranking_next[{i},{j}]: "))
        
        # Create action
        action = Action(
            current_batch=current_batch,
            force_current_batch=force_current_batch,
            next_batch=next_batch,
            ranking_next=ranking_next,
            order_raw_prod=rawproductorder
        )

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print("\nObservation:", observation)
        print("Reward:", reward)
        
        if truncated:
            terminated = True

    print('Final Reward = ', sum(rewards))
