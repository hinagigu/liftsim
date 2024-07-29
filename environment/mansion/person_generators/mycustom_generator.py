import random
import math
import numpy as np

# 常量配置
DEFAULT_ARRIVAL_RATE = 0.5
ARRIVAL_RATE_THRESHOLD_1 = 1800
ARRIVAL_RATE_THRESHOLD_2 = 36000
ARRIVAL_RATE_DECREASE_FACTOR = 12
ARRIVAL_RATE_INCREASE_FACTOR = 12
ORIGINS_INCREMENT = 0.04

# 停留时间分布配置
staytime_distributions = [
    {'type': 'normal', 'mean': 100, 'std': 20, 'weight': 0.3},  # 短时间停留
    {'type': 'normal', 'mean': 300, 'std': 50, 'weight': 0.4},  # 中时间停留
    {'type': 'normal', 'mean': 600, 'std': 100, 'weight': 0.3}  # 长时间停留
]

# 交通参数配置
traffic_params = {'U': 0.5, 'I': 0.3, 'D': 0.2}

class TrafficSimulator:
    def __init__(self, num_floors, population_per_floor, staytime_distributions, traffic_params):
        self.num_floors = num_floors
        self.floors = [self.Floor(population_per_floor, i + 1, staytime_distributions) for i in range(num_floors)]
        self.traffic_params = traffic_params
        self.building_population = sum(floor.population for floor in self.floors)
        self.od_matrix = self.set_matrix()
        self.time = 0
        self.arrival_rate = DEFAULT_ARRIVAL_RATE
        self.initialize_origins()

    class Floor:
        def __init__(self, population, number, staytime_distributions):
            self.population = population
            self.number = number
            self.staytime = self.generate_staytime(staytime_distributions)
            self.origins = 0
            self.possibilities = 0

        def generate_staytime(self, distributions):
            # 选择一个分布
            distribution = random.choices(distributions, weights=[d['weight'] for d in distributions])[0]
            if distribution['type'] == 'normal':
                return max(0, np.random.normal(distribution['mean'], distribution['std']))
            elif distribution['type'] == 'exponential':
                return np.random.exponential(distribution['scale'])
            else:
                raise ValueError("Unsupported distribution type")

    def initialize_origins(self):
        for floor in self.floors:
            floor.origins = (self.traffic_params['I'] + self.traffic_params['D']) * floor.population / self.building_population

    def set_matrix(self):
        matrix = [[0 for _ in range(self.num_floors)] for _ in range(self.num_floors)]
        for i in range(self.num_floors):
            for j in range(self.num_floors):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = self.traffic_params['I'] / (self.traffic_params['D'] + self.traffic_params['I']) * (self.floors[j].population / self.building_population)
        return matrix

    def update_time(self):
        self.time -= (math.log(random.random()) / self.arrival_rate)
        if self.time >= ARRIVAL_RATE_THRESHOLD_1 and self.arrival_rate > 0.1:
            self.arrival_rate /= ARRIVAL_RATE_DECREASE_FACTOR
        if self.time >= ARRIVAL_RATE_THRESHOLD_2 and self.arrival_rate < 0.1:
            self.arrival_rate *= ARRIVAL_RATE_INCREASE_FACTOR

    def elect_start(self):
        sum_origins = sum(floor.origins for floor in self.floors)
        if sum_origins == 0:
            return random.randint(1, self.num_floors)

        for floor in self.floors:
            floor.possibilities = floor.origins / sum_origins

        possibilities = [0] * self.num_floors
        for i in range(self.num_floors):
            for j in range(i + 1):
                possibilities[i] += self.floors[j].possibilities

        R = random.random()
        current_floor = 1
        if R < possibilities[0]:
            current_floor = 1
        for i in range(1, self.num_floors):
            if possibilities[i] >= R and possibilities[i - 1] <= R:
                current_floor = i + 1

        return current_floor

    def elect_destination(self, start):
        sums_od = [sum(self.od_matrix[i]) for i in range(self.num_floors)]
        possbs = [self.od_matrix[start - 1][j] / sums_od[start - 1] if sums_od[start - 1] != 0 else 0 for j in range(self.num_floors)]
        qijs = [sum(possbs[:j + 1]) for j in range(self.num_floors)]

        current_floor = 1
        ran = random.random()
        if ran < qijs[0] and start != 1:
            current_floor = 1
        for i in range(1, self.num_floors):
            if qijs[i] >= ran and qijs[i - 1] < ran and i + 1 != start:
                current_floor = i + 1

        while current_floor == start:
            ran = random.random()
            if ran < qijs[0] and start != 1:
                current_floor = 1
            for i in range(1, self.num_floors):
                if qijs[i] >= ran and qijs[i - 1] < ran and i + 1 != start:
                    current_floor = i + 1

        return current_floor

    def update_origins(self, dest_floor):
        dest_floor.origins += ORIGINS_INCREMENT
        sum_origins = sum(floor.origins for floor in self.floors)
        for floor in self.floors:
            floor.possibilities = floor.origins / sum_origins

    def simulate(self):
        self.update_time()
        start_floor = self.elect_start()
        dest_floor = self.elect_destination(start_floor)
        self.update_origins(self.floors[dest_floor - 1])
        return start_floor, dest_floor

# 使用示例
num_floors = 10
population_per_floor = 100

simulator = TrafficSimulator(num_floors, population_per_floor, staytime_distributions, traffic_params)
start_floor, dest_floor = simulator.simulate()
print(f"Start Floor: {start_floor}, Destination Floor: {dest_floor}")

# 使用示例
num_floors = 10
population_per_floor = 100
traffic_params = {'U': 0.5, 'I': 0.3, 'D': 0.2}

simulator = TrafficSimulator(num_floors, population_per_floor, staytime_distributions, traffic_params)
for i in range(110):
    start_floor, dest_floor = simulator.simulate()
    print(f"Start Floor: {start_floor}, Destination Floor: {dest_floor}")