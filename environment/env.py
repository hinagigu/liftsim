#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from metagym.liftsim.environment.mansion.person_generators.generator_proxy import set_seed
from metagym.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from metagym.liftsim.environment.mansion.mansion_config import MansionConfig
from metagym.liftsim.environment.mansion.mansion_manager import MansionManager
from metagym.liftsim.environment.mansion.utils import ElevatorAction

NoDisplay = False
try:
    from metagym.liftsim.environment.animation.rendering import Render
except Exception as e:
    NoDisplay = True

import gym
import argparse
import configparser
import random
import sys
import os
from gymnasium.spaces import Dict,Discrete,MultiDiscrete,MultiBinary,Box
from gymnasium.vector.utils.spaces import batch_space
class LiftSim(gym.Env):
    """
    environmentation Environment
    """

    def __init__(self, config_file=os.path.join(os.path.dirname(__file__) + '/../config.ini'), **kwargs):
        file_name = config_file

        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy'

        # Readin different person generators
        gtype = config['PersonGenerator']['PersonGeneratorType']
        person_generator = PersonGenerator(gtype)
        person_generator.configure(config['PersonGenerator'])
        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight'])
        )

        if ('LogLevel' in config['Configuration']):
            assert config['Configuration']['LogLevel'] in ['Debug', 'Notice', 'Warning'], \
                'LogLevel must be one of [Debug, Notice, Warning]'
            self._config.set_logger_level(config['Configuration']['LogLevel'])
        if ('Lognorm' in config['Configuration']):
            self._config.set_std_logfile(config['Configuration']['Lognorm'])
        if ('Logerr' in config['Configuration']):
            self._config.set_err_logfile(config['Configuration']['Logerr'])

        self._mansion = MansionManager(
            int(config['MansionInfo']['ElevatorNumber']),
            person_generator,
            self._config,
            config['MansionInfo']['Name']
        )


        self.viewer = None
        self.action_space = None
        # self.observation_space = None
        self.elevator_space = Dict({
        "Floor": Discrete(start=1,n=self._config.number_of_floors),
        "MaximumFloor": Discrete(start=1,n=self._config.number_of_floors),
        "Velocity": Box(low=-self._config.maximum_speed, high=self._config.maximum_speed, shape=(1,)),
        "MaximumSpeed": Box(low=self._config.maximum_speed, high=self._config.maximum_speed, shape=(1,)),
        "Direction": Discrete(3),
        "DoorState": Box(low=0.0, high=1.0, shape=(1,)),#door open rate
        "CurrentDispatchTarget": Discrete(self._config.number_of_floors + 1),
        "DispatchTargetDirection": Discrete(3),
        "LoadWeight": Box(low=0.0, high=self._config.maximum_capacity, shape=(1,)),
        "MaximumLoad": Box(low=self._config.maximum_capacity, high=self._config.maximum_capacity, shape=(1,)),
        "ReservedTargetFloors": MultiBinary(self._config.number_of_floors),
        "OverloadedAlarm": Box(low=0.0, high=2.0, shape=(1,)),
        "DoorIsOpening": Discrete(2),
        "DoorIsClosing": Discrete(2),
        })
        self.observation_space = Dict(
        {
            "elevator_states": batch_space(self.elevator_space,self._mansion._elevator_number),
            "upward_requests": MultiBinary(self._mansion._floor_number),
            "downward_requests": MultiBinary(self._mansion._floor_number)
        })
    def seed(self, seed=None):
        set_seed(seed)

    def step(self, action):
        assert type(action) is list, "Type of action should be list"
        assert len(action) == self._mansion.attribute.ElevatorNumber * 2, \
            "Action is supposed to be a list with length ElevatorNumber * 2"
        action_tuple = []
        for i in range(self._mansion.attribute.ElevatorNumber):
            action_tuple.append(ElevatorAction(action[i * 2], action[i * 2 + 1]))
        # 每个电梯接受两个数字作为action,范围必须是【-1，0，1】

        time_consume, energy_consume, given_up_persons = self._mansion.run_mansion(action_tuple)
        reward = - (time_consume + 5e-4 * energy_consume +
                    300 * given_up_persons) * 1.0e-4
        info = {'time_consume': time_consume, 'energy_consume': energy_consume, 'given_up_persons': given_up_persons}
        return self._mansion.state, reward, False, info
    def get_mansion(self):
        return self._mansion
    def reset(self):
        self._mansion.reset_env()
        info = {'time_consume': 0, 'energy_consume': 0, 'given_up_persons': 0}
        return self._mansion.state

    def render(self, mode="human"):
        if (mode != "human"):
            raise NotImplementedError("Only support human mode currently")
        if self.viewer is None:
            if NoDisplay:
                raise Exception('[Error] Cannot connect to display screen. \
                    \n\rYou are running the render() function on a manchine that does not have a display screen')
            self.viewer = Render(self._mansion)
        self.viewer.on_draw()

    def close(self):
        pass

    @property
    def attribute(self):
        return self._mansion.attribute

    @property
    def state(self):
        return self._mansion.state

    @property
    def statistics(self):
        return self._mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_notice

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal
