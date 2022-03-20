
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import torch
from collections import deque
import copy
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])


class Env:
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "MarinesZealotShieldToHealth",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=64, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                    use_camera_position=True),
        'step_mul': 4,
        'game_steps_per_episode' : 0,
        'visualize' : True,
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marine1 = None
        self.marine2 = None
        self.marine3 = None
        self.marine1_hp = -1
        self.marine2_hp = -1
        self.marine3_hp = -1
        self.zealot = None
        self.zealot_hp = -1
        self.marine1_ID = None
        self.marine2_ID = None
        self.marine3_ID = None
        self.raw_obs = None
        self.state = None
        self.partial_state_queue = deque()
        self.repeated_steps = 2
        self.marine_attack = np.array([False, False, False])

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marine1 = None
        self.marine2 = None
        self.marine3 = None
        self.zealot = None
        self.marine1_hp = -1
        self.marine2_hp = -1
        self.marine3_hp = -1
        self.zealot_hp = -1
        self.raw_obs = self.env.reset()[0]
        self.partial_state_queue.clear()
        self.update_state = True
        self.marine_attack = np.array([False, False, False])

        for i in range(20):
            arr = np.zeros((1, 64, 64))
            self.partial_state_queue.append(arr)
        return self.get_state_from_obs(True)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_state_from_obs(self, reset):
        zealot_array = self.get_units_by_type(self.raw_obs, units.Protoss.Zealot)
        if len(zealot_array) > 0:
            self.zealot = zealot_array[0]
        marines = self.get_units_by_type(self.raw_obs, units.Terran.Marine)
        if reset:
            self.marine1_ID = marines[0].tag
            self.marine2_ID = marines[1].tag
            self.marine3_ID = marines[2].tag
            self.marine1 = marines[0]
            self.marine2 = marines[1]
            self.marine3 = marines[2]
        else:
            mar_assigned = False
            for marine in marines:
                if marine.tag == self.marine1_ID:
                    mar_assigned = True
                    self.marine1 = marine
                    break
            if not mar_assigned:
                self.marine1 = None
            mar_assigned = False
            for marine in marines:
                if marine.tag == self.marine2_ID:
                    mar_assigned = True
                    self.marine2 = marine
                    break
            if not mar_assigned:
                self.marine2 = None
            mar_assigned = False
            for marine in marines:
                if marine.tag == self.marine3_ID:
                    mar_assigned = True
                    self.marine3 = marine
                    break
            if not mar_assigned:
                self.marine3 = None

        attack_matrix = np.zeros((1,64, 64))

        marine1_matrix = np.zeros((1,64, 64))
        if self.marine1 is not None:
             marine1_matrix[0,self.marine1.x,self.marine1.y] = self.marine1.health / 45.0
             if self.marine_attack[0]:
                 attack_matrix[0,self.marine1.x,self.marine1.y] = 1
        else:
            self.marine_attack[0] = False
        marine2_matrix = np.zeros((1,64, 64))
        if self.marine2 is not None:
             marine2_matrix[0,self.marine2.x, self.marine2.y] = self.marine2.health / 45.0
             if self.marine_attack[1]:
                 attack_matrix[0,self.marine2.x,self.marine2.y] = 1
        else:
            self.marine_attack[1] = False
        marine3_matrix = np.zeros((1,64, 64))
        if self.marine3 is not None:
             marine3_matrix[0,self.marine3.x, self.marine3.y] = self.marine3.health / 45.0
             if self.marine_attack[2]:
                 attack_matrix[0,self.marine3.x,self.marine3.y] = 1
        else:
            self.marine_attack[2] = False

        enemy_matrix = np.zeros((1,64, 64))
        if self.zealot is not None:
            enemy_matrix[0, self.zealot.x, self.zealot.y] = self.zealot.health / 127.0

        self.partial_state_queue.append(marine1_matrix)
        self.partial_state_queue.append(marine2_matrix)
        self.partial_state_queue.append(marine3_matrix)
        self.partial_state_queue.append(enemy_matrix)
        self.partial_state_queue.append(attack_matrix)

        for i in range(5):
            self.partial_state_queue.popleft()

        self.state = np.zeros((0, 64, 64))

        boundaries_matrix = np.zeros((1, 64, 64))
        for i in range(64):
            boundaries_matrix[0, 0, i] = 1
            boundaries_matrix[0, 63, i] = 1
            boundaries_matrix[0, i, 0] = 1
            boundaries_matrix[0, i, 63] = 1

        for x in self.partial_state_queue:
             self.state = np.append(self.state, x, axis=0)

        self.state = np.append(self.state, boundaries_matrix, axis=0)

        #self.state = np.stack([marine1_matrix, marine2_matrix, marine3_matrix, enemy_matrix], axis=0)
        return self.state

    def step(self, action):
        reward = 0.0
        for i in range(self.repeated_steps):
            self.take_action(action)
            if self.raw_obs.reward == 1:
                reward = 0
                break
            else:
                if self.marine1_hp == -1:
                    self.marine1_hp = self.marine1.health
                else:
                    if self.marine1 is not None and self.marine1.health < self.marine1_hp:
                        reward -= ((self.marine1_hp - self.marine1.health) / 45.0) * 10
                        self.marine1_hp = self.marine1.health
                if self.marine2_hp == -1:
                    self.marine2_hp = self.marine2.health
                else:
                    if self.marine2 is not None and self.marine2.health < self.marine2_hp:
                        reward -= ((self.marine2_hp - self.marine2.health) / 45.0) * 10
                        self.marine2_hp = self.marine2.health
                if self.marine3_hp == -1:
                    self.marine3_hp = self.marine3.health
                else:
                    if self.marine3 is not None and self.marine3.health < self.marine3_hp:
                        reward -= ((self.marine3_hp - self.marine3.health) / 45.0) * 10
                        self.marine3_hp = self.marine3.health
                if self.zealot is not None:
                    if self.zealot_hp == -1:
                        self.zealot_hp = self.zealot.health
                    else:
                        if self.zealot.health < self.zealot_hp:
                            reward += ((self.zealot_hp - self.zealot.health) / 127.0) * 10
                            self.zealot_hp = self.zealot.health



            if self.raw_obs.last():
                break
        new_state = self.get_state_from_obs(False)
        return new_state, reward, self.raw_obs.last()

    def take_action(self, action):
        x_axis_offset = 0
        y_axis_offset = 0
        if action % 9 == 0 :
            x_axis_offset -= 3
            y_axis_offset -= 3
        elif action % 9 == 1 :
            y_axis_offset -= 3
        elif action % 9 == 2 :
            x_axis_offset += 3
            y_axis_offset -= 3
        elif action % 9 == 3 :
            x_axis_offset += 3
        elif action % 9 == 4 :
            x_axis_offset += 3
            y_axis_offset += 3
        elif action % 9 == 5 :
            y_axis_offset += 3
        elif action % 9 == 6 :
            x_axis_offset -= 3
            y_axis_offset += 3
        elif action % 9 == 7 :
            x_axis_offset -= 3
        elif action % 9 != 8 :
            assert False

        if action < 9:
            if self.marine1 is not None:
                last_target_pos = [self.marine1.x + x_axis_offset, self.marine1.y + y_axis_offset]
                if action < 8:
                    mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine1.tag, last_target_pos)
                    self.marine_attack[0] = False
                else:
                    mapped_action = actions.RAW_FUNCTIONS.Attack_unit("now", self.marine1.tag, self.zealot.tag)
                    self.marine_attack[0] = True
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        elif 18 > action >= 9:
            if self.marine2 is not None:
                last_target_pos = [self.marine2.x + x_axis_offset, self.marine2.y + y_axis_offset]
                if action < 17:
                    mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine2.tag, last_target_pos)
                    self.marine_attack[1] = False
                else:
                    mapped_action = actions.RAW_FUNCTIONS.Attack_unit("now", self.marine2.tag, self.zealot.tag)
                    self.marine_attack[1] = True
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        elif 27 > action >= 18:
            if self.marine3 is not None:
                last_target_pos = [self.marine3.x + x_axis_offset, self.marine3.y + y_axis_offset]
                if action < 26:
                    mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine3.tag, last_target_pos)
                    self.marine_attack[2] = False
                else:
                    mapped_action = actions.RAW_FUNCTIONS.Attack_unit("now", self.marine3.tag, self.zealot.tag)
                    self.marine_attack[2] = True
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        else:
            assert False
        self.raw_obs = self.env.step([mapped_action])[0]

    def get_units_by_type(self, obs, unit_type):
        unit_list = []
        for unit in obs.observation.raw_units:
            if unit.unit_type == unit_type:
                unit_list.append(unit)
        return unit_list

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def get_raw_obs(self):
        return self.raw_obs
