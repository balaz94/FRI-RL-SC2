
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import torch

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])


class Env:
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "FindAndDefeatZerglings",
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
        self.marine1_ID = None
        self.marine2_ID = None
        self.marine3_ID = None
        self.last_target_pos = None
        self.camera_pos = None

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marine1 = None
        self.marine2 = None
        self.marine3 = None
        raw_obs = self.env.reset()[0]
        self.camera_pos = raw_obs.observation.camera_position
        return self.get_state_from_obs(raw_obs, True)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)

    def get_state_from_obs(self, raw_obs, reset):
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine)
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
        entity_matrix = np.array(raw_obs.observation.feature_minimap.player_relative)
        visibility_matrix = np.array(raw_obs.observation.feature_minimap.visibility_map)
        camera_matrix = np.array(raw_obs.observation.feature_minimap.camera)

        self.state = np.stack([entity_matrix, visibility_matrix, camera_matrix], axis=0)
        return self.state

    def step(self, action):
        raw_obs = self.take_action(action)
        new_state = self.get_state_from_obs(raw_obs, False)
        self.camera_pos = raw_obs.observation.camera_position
        return new_state, int(raw_obs.reward), raw_obs.last(), 0

    def take_action(self, action):
        x_axis_offset = 0
        y_axis_offset = 0
        if action % 8 == 0 :
            x_axis_offset -= 3
            y_axis_offset -= 3
        elif action % 8 == 1 :
            y_axis_offset -= 3
        elif action % 8 == 2 :
            x_axis_offset += 3
            y_axis_offset -= 3
        elif action % 8 == 3 :
            x_axis_offset += 3
        elif action % 8 == 4 :
            x_axis_offset += 3
            y_axis_offset += 3
        elif action % 8 == 5 :
            y_axis_offset += 3
        elif action % 8 == 6 :
            x_axis_offset -= 3
            y_axis_offset += 3
        elif action % 8 == 7 :
            x_axis_offset -= 3
        else:
            assert False
        if action < 8:
            if self.marine1 is not None:
                self.last_target_pos = [self.marine1.x + x_axis_offset, self.marine1.y + y_axis_offset]
                mapped_action = actions.RAW_FUNCTIONS.Attack_pt("now", self.marine1.tag, self.last_target_pos)
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        elif  16 > action >= 8:
            if self.marine2 is not None:
                self.last_target_pos = [self.marine2.x + x_axis_offset, self.marine2.y + y_axis_offset]
                mapped_action = actions.RAW_FUNCTIONS.Attack_pt("now", self.marine2.tag, self.last_target_pos)
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        elif  24 > action >= 16:
            if self.marine3 is not None:
                self.last_target_pos = [self.marine3.x + x_axis_offset, self.marine3.y + y_axis_offset]
                mapped_action = actions.RAW_FUNCTIONS.Attack_pt("now", self.marine3.tag, self.last_target_pos)
            else:
                mapped_action = actions.RAW_FUNCTIONS.no_op()
        elif  32 > action >= 24:
            mapped_action = actions.RAW_FUNCTIONS.raw_move_camera([self.camera_pos[0] + x_axis_offset, self.camera_pos[1] + y_axis_offset])
        else:
            assert False
        raw_obs = self.env.step([mapped_action])[0]

        return raw_obs

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