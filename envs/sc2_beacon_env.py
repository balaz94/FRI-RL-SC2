from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])


class Env:
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "MoveToBeacon",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
        'step_mul': 0,
        'game_steps_per_episode' : 0,
        'visualize' : False,
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marine = None
        self.action_counter = 0

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marine = None
        self.action_counter = 0

        raw_obs = self.env.reset()[0]
        return self.get_state_from_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)

    def get_state_from_obs(self, raw_obs):
        self.marine = self.get_first_unit_by_type(raw_obs, units.Terran.Marine)
        screen_matrix = np.array(raw_obs.observation.feature_screen.player_relative)
        player_matrix = np.copy(screen_matrix)
        beacon_matrix = screen_matrix
        player_matrix[player_matrix > 1] = 0 # 1 for players units - marine
        beacon_matrix[beacon_matrix < 2] = 0 # 3 is for neutral units - including beacon
        state = np.stack([player_matrix,beacon_matrix],axis=0)

        return state

    def step(self, action):
        terminal = False
        reward = 0
        raw_obs = self.take_action(action)
        new_state = self.get_state_from_obs(raw_obs)
        if raw_obs.reward > 0:
            reward = 2
            terminal = True
        elif self.action_counter >= 100:
            terminal = True
        else:
            self.action_counter += 1
            reward = -0.0001
        return new_state, reward, terminal

    def take_action(self, action):
        x_axis_offset = 0
        y_axis_offset = 0
        if action == 0:
            x_axis_offset -= 1
            y_axis_offset -= 1
        elif action == 1:
            y_axis_offset -= 1
        elif action == 2:
            x_axis_offset += 1
            y_axis_offset -= 1
        elif action == 3:
            x_axis_offset += 1
        elif action == 4:
            x_axis_offset += 1
            y_axis_offset += 1
        elif action == 5:
            y_axis_offset += 1
        elif action == 6:
            x_axis_offset -= 1
            y_axis_offset += 1
        else:
            x_axis_offset -= 1
        mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine.tag, [self.marine.x + x_axis_offset,self.marine.y+y_axis_offset])
        raw_obs = self.env.step([mapped_action])[0]
        return raw_obs

    def get_first_unit_by_type(self, obs, unit_type):
        for unit in obs.observation.raw_units:
            if unit.unit_type == unit_type:
                return unit

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()