from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])


class Env:
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "CollectMineralShards",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=64, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
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
        self.marine1_ID = None
        self.marine2_ID = None
        self.last_op = None

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marine1 = None
        self.marine2 = None
        raw_obs = self.env.reset()[0]
        return self.get_state_from_obs(raw_obs, True)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)

    def get_state_from_obs(self, raw_obs, reset):
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine)
        if reset:
            self.marine1_ID = marines[0].tag
            self.marine2_ID = marines[1].tag
            self.marine1 = marines[0]
            self.marine2 = marines[1]
        else:
            if self.marine1_ID == marines[0].tag:
                self.marine1 = marines[0]
                self.marine2 = marines[1]
            elif self.marine1_ID == marines[1].tag:
                self.marine1 = marines[1]
                self.marine2 = marines[0]
            else:
                assert False
        shard_matrix = np.array(raw_obs.observation.feature_minimap.player_relative)
        shard_matrix[shard_matrix < 2] = 0

        marine1_matrix = np.zeros([64,64])
        marine1_matrix[self.marine1.x,int(self.marine1.y)] = 1

        marine2_matrix = np.zeros([64, 64])
        marine2_matrix[self.marine2.x, int(self.marine2.y)] = 2

        self.state =  np.stack([shard_matrix,marine1_matrix, marine2_matrix],axis=0)
        return self.state

    def step(self, action):
        raw_obs = self.take_action(action)
        new_state = self.get_state_from_obs(raw_obs, False)
        return new_state, int(raw_obs.reward), raw_obs.last(), 0

    def take_action(self, action):
        x_axis_offset = 0
        y_axis_offset = 0
        if action == 0 or action == 8:
            x_axis_offset -= 3
            y_axis_offset -= 3
        elif action == 1 or action == 9:
            y_axis_offset -= 3
        elif action == 2 or action == 10:
            x_axis_offset += 3
            y_axis_offset -= 3
        elif action == 3 or action == 11:
            x_axis_offset += 3
        elif action == 4 or action == 12:
            x_axis_offset += 3
            y_axis_offset += 3
        elif action == 5 or action == 13:
            y_axis_offset += 3
        elif action == 6 or action == 14:
            x_axis_offset -= 3
            y_axis_offset += 3
        elif action == 7 or action == 15:
            x_axis_offset -= 3
        else:
            assert False
        if action < 8:
            mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine1.tag, [self.marine1.x + x_axis_offset, self.marine1.y + y_axis_offset])
            self.last_op = mapped_action
        else:
            mapped_action = actions.RAW_FUNCTIONS.Move_pt("now", self.marine2.tag, [self.marine2.x + x_axis_offset, self.marine2.y + y_axis_offset])
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