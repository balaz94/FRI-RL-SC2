import gym
import numpy as np
import cv2


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, penalty=False, steps_after_reset=0):
        super(SkipEnv, self).__init__(env)
        self._skip = skip
        self._lives = 0
        self._penalty = penalty
        self.steps_after_reset = steps_after_reset

    def step(self, action):
        t_reward = 0.0
        done = False

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if self._penalty:
                t_reward += self.ale.lives() - self._lives
                self._lives = self.ale.lives()
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        output = self.env.reset()
        for i in range(self.steps_after_reset):
            output, _, _, _ = self.env.step(self.env.action_space.sample())

        self._lives = self.ale.lives()
        return output


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name="video.avi"):
        super(VideoRecorder, self).__init__(env)

        self.height = 2 * env.observation_space.shape[0]
        self.width = 2 * env.observation_space.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(file_name, fourcc, 50.0, (self.width, self.height))
        self.frame_counter = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.frame_counter % 4 == 0:
            im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

            resized = cv2.resize(im_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

            self.writer.write(resized)

        self.frame_counter += 1

        return state, reward, done, info

    def reset(self):
        return self.env.reset()


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None, y1=0, y2=210, x1=0, x2=160):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 1), dtype=np.uint8)
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def observation(self, obs):
        return PreProcessFrame.process(obs, self.y1, self.y2, self.x1, self.x2)

    @staticmethod
    def process(frame, y1, y2, x1, x2):
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame[y1:y2, x1:x2], (96, 96), interpolation=cv2.INTER_AREA)  # 35:195
        new_frame = np.reshape(new_frame, (96, 96, 1))
        return new_frame.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.observation_space.shape[-1],
                                                       self.observation_space.shape[0],
                                                       self.observation_space.shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(n_steps, axis=0),
                                                env.observation_space.high.repeat(n_steps, axis=0),
                                                dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name, y1=0, y2=210, x1=0, x2=160, penalty=False, steps_after_reset=0):
    env = gym.make(env_name)
    env = SkipEnv(env, penalty=penalty, steps_after_reset=steps_after_reset)
    env = PreProcessFrame(env, y1, y2, x1, x2)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)


def make_env_with_record(env_name, y1=0, y2=210, x1=0, x2=160, penalty=False, steps_after_reset=0):
    env = gym.make(env_name)
    env = SkipEnv(env, penalty=penalty, steps_after_reset=steps_after_reset)
    env = VideoRecorder(env)
    env = PreProcessFrame(env, y1, y2, x1, x2)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)