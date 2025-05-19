import gym
import numpy as np
from gym import spaces
import pygame

from ctypes import DEFAULT_MODE
from math import sqrt
AGENT_SIZE = 15
WIDTH = 1000
HEIGHT = 600
MAX_SPEED = 10
MAX_ACCELERATE = 1
OBSERVED_RADIUS = 300
SAFE_SPEED_RANGE = [1,4]
DEFAULT_DISTANCE = sqrt(WIDTH**2 + HEIGHT**2)
default_location = {
    50: [WIDTH/3 - 90, HEIGHT/3 + 20],
    64: [WIDTH/4 - 40, HEIGHT/3],
    36: [WIDTH/3 - 75, HEIGHT/3 - 45],
    49: [WIDTH/2 + 30, HEIGHT/2 + 15]
}
class CarsModel:
    def __init__(self, N, history_len=10,bot_mode="random"):
        '''
        N: number of cars
        First car is ego car, others are bots
        history_len: length of history of each car
        '''
        self.N = N
        self.history_len = history_len
        self.state_dim = 5  # [x, y, angle, speed, acceleration]
        self.car_states = np.zeros((N, self.state_dim), dtype=np.float32)
        self.histories = np.zeros((N, history_len, self.state_dim), dtype=np.float32)
        self.change_angle_freqs = np.zeros(N, dtype=np.int32)
        self.change_accel_freqs = np.zeros(N, dtype=np.int32)
        self.rects = [pygame.Rect(0, 0, AGENT_SIZE, AGENT_SIZE) for _ in range(N)]
        self.angle_rects = [pygame.Rect(0, 0, 4, 4) for _ in range(N)]

        self.car_states[0, 2] = -90
        self.car_states[0, 3] = 0  # speed
        self.bot_mode = bot_mode
        self.default_speed = 2

        self.max_angle_delta = 30
        self.min_angle_delta = 5

        self.ego_state_code = 0

        self._init_agent()

    def set_ego_action(self, angle_delta, acc_delta):
        # Assume this is called before step()
        self.car_states[0, 2] += np.clip(angle_delta,-self.max_angle_delta,self.max_angle_delta)  # angle
        self.car_states[0, 4] = np.clip(acc_delta, -MAX_ACCELERATE, MAX_ACCELERATE)  # acceleration

    def update_histories(self):
        self.histories = np.roll(self.histories, -1, axis=1)
        self.histories[:, -1, :] = self.car_states

    def bot_logic(self):
        if self.bot_mode == "random":
            self._bot_random_move()
        elif self.bot_mode == "chase_ego":
            self._chase_ego()


    def apply_movement(self):
        
        speeds = self.car_states[:, 3] + self.car_states[:, 4]
        speeds = np.clip(speeds, 0, MAX_SPEED)
        self.car_states[:, 3] = speeds

        angles = self.car_states[:, 2]
        dx = speeds * np.cos(np.radians(-angles))
        dy = speeds * np.sin(np.radians(-angles))
        self.car_states[:, 0] += dx
        self.car_states[:, 1] += dy

        # self._validate_movement_for_ego()
    
    def _validate_movement_for_ego(self):
        '''
        If the ego car is not moving, it just modify angle in specific ragnge
        '''
        ego_state = self.car_states[0]
        if ego_state[3] < 1:
            if self.ego_state_code == 0:     
                self.ego_state_code = 1
                self.ego_allowed_angle = [ego_state[2] - self.max_angle_delta, ego_state[2] + self.max_angle_delta]
            ego_state[2] = np.clip(ego_state[2], self.ego_allowed_angle[0], self.ego_allowed_angle[1])
        else:
            self.ego_state_code = 0
        

    def apply_collision(self):
        '''
        Just apply collision for cars except ego car
        The ego can go through the edges of the screen
        '''
        #Ego
        self.car_states[0, 0], self.car_states[0, 1] = self.car_states[0, 0] % WIDTH, self.car_states[0, 1] % HEIGHT
        #Bots
        x, y, angle, authorized_area = self.car_states[1:, 0], self.car_states[1:, 1], self.car_states[1:, 2], self.authorized_area[1:]
        collide_x_left = x < authorized_area[:, 0]
        collide_x_right = x + AGENT_SIZE > authorized_area[:, 2]
        x[collide_x_left] = authorized_area[collide_x_left, 0]
        x[collide_x_right] = authorized_area[collide_x_right, 2] - AGENT_SIZE

        collide_left_or_right = collide_x_left | collide_x_right
        angle[collide_x_left | collide_x_right] = 180 - angle[collide_left_or_right]

        collide_y_top = y < authorized_area[:, 1]
        collide_y_bottom = y + AGENT_SIZE > authorized_area[:, 3]
        y[collide_y_top] = authorized_area[collide_y_top, 1]
        y[collide_y_bottom] = authorized_area[collide_y_bottom, 3] - AGENT_SIZE
        collide_top_or_bottom = collide_y_top | collide_y_bottom
        angle[collide_y_top | collide_y_bottom] = -angle[collide_top_or_bottom]

        both = collide_left_or_right & collide_top_or_bottom
        angle[both] = -angle[both]

        self.car_states[1:, 0], self.car_states[1:, 1], self.car_states[1:, 2] = x, y, angle

    def sync_rects(self):
        distances = AGENT_SIZE/2 + 4
        for i in range(self.N):
            rect = self.rects[i]
            angle_rect = self.angle_rects[i]
            rect.x = int(self.car_states[i, 0])
            rect.y = int(self.car_states[i, 1])
            # angle_rects are used to draw the direction of the car
            angle_rect.centerx = int(distances * np.cos(np.radians(-self.car_states[i, 2]))) + rect.centerx
            angle_rect.centery = int(distances * np.sin(np.radians(-self.car_states[i, 2]))) + rect.centery

    def step(self,is_training=False):
        self.bot_logic()
        self.apply_collision()
        self.apply_movement()
        if not(is_training): self.sync_rects()
        self.update_histories()

    def render(self, screen):
        for i, rect in enumerate(self.rects):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.rect(screen, color, rect)

        for i, angle_rect in enumerate(self.angle_rects):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.rect(screen, color, angle_rect)

        #Draw observation of ego
        ego_location = self.car_states[0, :2]
        area = [ego_location[0] - OBSERVED_RADIUS/2, ego_location[1] - OBSERVED_RADIUS/2, OBSERVED_RADIUS, OBSERVED_RADIUS]
        pygame.draw.rect(screen, (0, 0, 255), (int(area[0]), int(area[1]), int(area[2]), int(area[3])), 1)

        # Draw a rectangle 200 x 200 for test
        # test_rect = pygame.Rect(200, 200, 200, 200)
        # pygame.draw.rect(screen, (0, 0, 255), test_rect)


    def get_ego_state(self):
        return self.car_states[0]

    def get_latest_histories(self,len=None):
        if len is None: return self.histories
        else:
            return self.histories[:, -len:, :]

    def get_closest_agents(self, radius=-1, num_closet_agents=5):
        '''
        Return indices of the closest agents within the given radius.
        '''
        ego_pos = self.car_states[0, :2]
        others = self.car_states[1:, :2]

        # Compute wrapped (toroidal) distances
        dx = np.abs(others[:, 0] - ego_pos[0])
        dx = np.minimum(dx, WIDTH - dx)

        dy = np.abs(others[:, 1] - ego_pos[1])
        dy = np.minimum(dy, HEIGHT - dy)

        dists = np.sqrt(dx**2 + dy**2)

        # Get closest N agents within radius
        sorted_idxs = np.argsort(dists)
        closest_idxs = sorted_idxs[:num_closet_agents]
        if radius < 0:
            return closest_idxs + 1
        mask = dists[closest_idxs] < radius
        return closest_idxs[mask] + 1  # +1 to adjust for skipped ego agent

    def _init_agent(self):
        '''
            Locate evenly the agents in the environment.
            Randomly choose angle and set speed to 1.
            Set the authorized area for the agents which is circle with radius AGENT_SIZE * 4.
        '''
        N = self.N
        # Location
        nums = int(np.sqrt(N))
        for i in range(1,N):
            self.car_states[i, 0] = (i % nums) * (WIDTH / nums) + AGENT_SIZE / 2
            self.car_states[i, 1] = (i // nums) * (HEIGHT / nums) + AGENT_SIZE / 2
        # Random angle and set speed to 1
        angles = np.random.randint(0, 360, size=N - 1)
        self.car_states[1:, 2] = angles
        self.car_states[1:, 3] = self.default_speed

        #Location for ego
        if default_location.get(N) is None:
            idx_x = min(2,nums / 2)
            idx_y = min(2, nums / 2)
            idx_location = idx_x + idx_y * nums
            self.car_states[0, 0] =  (self.car_states[idx_location, 0] + self.car_states[idx_location + 1, 0]) / 2 
            self.car_states[0, 1] =  (self.car_states[idx_location, 1] + self.car_states[idx_location + 1, 1]) / 2 
        else:
            self.car_states[0, 0] = default_location[N][0] # x
            self.car_states[0, 1] = default_location[N][1]  # y
            self.inital_location = np.array(default_location[N])

        # Authorized area (x_min, y_min, x_max, y_max)
        self.authorized_area = np.zeros((self.N, 4), dtype=np.float32)
        self.authorized_area[0, 0:4] = 0, 0, WIDTH, HEIGHT
        self.authorized_area[1:, 0] = np.maximum(0, self.car_states[1:, 0] - AGENT_SIZE * 10)
        self.authorized_area[1:, 1] = np.maximum(0, self.car_states[1:, 1] - AGENT_SIZE * 10)
        self.authorized_area[1:, 2] = np.minimum(WIDTH, self.car_states[1:, 0] + AGENT_SIZE * 10)
        self.authorized_area[1:, 3] = np.minimum(HEIGHT, self.car_states[1:, 1] + AGENT_SIZE * 10)


    def check_stand_still(self,ratio_timesteps=-1):
        '''
            Check if the ego car is still and distance to initial location < 50
        '''
        unvalid_distance = False
        if ratio_timesteps > 0 and ratio_timesteps < 1:
            unvalid_distance = np.linalg.norm(self.car_states[0, :2] - self.inital_location) < WIDTH * ratio_timesteps
        return self.car_states[0, 3] < 1

    def _bot_random_move(self):
        '''
            Mask is index of the cars which apply the random move.
        '''

        mask_angle = (self.change_angle_freqs[1:] >= 10)
        angle_updates = np.random.randint(-30, 30, size=self.N - 1)
        self.car_states[1:][mask_angle, 2] += angle_updates[mask_angle]
        self.change_angle_freqs[1:][mask_angle] = 0
        self.change_angle_freqs[1:][~mask_angle] += 1

        accel = self.car_states[1:, 4]
        mask_up = (self.change_accel_freqs[1:] < 10)
        mask_down = (self.change_accel_freqs[1:] >= 10) & (self.change_accel_freqs[1:] < 20)
        mask_reset = (self.change_accel_freqs[1:] >= 20)

        accel[mask_up] = np.maximum(0, accel[mask_up] + 0.02)
        accel[mask_down] = np.minimum(0, accel[mask_down] - 0.03)
        accel[mask_reset] = 0.0
        self.car_states[1:, 4] = accel
        self.car_states[1:, 3][mask_reset] = self.default_speed
        self.change_accel_freqs[1:] += 1
        self.change_accel_freqs[1:][mask_reset] = 0

    def _chase_ego(self):
        """
        Makes bot cars chase the ego car if it's within their authorized area.
        The bots adjust their angle toward the ego and increase their speed slightly when chasing.
        """
        ego_pos = self.car_states[0, :2]
        bot_positions = self.car_states[1:, :2]
        bot_angles = self.car_states[1:, 2]

        # Determine which bots can "see" the ego car (i.e., within their authorized area)
        x_min, y_min = self.authorized_area[1:, 0], self.authorized_area[1:, 1]
        x_max, y_max = self.authorized_area[1:, 2], self.authorized_area[1:, 3]

        in_x_range = (ego_pos[0] > x_min) & (ego_pos[0] < x_max)
        in_y_range = (ego_pos[1] > y_min) & (ego_pos[1] < y_max)
        can_chase = in_x_range & in_y_range

        # Calculate angle to ego for all bots
        vector_to_ego = ego_pos - bot_positions
        desired_angles = np.degrees(np.arctan2(vector_to_ego[:, 1], vector_to_ego[:, 0])) % 360

        # Apply random movement to all bots first
        self._bot_random_move()

        # Adjust angles for chasing bots
        current_angles = bot_angles % 360
        angle_diff = np.abs(current_angles - desired_angles)

        # Mask for bots that can directly align with ego
        align_mask = (angle_diff < 30) & can_chase
        adjust_mask = (angle_diff >= 30) & can_chase

        self.car_states[1:, 2][align_mask] = - desired_angles[align_mask]
        self.car_states[1:, 2][adjust_mask] = - (desired_angles[adjust_mask] +
                                            np.random.randint(-30, 30, size=np.sum(adjust_mask)))

        # Set speed and acceleration for chasing bots as ego speed and 0
        self.car_states[1:, 3][can_chase] = self.default_speed
        self.car_states[1:, 4][can_chase] = 0.05


class GameObservation:
    def __init__(self, features_num, obs_dim, cars_num):
        super().__init__()
        self.features_num = features_num
        self.obs_dim = obs_dim
        self.cars_num = cars_num
    def get_cars_num(self):
        return self.cars_num
    def get_ego_idx(self):
        return -1


class CarEnv(gym.Env):
    safe_distance = AGENT_SIZE * 1.5
    collide_distance = sqrt((AGENT_SIZE**2)* 2)
    def __init__(self, num_agents=2, observation_dim=3, history_len=10,is_training=False,bot_mode="random", max_collide_num=30):
        '''
        Observation space: (num_closest_agents, history_len, observation_dim) containing (speed, angle, distance to ego)
        Action space: (angle_delta, acceleration) where angle_delta is in degrees and acceleration is in m/s^2
        '''
        super().__init__()
        self.num_agents = num_agents
        self.agents = CarsModel(num_agents, history_len=history_len, bot_mode="random")  # ego = index 0
        self.ego = self.agents.car_states[0]
        self.bot_mode = bot_mode
        self.max_collide_num = max_collide_num
        self.target = np.array([700, 100])

        self.history_len = history_len
        self.observation_dim = 4  # speed, angle, distance to ego, time_steps
        self.is_training = is_training
        self.num_closet_agents = 5
        
        self.game_obs = GameObservation(self.num_closet_agents * self.history_len + 1,4,self.num_closet_agents)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_closet_agents * self.history_len + 1, self.observation_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self._init_essential_items()


    def reset(self):
        self.agents = CarsModel(self.num_agents,bot_mode=self.bot_mode)
        self.step_count = 0
        self.initial_distance_to_target = np.linalg.norm(
            self.target - np.array(self.agents.car_states[0, :2])
        )
        self.current_steps = 0
        self.agent_steps = 0
        self.acumulated_reward = 0.0
        self.current_collide_time = 0
        self.stand_still_num = 0
        self.change_angle_num = 0
        self.change_acc_num = 0
        #Random target
        self.target = np.array([np.random.randint(WIDTH /2, WIDTH), np.random.randint(50, HEIGHT - 50)])
        return self._get_obs()

    def _get_obs(self):
        '''
        Returns the observation for the ego agent.
        Update dangerous car
        '''
        ego_pos = self.agents.car_states[0, :2]
        obs = np.zeros((self.num_closet_agents * self.history_len + 1, self.observation_dim), dtype=np.float32)

        closest_idxs = self.agents.get_closest_agents(num_closet_agents=self.num_closet_agents)
        current_idx = 0

        histories = self.agents.get_latest_histories(len=self.history_len)
        for i, idx in enumerate(closest_idxs):
            hist = histories[idx]
            # inverse_location_hist = self.agents.inital_location.copy()
            # inverse_location_hist[0] = WIDTH - inverse_location_hist[0]
            # inverse_location_hist[2] = WIDTH - inverse_location_hist[2]
            dx = ego_pos[0] - hist[:, 0]
            dx = np.minimum(dx, WIDTH - dx)
            dy = ego_pos[1] - hist[:, 1]
            dy = np.minimum(dy, HEIGHT - dy)
            current_idx = i * self.history_len
            obs[current_idx:current_idx + self.history_len, 0] = np.sqrt(dx**2 + dy**2) / DEFAULT_DISTANCE        # normalized
            obs[current_idx:current_idx + self.history_len, 1] = hist[:,3] / MAX_SPEED         # normalized speed
            obs[current_idx:current_idx + self.history_len, 2] = hist[:,2] / 180.0              # normalized angle

        angle_target_ego = Utils.get_angle_by_2points(ego_pos, self.target)
        angle_target_ego = (angle_target_ego - self.agents.car_states[0, 2] % 360) % 180
        # Ego state: distance to target, speed, angle
        obs[self.num_closet_agents * self.history_len, :] = np.linalg.norm(self.target - ego_pos) / DEFAULT_DISTANCE, self.agents.car_states[0, 3] / MAX_SPEED, angle_target_ego / 180, self.current_steps / self.max_steps

        return obs


    def step(self, action):
        '''
            Update the environment with the given action.
            Update number of steps taken by the agent and the environment.
        '''
        angle_delta, acc = action
        self.agents.set_ego_action(angle_delta * self.agents.max_angle_delta, acc * MAX_ACCELERATE)
        self.agents.step(self.is_training)

        if self.agents.check_stand_still(ratio_timesteps=self.current_steps/self.max_steps):
            self.stand_still_num += 1
        else:
            self.agent_steps += 1
            self.stand_still_num = 0
        self.current_steps += 1

        if angle_delta * self.prev_angle > 0:
            self.change_angle = False
        else:
            self.change_angle_num += 1
            self.change_angle = True

        if acc * self.prev_acc > 0:
            self.change_acc = False
        else:
            self.change_acc_num += 1
            self.change_acc = True


        self.prev_angle = angle_delta
        self.prev_acc = acc

        obs = self._get_obs()
        reward, done = self._get_reward()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return obs, reward, done, {}

    def _get_reward(self):
        random_num = np.random.rand()
        ego_pos = self.agents.car_states[0, :2]
        ego_acc = self.agents.car_states[0, 4]
        target_dist = np.linalg.norm(ego_pos - self.target)
        ego_rect = self.agents.rects[0]
        reward = 0.0
        max_reward = self.max_reward

        num_dangerous_agents = 0
        ratio_step = self.current_steps / self.max_steps
        accumulated_ratio_step = self.current_steps * (self.current_steps + 1) / (2*self.max_steps)
        ratio_distance = target_dist / self.initial_distance_to_target

        # Penalty
        # safe_speed = SAFE_SPEED_RANGE[0]*(1-random_num) + random_num * SAFE_SPEED_RANGE[1]
        # expected_distance = self.initial_distance_to_target - safe_speed * self.current_steps
        # if target_dist > expected_distance:
        #     reward -= max_reward


        ##### HANDLE COLLISION #####

        #Collision version 2 :V
        current_dangerous_car_idx = self.agents.get_closest_agents(radius=CarEnv.safe_distance,num_closet_agents=self.num_closet_agents)
        avoid_dangerous_idx = np.setdiff1d(self.prev_dangerous_car_idx, current_dangerous_car_idx)
        new_dangerous_idx = np.setdiff1d(current_dangerous_car_idx, self.prev_dangerous_car_idx)
        union_idx = np.union1d(current_dangerous_car_idx, self.prev_dangerous_car_idx)
        self.prev_dangerous_car_idx = current_dangerous_car_idx # Update prev for next step
        for idx in new_dangerous_idx:
            dist = np.linalg.norm(ego_pos - self.agents.car_states[idx, :2])
            self.current_collide_time += 1
            if ratio_step > 0: reward -= max_reward
            if self.current_collide_time >= self.max_collide_num:
                reward -= max_reward
                print("Collision limit reached at env-step:", self.current_steps, "with ego-step: ", self.agent_steps,",accumalated reward: ", self.acumulated_reward + reward,",change_angle_num:",self.change_angle_num,",change_acc_num:",self.change_acc_num)
                return reward, True
        if len(new_dangerous_idx) > 0: return reward, False

        # closet_idxs = self._get_closest_agents(radius=CarEnv.safe_distance)
        # for idx in closet_idxs:
        #     self.current_collide_time += 1
        #     reward -= max_reward * (self.max_steps / self.current_steps)
        # if self.current_collide_time >= self.max_collide_num:
        #     print("Collision limit reached at env-step:", self.current_steps, " with ego-step: ", self.agent_steps,"and accumalated reward: ", self.acumulated_reward + reward)
        #     return reward, True


        # ratio_step = self.agent_steps / self.max_steps
        # ratio_dist = 1 - target_dist / self.initial_distance_to_target
        # if ratio_step < ratio_dist:
        #     reward += max_reward

        expected_distance = MAX_SPEED / 3 * self.current_steps
        expected_ratio_distance = target_dist / (self.initial_distance_to_target - expected_distance)
        angle_target_ego = (Utils.get_angle_by_2points(ego_pos, self.target)- self.agents.car_states[0, 2] % 360) % 180
        right_dir = angle_target_ego < 10 and self.agents.car_states[0, 3] > 1
        if expected_ratio_distance < 1:
            reward += max_reward * 0.2 * (1 - ratio_distance)
            if right_dir:
                reward += max_reward * 0.1 * (1 - ratio_distance)
        else: 
            if right_dir:
                if ratio_distance > 1:
                    reward += max_reward * 0.05
                else:
                    reward += max_reward * 0.1 * (1 - ratio_distance)
            else: reward -= max_reward * 0.2 * (expected_ratio_distance - 1)
        
        # if ratio_distance - 1 < 0:
        #     reward += 0.01 * max_reward * (1 - ratio_distance)
        # else:
        #     reward -= 0.05 * max_reward

        # if self.change_acc:
        #   reward += max_reward * ratio_step
        # angle_delta = np.abs(self.agents.car_states[0, 2] - self.agents.histories[0, 1, 2])

        # angle_delta = np.abs(np.abs(self.get_target_angle() % 360) - np.abs(self.agents.car_states[0, 2] % 360))
        # if angle_delta > 20:
        #     reward -= max_reward
        

        # self.acumulated_reward += reward
        if target_dist < AGENT_SIZE:
            reward += max_reward * 2
            self.acumulated_reward += reward
            print("Reached target at step:", self.current_steps, "with accumalated reward: ", self.acumulated_reward,",agent step: ",self.agent_steps)
            return 0, True
        self.acumulated_reward += reward

        return reward, False

    def render(self, screen, mode="human"):
        if mode == "human":
            screen.fill((0, 0, 0))
            self.agents.render(screen)

            # Draw target
            target_rect = pygame.Rect(self.target[0], self.target[1], 10, 10)
            pygame.draw.rect(screen, (255, 0, 0), target_rect)
            pygame.display.flip()
        else:
            raise NotImplementedError("Only human render mode is implemented.")

    def get_target_angle(self):
        '''
            Get the angle to the target.
        '''
        target_angle = np.arctan2(self.target[1] - self.ego[1], self.target[0] - self.ego[0])
        return np.degrees(target_angle)
    def _init_essential_items(self):
        self.initial_distance_to_target = np.linalg.norm(
            self.target - np.array(self.agents.car_states[0, :2])
        )
        self.max_reward = 10

        self.current_steps = 0
        self.agent_steps = 0
        self.max_steps = 2048

        self.current_collide_time = 0
        self.stand_still_num = 0
        self.acumulated_reward = 0.0
        self.prev_dangerous_car_idx = []

        self.change_angle = False
        self.prev_angle = -1
        self.change_angle_num = 0
        self.change_acc = False
        self.prev_acc = -1
        self.change_acc_num = 0

        self.gamma = 0.99



class Utils:
    @staticmethod
    def get_angle_by_2points(p1, p2):
        """
        Calculate the angle between two points in degrees.
        """
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = np.arctan2(delta_y, delta_x)
        angle = np.degrees(angle)
        return angle % 360