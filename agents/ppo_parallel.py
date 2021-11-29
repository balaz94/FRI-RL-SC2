import torch
import torch.nn.functional as F
import numpy as np
from utils.stats import write_to_file
from random import randint
import datetime
from torch.multiprocessing import Process, Pipe

def worker(connection, env_info, env_func, count_of_envs, count_of_iterations, count_of_steps, device, gamma, lam = 0.95):
    envs = [env_func(*env_info) for _ in range(count_of_envs)]
    game_score = [0 for _ in range(count_of_envs)]
    observations = torch.stack([torch.from_numpy(env.reset()).float() for env in envs])
    _, dim1, dim2, dim3 = observations.shape

    mem_observations = torch.zeros(count_of_steps, count_of_envs, dim1, dim2, dim3)
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype = torch.long)
    mem_pred_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        score_of_end_games = []

        for step in range(count_of_steps):
            connection.send(observations)
            logits, values = connection.recv()

            mem_observations[step] = observations.clone()
            probs, log_probs = F.softmax(logits, dim = -1), F.log_softmax(logits, dim = -1)
            actions = probs.multinomial(num_samples=1)
            mem_actions[step] = actions.clone()
            mem_log_probs[step] = log_probs.gather(1, actions).clone()
            mem_pred_values[step] = values.clone()

            observations = []
            for i in range(count_of_envs):
                obs, reward, terminal, _ = envs[i].step(actions[i].item())
                mem_rewards[step, i, 0] = reward
                game_score[i] += reward

                if terminal:
                    mem_non_terminals[step, i] = 0
                    score_of_end_games.append(game_score[i])
                    game_score[i] = 0
                    obs = envs[i].reset()
                observations.append(torch.from_numpy(obs).float())
            observations = torch.stack(observations)

        connection.send(observations)
        mem_pred_values[step + 1] = connection.recv()

        mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
        advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        target_values = torch.zeros((count_of_steps, count_of_envs, 1))
        t_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_rewards[step] + gamma * mem_pred_values[step + 1] * mem_non_terminals[step] - mem_pred_values[step]
            t_gae = delta + gamma * lam * t_gae * mem_non_terminals[step]
            target_values[step] = t_gae + mem_pred_values[step]
            advantages[step] = t_gae.clone()

        connection.send([mem_observations.clone(), mem_actions.clone(), mem_log_probs.clone(), target_values.clone(), advantages.clone(), score_of_end_games])
    connection.recv()
    connection.close()

class Agent:
    def __init__(self, model, gamma = 0.99, entropy_loss_coef = 0.001, value_loss_coef = 0.5, epsilon = 0.1, lr = 0.00025,
                 name = 'ppo', optim = 'Adam', device = 'cpu', results_path = 'results/ppo/pong/'):
        self.device = device
        self.model = model
        self.model.to(device)

        if optim == 'Adam':
            print('optimizer: Adam')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        elif optim == 'SGD':
            print('optimizer: SGD wiht momentum = 0.9')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)
        elif optim == 'RMS':
            print('optimizer: RMSProp')
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr)
        else:
            print('optimizer: SGD wiht momentum = 0.0')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)

        self.gamma = gamma
        self.entropy_loss_coef = entropy_loss_coef
        self.value_loss_coef = value_loss_coef

        self.upper_bound = 1 + epsilon
        self.lower_bound = 1 - epsilon

        self.lr = lr

        self.name = name
        self.results_path = results_path

    def train(self, env_info, env_func, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs = 4, first_iteration = 0, input_dim = (4, 80, 80)):
        print('Training is starting')
        logs = 'iteration,episode,avg_score,best_score,best_avg_score'
        logs_losses = 'iteration,episode,policy,value,entropy'

        count_of_steps_per_iteration = count_of_processes * count_of_steps * count_of_envs
        count_of_losses = count_of_steps_per_iteration * count_of_epochs / batch_size
        scores, best_avg_score, best_score, count_of_episodes = [], -1e9, -1e9, 0

        processes, connections = [], []
        for _ in range(count_of_processes):
            parren_connection, child_connection = Pipe()
            process = Process(target = worker, args = (child_connection, env_info, env_func, count_of_envs, count_of_iterations, count_of_steps, self.device, self.gamma))
            connections.append(parren_connection)
            processes.append(process)
            process.start()

        start = datetime.datetime.now()

        for iteration in range(first_iteration, count_of_iterations):
            for step in range(count_of_steps):
                observations = []
                for coonection in connections:
                    obs = coonection.recv()
                    observations.append(obs.clone())

                with torch.no_grad():
                    logits, values = self.model(torch.stack(observations).to(self.device).view(-1, *input_dim))
                    dim0, dim1 = logits.shape
                    logits, values = logits.view(-1, count_of_envs, dim1).cpu(), values.view(-1, count_of_envs, 1).cpu()

                for conn_idx in range(count_of_processes):
                    connections[conn_idx].send([logits[conn_idx], values[conn_idx]])

            observations = [connection.recv() for connection in connections]
            with torch.no_grad():
                _, values = self.model(torch.stack(observations).to(self.device).view(-1, *input_dim))
                values = values.view(-1, count_of_envs, 1).cpu()
            for conn_idx in range(count_of_processes):
                connections[conn_idx].send(values[conn_idx])

            if iteration > 0 and iteration % 1000 == 0:
                torch.save(self.model.state_dict(), self.results_path + 'models/' + self.name + '_' + str(iteration) + '.pt')

            mem_observations, mem_actions, mem_log_probs, mem_target_values, mem_advantages, end_games = [], [], [], [], [], []

            for connection in connections:
                observations, actions, log_probs, target_values, advantages, score_of_end_games = connection.recv()
                
                mem_observations.append(observations)
                mem_actions.append(actions)
                mem_log_probs.append(log_probs)
                mem_target_values.append(target_values)
                mem_advantages.append(advantages)
                end_games.extend(score_of_end_games)

            count_of_end_games = len(end_games)
            if count_of_end_games > 0:
                new_score = True
                count_of_episodes += count_of_end_games
                scores.extend(end_games)
                best_score = max(best_score, np.max(end_games))

                length = len(scores)
                if length > 100:
                    scores = scores[length - 100:]
                avg_score = np.average(scores)
                best_avg_score = max(best_avg_score, avg_score)
                logs += '\n' + str(iteration) + ',' + str(count_of_episodes) + ',' + str(avg_score) + ',' + str(best_score) + ',' + str(best_avg_score)
                if iteration % 100 == 0:
                    print('iteration', iteration, '\tepisode', count_of_episodes, '\tavg score', avg_score, '\tbest score', best_score, '\tbest avg score', best_avg_score)

            mem_observations = torch.stack(mem_observations).to(self.device).view((-1, ) + input_dim)
            mem_actions = torch.stack(mem_actions).to(self.device).view(-1, 1)
            mem_log_probs = torch.stack(mem_log_probs).to(self.device).view(-1, 1)
            mem_target_values = torch.stack(mem_target_values).to(self.device).view(-1, 1)
            mem_advantages = torch.stack(mem_advantages).to(self.device).view(-1, 1)
            mem_advantages = (mem_advantages - torch.mean(mem_advantages)) / (torch.std(mem_advantages) + 1e-5)

            sum_policy_loss, sum_value_loss, sum_entropy_loss = 0, 0, 0
            for epoch in range(count_of_epochs):
                perm_indices = torch.randperm(count_of_steps_per_iteration, device = self.device).view(-1, batch_size)
                for indices in perm_indices:
                    logits, values = self.model(mem_observations[indices])
                    probs, log_probs = F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[indices])
                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    value_loss = F.mse_loss(values, mem_target_values[indices])

                    adv = mem_advantages[indices]
                    ratio = torch.exp(new_log_probs - mem_log_probs[indices])
                    surr_policy = ratio * adv
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) * adv

                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    sum_policy_loss += policy_loss.item()
                    sum_value_loss += value_loss.item()
                    sum_entropy_loss += entropy_loss.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_loss_coef * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            logs_losses +=  '\n' + str(iteration) + ',' + str(len(scores)) + ',' + str(sum_policy_loss / count_of_losses) + ',' + str(sum_value_loss / count_of_losses) + ',' + str(sum_entropy_loss / count_of_losses)
            if iteration % 10 == 0:
                write_to_file(logs, self.results_path + 'data/' + self.name + '.txt')
                write_to_file(logs_losses, self.results_path + 'data/' + self.name + '_loss.txt')

        print('training', datetime.datetime.now() - start)

        for connection in connections:
            connection.send(1)
        [process.join() for process in processes]
