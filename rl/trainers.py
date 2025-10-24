import time
from rl.loggers import *
from rl.repmem import *
from myutils.filesys import gp


class OffPolicyTrainer:
    __defaults = dict(
        train_ratio=256, batch_size=256, self_expl_start=0, learning_start=None, reward_scale=1.0
    )

    def __init__(self, rep_mem=None, **kwargs):
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        for k, v in self.__defaults.items():        
            kwargs.setdefault(k, v)

        self.train_ratio = kwargs['train_ratio']
        self.batch_size = kwargs['batch_size']
        self.self_expl_start = kwargs['self_expl_start']
        self.learning_start = self.batch_size if kwargs['learning_start'] is None else kwargs['learning_start']
        self.reward_scale = kwargs['reward_scale']

    def train(self, env, agent, step_budget, path, log_points=1000, log_trajs=False):
        print(f"Start training {agent.__class__.__name__}. Log path: {path}")
        self.__reset()

        # Create loggers
        log_itv = step_budget // log_points
        base_logger = CsvLogger(gp(path, 'agent_log.csv'), agent.log_keys, log_itv)
        traj_logger = TrajectoryLogger(path) if log_trajs else None

        steps, replayed = 0, 0
        start_time = time.time()
        agent_time = 0
        o, _ = env.reset()
        while steps < step_budget:
            steps += 1
            completion_rate = steps / step_budget
            t = time.perf_counter()
            a = agent.make_decision(o) if steps >= self.self_expl_start else env.action_space.sample()
            agent_time += time.perf_counter() - t
            op, r, d, truncat, _ = env.step(a)
            r *= self.reward_scale
            self.rep_mem.add(o, a, r, op, d)
            if traj_logger is not None:
                traj_logger.update(steps, o, a, r, op, d or truncat)
            able2update = len(self.rep_mem) > self.batch_size and steps >= self.learning_start
            while able2update and replayed < (steps - self.learning_start) * self.train_ratio:
                batch, importance = self.rep_mem.sample(self.batch_size)
                t = time.perf_counter()
                agent_log = agent.update(batch, importance)
                agent_time += time.perf_counter() - t
                if self.rep_mem.__class__.__name__ == 'PERMem':
                    self.rep_mem.update(agent, completion_rate)
                base_logger.update(steps, time.time() - start_time, agent_time, agent_log)
                replayed += 1

            yield steps, time.time() - start_time, agent_time
            o = op
            agent.step_callback(completion_rate=completion_rate)
            if d or truncat:
                o, _ = env.reset()
                agent.ep_callback(completion_rate=completion_rate)
        base_logger.close(steps, time.time() - start_time, agent_time)
        agent.save(gp(f'{path}/final.pt'))

    def __reset(self):
        self.rep_mem.clear()


