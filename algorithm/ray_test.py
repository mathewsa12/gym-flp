# Import the RL algorithm (Algorithm) we would like to use.
from gym_flp.envs import OfpEnv

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray import tune, air


env_creator = {
    'instance': 'P6'
}

def env_creator(env_config):
    return OfpEnv()  # return an env instance

register_env("flp", env_creator)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["horizon"] = 32
config["log_level"] = 'INFO'
config['framework'] = 'torch'
config["evaluation_interval"]=1000
config["evaluation_num_workers"]=1
config["keep_per_episode_custom_metrics"]=True
algo = ppo.PPO(config=config, env="flp")

# Can optionally call algo.restore(path) to load a checkpoint.
'''
for i in range(1):
    print(i)
   # Perform one iteration of training the policy with PPO
    result = algo.train()
    print(pretty_print(result))
'''
stop = {
        "timesteps_total": 1e5,
    }

results = tune.Tuner(
        'PPO', param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)
    ).fit()

algo.evaluate()
ray.shutdown()