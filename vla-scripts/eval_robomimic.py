import argparse
import json
import numpy as np
import time
import tyro 
import yaml
import os
import sys
import traceback

from collections import OrderedDict

from pathlib import Path
from typing import Union

import torch
import torch.backends.cudnn as cudnn

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import robomimic.utils.tensor_utils as TensorUtils
import torch.nn as nn
from torchvision import transforms

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import json
from peft import PeftModel


# import kinpy

# Max_Gripper_Width = 0.085
# chain = kinpy.build_chain_from_urdf(open("assets/franka/panda.urdf").read())


class OpenVLARolloutPolicy(RolloutPolicy):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, processor, unnorm_key="robomimic_lift_dataset_sample", use_proprio=False,
                 image_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                 proprio_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos','robot0_joint_pos'],
                 rot_6d=False,
                 ):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        super().__init__(policy)
        self.policy = policy
        self.processor = processor
        self.unnorm_key = unnorm_key
        self.use_proprio = use_proprio

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        # self.policy.reset(self.action_exec_horizon)
        pass
        
        


    def _prepare_observation(self, ob, add_eos=True):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        
        
        return ob

    def _update_rot_6d(self, ob):
        """
        update proprio and action to 6d rotation
        """
        # breakpoint()
        ob["proprio"] = np.concatenate([ob["proprio"][:, :3], quat_to_rot_6d(ob["proprio"][:, 3:7]), ob["proprio"][:, 7:]], axis=-1)
        if "action" in ob and ob["action"] is not None:
            ob["action"] = np.concatenate([ob["action"][:, :3], euler_to_rot_6d(ob["action"][:, 3:6]), ob["action"][:, 6:]], axis=-1)
            print("converted 6d action (should be same as predicted action): ", ob["action"])
        else:
            print("no action in observation")
        return ob


    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
            goal (dict): goal observation
        """
        # for k in ob:
        #     ob[k] = ob[k][None]
        # ob = self._prepare_observation(ob)
        # if goal is not None:
        #     goal = self._prepare_observation(goal)
        
        # Get image & action
        # breakpoint()
        # reshape and resize from 3, 84, 84 to 256, 256, 3
        import cv2
        img = ob["agentview_image"]
        img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (256, 256))
        cv2.imwrite("test.png", 255 * img)
        # from float to int
        img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img).convert("RGB")

        wrist_image = ob["robot0_eye_in_hand_image"]
        wrist_image = np.transpose(wrist_image, (1, 2, 0))
        wrist_image = cv2.resize(wrist_image, (256, 256))
        wrist_image = (wrist_image * 255).astype(np.uint8)
        wrist_image = Image.fromarray(wrist_image).convert("RGB")


        # Predict Action
        # breakpoint()
        if self.use_proprio and False:
            proprio = np.concatenate((ob['robot0_eef_pos'], ob['robot0_eef_quat'], ob['robot0_gripper_qpos'][[0]])).astype(np.float32)
            # create random proprio
            proprio = np.random.rand(8).astype(np.float32) * 0
            print("random proprio: ", proprio)
            inputs = self.processor(f"In: The robot gripper quaternion is {proprio}. What action should the robot take to lift the cube?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
        else:
            inputs = self.processor("In: What action should the robot take to lift the cube?\nOut:", [image, wrist_image]).to("cuda:0", dtype=torch.bfloat16)
            # inputs = self.processor("In: What action should the robot take to lift the cube?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
        pred_action = self.policy.predict_action(**inputs, do_sample=False, unnorm_key=self.unnorm_key)
        
        # predicted gripper action is 1 for open and 0 for close
        # robomimic action is -1 for open and 1 for close
        pred_action[6] = (1 - pred_action[6]) * 2 - 1
        print(pred_action)

        # print("predicted action: ", pred_action)
        action = TensorUtils.to_numpy(pred_action)

        return action

    

def run(config, env_meta, task, output_dir, rollout_model):
    """
    Run the policy from model in the environment.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    
    # print(config)
    log_dir = os.path.join(output_dir, task+ "_logs")
    os.makedirs(log_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, task+ "_videos")
    os.makedirs(video_dir, exist_ok=True)

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    # create environments for validation runs
    env_names = [env_meta["env_name"]]



    for env_name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name, 
            render=False, 
            render_offscreen=config.experiment.render_video,
            use_image_obs=True,
            use_depth_obs=False,
        )
        envs[env.name] = env
        print(envs[env.name])


    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # print("\n============= Model Summary =============")
    # print(model)  # print model summary
    # print("")


    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    
    num_episodes = config.experiment.rollout.n
    epoch = 1
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
    )
    
    # summarize results from rollouts to tensorboard and terminal
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        for k, v in rollout_logs.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
            else:
                data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

        print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
        print('Env: {}'.format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))


    # terminate logging
    data_logger.close()




def main(
    openvla_path : Union[str, Path],
    n_episodes: int = 2,
    unnorm_key="robomimic_lift_dataset_sample",
    data_config: str = "/home/lawchen/project/icrl/icrl_lite/config/robomimic_eval.json",
    task_name : str = "lift", 
    ):
    
    """Plotting the predicted actions vs the ground truth actions

    Args: 
        train_yaml_path: str, path to the yaml file containing the training configuration
        openvla_path: str, path to the folder containig the checkpoint
        task_name: str, one of ["all", "square", "can", "lift"]
    """
    # creating the output directory and logging directory for test
    output_dir = os.path.join(openvla_path, "test_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Logging results to ", output_dir)
    
    


    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        openvla_path, 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")


    # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
    with open(Path(openvla_path) / "dataset_statistics.json", "r") as f:
        vla.norm_stats = json.load(f)

    
    if "proprio" in openvla_path:
        use_proprio = True
    else:
        use_proprio = False

    rollout_model = OpenVLARolloutPolicy(vla, processor, unnorm_key=unnorm_key, use_proprio=use_proprio)

    
    ext_cfg = json.load(open(data_config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.unlocked():
        config.update(ext_cfg)
        

    config.train.data = ["/home/lawchen/project/robomimic/datasets/lift/ph/lift_robomimic_icrl_format_256.hdf5"]


    # override config from args
    config.train.cuda = True
    config.train.seq_length = 1
    config.train.hdf5_cache_mode = None
    config.train.output_dir = output_dir
    config.train.seed = 0
    config.experiment.rollout.n = n_episodes
    config.experiment.rollout.horizon = 150

    print("\n============= Loaded Environment Metadata =============")
    with open('/home/lawchen/project/icrl/dataset/env_attr_square_lift_can.json') as json_data:
        env_metas = json.load(json_data)
    tasks = list(env_metas.keys())
    if task_name == "all":
        tasks = tasks
    else:
        assert task_name in tasks, f"Task {task_name} not found in {tasks}"
        tasks = [task_name]

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

  
    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    for task in tasks:
        print("\n============= Running Task {} =============".format(task))
        # catch error during training and print it
        try:
            run(config, env_metas[task], 'robomimic_'+task, output_dir, rollout_model)
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla_path", type=str, required=True, help="path to the openvla model")
    parser.add_argument("--n_episodes", type=int, default=2, help="number of episodes to run")
    parser.add_argument("--unnorm_key", type=str, default="robomimic_lift_dataset_subsampled50_cams", help="key to use for unnorm")
    parser.add_argument("--data_config", type=str, default="/home/lawchen/project/icrl/icrl_lite/config/robomimic_eval.json", help="path to the data config")
    parser.add_argument("--task_name", type=str, default="lift", help="task name to run")
    args = parser.parse_args()
    main(args.openvla_path, args.n_episodes, args.unnorm_key, args.data_config, args.task_name)
