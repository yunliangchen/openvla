from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

path = "/home/lawchen/project/droid/data/tiger_pick_sample_dataset/Thu_Jan_11_21:30:17_2024/trajectory_im128.h5"
traj = h5py.File(path, "r")

num_steps = len(traj["action/cartesian_velocity"][()])

# openvla_path = "openvla/openvla-7b"
# openvla_path = "/home/lawchen/project/openvla/logs/openvla-7b+robomimic_lift_dataset+b16+lr-3e-05+lora-r32+dropout-0.0"
openvla_path = "/home/lawchen/project/openvla/logs/openvla-7b+r2_d2+b16+lr-2e-05+lora-r32+dropout-0.0"
# Load Processor & VLA
processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     openvla_path, 
#     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True, 
#     trust_remote_code=True
# ).to("cuda:0")


# # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
# with open(Path(openvla_path) / "dataset_statistics.json", "r") as f:
#     vla.norm_stats = json.load(f)


# predicted_actions = []
# # loop through each step
# for i in range(num_steps):
#     # Get image & action
#     image = Image.fromarray(traj[f"observation/camera/image/varied_camera_1_left_image"][()][i]).convert("RGB")
#     action = torch.tensor(traj[f"action/cartesian_velocity"][()][i], dtype=torch.float32).to("cuda:0")

#     # Predict Action
#     # breakpoint()
#     inputs = processor("In: What action should the robot take to put the toy tiger into the bowl?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
#     pred_action = vla.predict_action(**inputs, do_sample=False, unnorm_key="r2_d2")
#     # pred_action = vla.predict_action(**inputs, do_sample=False, unnorm_key="viola")
#     print(pred_action)
#     predicted_actions.append(pred_action)

# # plot predicted actions vs. ground truth actions
# predicted_actions = np.stack(predicted_actions)
# ground_truth_actions = traj["action/cartesian_velocity"][()]
# # plot for each axis
# for i in range(6):
#     plt.plot(ground_truth_actions[:, i], label=f"Ground Truth {i}")
#     plt.plot(predicted_actions[:, i], label=f"Predicted {i}")
#     plt.legend()
#     plt.show()





from prismatic.vla.datasets import EpisodicRLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
import imageio
# Create Action Tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)




batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder if "v01" not in openvla_path else VicunaV15ChatPromptBuilder,
)

vla_dataset = EpisodicRLDSDataset(
    "/home/lawchen/tensorflow_datasets/",
    "r2_d2",
    batch_transform,
    resize_resolution=tuple([224, 224]),
    shuffle_buffer_size=100_000,
    image_aug=True,
)

episodic_dataset, num_trajectories, episodic_dataset_stats = vla_dataset.make_dataset(rlds_config=vla_dataset.rlds_config)
for episode in episodic_dataset:
    # breakpoint()
    print(episode.keys()) # dict_keys(['observation', 'task', 'action', 'dataset_name', 'absolute_action_mask'])
    # print(episode["action"]) # shape (T, 1, 7)
    # print(episode["observation"]["image_primary"]) # shape (T, 1, 224, 224, 3) uint8
    # print(episode["task"]["language_instruction"]) # shape (T,)

    # create gif
    images = []
    for i in range(episode["observation"]["image_primary"].shape[0]):
        images.append(episode["observation"]["image_primary"][i][0])
    imageio.mimsave('movie.gif', images)
    
    # plot action
    for i in range(6):
        plt.plot(episode["action"][:, 0, i], label=f"Ground Truth {i}")
    plt.legend()
    plt.show()
