from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

path = "/home/lawchen/project/robomimic/datasets/lift/ph/image_v141.hdf5"
traj = h5py.File(path, "r")
traj_id = "demo_0"
num_steps = len(traj[f"data/{traj_id}/actions"][()])

# openvla_path = "openvla/openvla-7b"
# openvla_path = "/home/lawchen/project/openvla/logs/openvla-7b+robomimic_lift_dataset+b16+lr-3e-05+lora-r32+dropout-0.0"
openvla_path = "/home/lawchen/project/openvla/logs/openvla-7b+robomimic_lift_dataset+b128+lr-2e-05+lora-r32+dropout-0.0"
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



predicted_actions = []
# loop through each step
for i in range(num_steps):
    # Get image & action
    image = Image.fromarray(traj[f"data/{traj_id}/obs/agentview_image"][()][i]).convert("RGB")
    action = torch.tensor(traj[f"data/{traj_id}/actions"][()][i], dtype=torch.float32).to("cuda:0")

    # Predict Action
    # breakpoint()
    inputs = processor("In: What action should the robot take to lift the cube?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
    pred_action = vla.predict_action(**inputs, do_sample=False, unnorm_key="robomimic_lift_dataset")
    # pred_action = vla.predict_action(**inputs, do_sample=False, unnorm_key="viola")
    print(pred_action)
    predicted_actions.append(pred_action)

# plot predicted actions vs. ground truth actions
predicted_actions = np.stack(predicted_actions)
ground_truth_actions = traj[f"data/{traj_id}/actions"][()]
# plot for each axis
for i in range(7):
    plt.plot(ground_truth_actions[:, i], label=f"Ground Truth {i}")
    plt.plot(predicted_actions[:, i], label=f"Predicted {i}")
    plt.legend()
    plt.show()


