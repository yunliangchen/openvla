from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from peft import PeftModel



openvla_path_base = "openvla/openvla-7b"
openvla_path_base = "/home/lawchen/project/openvla/logs/openvla-7b+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step_old"
openvla_path = "/home/lawchen/project/openvla/logs/openvla-7b+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step"
adapter_dir = "/home/lawchen/project/openvla/openvla/adapter-tmp/openvla-7b+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step"
# breakpoint()
merge_weights = True
if merge_weights:
    base_vla = AutoModelForVision2Seq.from_pretrained(
                        openvla_path_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(openvla_path)

# breakpoint()




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
    "icrl_vla_tfds",
    batch_transform,
    resize_resolution=tuple([224, 224]),
    shuffle_buffer_size=100_000,
    image_aug=True,
)

episodic_dataset, num_trajectories, episodic_dataset_stats = vla_dataset.make_dataset(rlds_config=vla_dataset.rlds_config)
for episode in episodic_dataset:
    print(episode.keys()) # dict_keys(['observation', 'task', 'action', 'dataset_name', 'absolute_action_mask'])
    # print(episode["action"]) # shape (T, window_size + future_action_window_size, 7)
    # print(episode["observation"]["image_primary"]) # shape (T, 1, 224, 224, 3) uint8
    # print(episode["observation"]["proprio"]) # shape (T, 1, 8) uint8
    # print(episode["task"]["language_instruction"]) # shape (T,)

    # breakpoint()

    num_steps = episode["action"].shape[0]
    predicted_actions = []
    # loop through each step
    for i in range(num_steps):
        # Get image & action
        image = Image.fromarray(episode["observation"]["image_primary"][i][0].numpy()).convert("RGB")
        wrist_image = Image.fromarray(episode["observation"]["image_wrist"][i][0].numpy()).convert("RGB")
        action = torch.tensor(episode["action"][i, 0, :].numpy(), dtype=torch.float32).to("cuda:0")
        proprio = episode["observation"]["proprio"].numpy()

        # Predict Action
        # breakpoint()
        if "proprio" in openvla_path:
            if "wrist" in openvla_path:
                inputs = processor(f"In: The robot gripper quaternion is {proprio}. What action should the robot take to lift the cube?\nOut:", [image, wrist_image]).to("cuda:0", dtype=torch.bfloat16)
            else:
                inputs = processor(f"In: The robot gripper quaternion is {proprio}. What action should the robot take to lift the cube?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
        else:
            if "wrist" in openvla_path:
                print("here")
                inputs = processor(f"In: What action should the robot take to {episode['task']['language_instruction'][0].numpy().decode('utf-8')}?\nOut:", [image, wrist_image]).to("cuda:0", dtype=torch.bfloat16)
            else:
                inputs = processor("In: What action should the robot take to lift the cube?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
        # breakpoint()
        pred_action = vla.predict_action(**inputs, do_sample=False, unnorm_key="icrl_vla_tfds")
        
        # predicted gripper action is 1 for open and 0 for close
        pred_action[6] = pred_action[6]

        print(pred_action)
        predicted_actions.append(pred_action)

    # plot predicted actions vs. ground truth actions
    predicted_actions = np.stack(predicted_actions)
    ground_truth_actions = episode["action"][:, 0, :]

    # remember: /home/lawchen/project/openvla/openvla/prismatic/vla/datasets/rlds/dataset.py:L242-249 normalize_action_and_proprio comment out
    # otherwise the loaded groundtruth action will be normalized between -1 and 1

    # plot for each axis
    for i in range(7):
        plt.plot(ground_truth_actions[:, i], label=f"Ground Truth {i}")
        plt.plot(predicted_actions[:, i], label=f"Predicted {i}")
        plt.legend()
        plt.show()
        # plt.savefig(f"test_{i}.png")
        # plt.clf()


