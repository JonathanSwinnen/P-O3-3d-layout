import torch
import tv_training_code


model = tv_training_code.PennFudanDataset()
model.load_state_dict(torch.load("pedestrian_model.pt"))
model.eval()