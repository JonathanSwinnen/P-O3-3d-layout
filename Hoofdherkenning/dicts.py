import torch
i = 100
model = torch.load("./saved_models/PO3_v3/training_" + str(i) + ".pth")
torch.save(model.state_dict(), "./saved_models/PO3_v3/dict_training_" + str(i) + ".pth")
