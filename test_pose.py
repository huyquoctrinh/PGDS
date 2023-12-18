import torch
from pytorchOpenpose.src.model import bodypose_model
from pytorchOpenpose.src import util
import cv2

model_path = "/storage/huytq14/multpersonTracking/SOLIDER-REID/body_pose_model.pth"
model = bodypose_model()
# if torch.cuda.is_available():
    # model = model.cuda()
model_dict = util.transfer(model, torch.load(model_path))
model.load_state_dict(model_dict)
model.eval()
print(model)
x = torch.randn(1, 3, 384, 128)
# x.cuda()
# Let's print it
out = model(x)
print(out[0].shape)
# body_estimation = Body('/storage/huytq14/multpersonTracking/SOLIDER-REID/body_pose_model.pth')
# print(body_estimation)

# img = cv2.imread("/storage/huytq14/multpersonTracking/SOLIDER-REID/train_data/multi_cloth/CelebReID/train/1_2_0.jpg")
# candidate, subset = body_estimation(img)
# print(len(candidate), len(subset))