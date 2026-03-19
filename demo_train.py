from score_models.score_model import ScoreModel
from score_models.architectures import DDPM
from pytorch_dataset_demo import CustomImageDataset

dataset = CustomImageDataset(annotations_file='/projects/bfpq/work/aberres2/demo_roman2/annotations_hot_1000.csv', img_dir='/projects/bfpq/work/aberres2/demo_roman2/data')
checkpoints_directory = '/projects/bfpq/rubin2roman/checkpoints/demo5'

net = DDPM(channels=1)
model = ScoreModel(model=net, sigma_min=1e-4, sigma_max=500, device="cuda")

model.fit(dataset, epochs=100, batch_size=2, learning_rate=1e-4, checkpoints_directory = checkpoints_directory)