import torch
import numpy as np
from pointnet_encoder import PCAE
import visualize
from configparser import ConfigParser
import os


def load_checkpoint(filepath):
    # TODO: remove number of points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCAE(1024).to(device)
    model.load_state_dict(torch.load(filepath))

    model.eval()

    return model


def predict(point_file, number_of_points):
    point_cloud = np.loadtxt(point_file).astype(np.float32)
    if number_of_points:
        sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
        point_cloud = point_cloud[sampling_indices, :]
    point_cloud = torch.from_numpy(point_cloud)
    point_cloud = point_cloud.unsqueeze(0)
    if torch.cuda.is_available():
        point_cloud = point_cloud.cuda()
    # input = torch.Variable(point_cloud)
    model = load_checkpoint('output/shapenet_classification_model.pth')
    reconstructed = model(point_cloud)
    return reconstructed


if __name__ == "__main__":
    config_file_name = "config.ini"
    config = ConfigParser()

    config.read(config_file_name)
    ouputdir = config['train']['output_dir']
    datasetdir = config['train']['dataset_folder']

    point_file = '03001627/points/1ac6531a337de85f2f7628d6bf38bcc4.pts'
    output = predict(os.path.join(datasetdir, point_file), 1024)
    output = output.view(-1, 3)
    np.savetxt(os.path.join(ouputdir, '1ac6531a337de85f2f7628d6bf38bcc4.pts'), output.detach().cpu().numpy(), delimiter=' ', fmt='%1.4e')
    # visualize.visualize_pcl(point_array=output.detach().cpu().numpy())
    visualize.visualize_pcl(point_file='output/1ac6531a337de85f2f7628d6bf38bcc4.pts')
