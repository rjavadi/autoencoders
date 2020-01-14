import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import torch
import torch.optim as optim
import numpy as np
import argparse
from configparser import ConfigParser
from model.pointnet_encoder import PCAE
from dataset import ShapeNetDataSet
from fastprogress import master_bar, progress_bar
from chamfer_distance import ChamferDistance
from torch.utils.tensorboard import SummaryWriter
import datetime


config_file_name = "config.ini"

current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

summary_writer = SummaryWriter(log_dir='logs/shape-ae/' + current_time)


def train(dataset_dir, num_of_points, batch_size, epochs, learning_rate, output_dir):
    train_dataset = ShapeNetDataSet(dataset_dir=dataset_dir, num_of_points=num_of_points)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    test_dataset = ShapeNetDataSet(dataset_dir, num_of_points=num_of_points, train=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, )


    model = PCAE(num_of_points)
    print(model)
    ch_distance = ChamferDistance()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'training_log.csv'), 'w+') as fid:
        fid.write('epoch,train_loss,test_loss\n')

    mb = master_bar(range(epochs))

    train_loss = []
    test_loss = []

    for epoch in mb:
        epoch_train_loss = []
        epoch_train_acc = []
        batch_number = 0
        for data in progress_bar(train_dataloader, parent=mb):
            batch_number += 1
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            if points.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            model = model.train()
            reconstructed = model(points)
            rand = np.random.randint(0, 100)
            if rand < 10:
                np.savetxt(os.path.join(output_dir, str(batch_number) + str(epoch) + '_train.pts'),
                           points[0].detach().cpu().numpy(),
                           delimiter=' ', fmt='%1.4e')

            dist1, dist2 = ch_distance(points, reconstructed)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            epoch_train_loss.append(loss.cpu().item())
            epoch_train_loss.append(loss.item())
            summary_writer.add_scalar('training loss',
                              loss.item(),
                              epoch * len(train_dataloader) + batch_number)

            loss.backward()
            optimizer.step()
            mb.child.comment = 'train loss: %f, train accuracy: %f' % (np.mean(epoch_train_loss),
                                                                       np.mean(epoch_train_acc))

        epoch_test_loss = []
        for batch_number, data in enumerate(test_dataloader):
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()


            model = model.eval()
            reconstructed = model(points)
            dist1, dist2 = ch_distance(points, reconstructed)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            if loss > 0.5:
                np.savetxt(os.path.join(output_dir, str(batch_number) + str(epoch) + '_val_ground.pts'),
                           points.detach().cpu().numpy(),
                           delimiter=' ', fmt='%1.4e')
                np.savetxt(os.path.join(output_dir, str(batch_number) + str(epoch) + '_val_constructed.pts'),
                           reconstructed.detach().cpu().numpy(),
                           delimiter=' ', fmt='%1.4e')
            epoch_test_loss.append(loss.cpu().item())
            epoch_test_loss.append(loss.item())
            mb.write('Epoch %s: train loss: %s, val loss: %s'
                     % (epoch,
                        np.mean(epoch_train_loss),
                        np.mean(epoch_test_loss)))
            summary_writer.add_scalar('validation loss',
                                      loss.item(),
                                      epoch * len(test_dataloader) + batch_number)
            if test_loss and np.mean(epoch_test_loss) < np.min(test_loss):
                torch.save(model.state_dict(), os.path.join(output_dir, 'shapenet_classification_model.pth'))

            with open(os.path.join(output_dir, 'training_log.csv'), 'a') as fid:
                fid.write('%s,%s,%s\n' % (epoch,
                                          np.mean(epoch_train_loss),
                                          np.mean(epoch_test_loss)))
            train_loss.append(np.mean(epoch_train_loss))
            test_loss.append(np.mean(epoch_test_loss))

if __name__ == '__main__':
    print("main***********")
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('output_dir', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=1024, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    #set values in conifg file
    cfgfile = open(config_file_name, 'r+')
    config = ConfigParser()
    config.read(config_file_name)
    config.set('train', 'dataset_folder', args.dataset_folder)
    config.set('train', 'output_dir', args.output_dir)
    config.set('train', 'number_of_points', str(args.number_of_points))
    config.set('train', 'batch_size', str(args.batch_size))
    config.set('train', 'epochs', str(args.epochs))
    config.set('train', 'learning_rate', str(args.learning_rate))
    config.write(cfgfile)
    cfgfile.close()




    train(dataset_dir=args.dataset_folder,
          num_of_points=args.number_of_points,
          batch_size=args.batch_size,
          epochs=args.epochs,
          learning_rate=args.learning_rate,
          output_dir=args.output_dir)
