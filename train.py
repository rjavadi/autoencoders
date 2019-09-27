import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from pointnet_encoder import ClassificationPointNet
from dataset import ShapeNetDataSet
from fastprogress import master_bar, progress_bar


def train(dataset_dir, num_of_points, batch_size, epochs, learning_rate, output_dir):
    train_dataset = ShapeNetDataSet(dataset_dir=dataset_dir, num_of_points=num_of_points)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    test_dataset = ShapeNetDataSet(dataset_dir, num_of_points=num_of_points, train=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, )

    model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'training_log.csv'), 'w+') as fid:
        fid.write('train_loss,test_loss,train_accuracy,test_accuracy\n')

    mb = master_bar(range(epochs))

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

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
            preds, feature_transform = model(points)
            identity = torch.eye(feature_transform.shape[-1])
            if torch.cuda.is_available():
                identity = identity.cuda()
            regularization_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1))
            )
            loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
            epoch_train_loss.append(loss.cpu().item())
            epoch_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            preds = preds.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / float(batch_size)
            epoch_train_acc.append(accuracy)
            mb.child.comment = 'train loss: %f, train accuracy: %f' % (np.mean(epoch_train_loss),
                                                                       np.mean(epoch_train_acc))

        epoch_test_loss = []
        epoch_test_acc = []
        for batch_number, data in enumerate(test_dataloader):
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            model = model.eval()
            preds, feature_transform = model(points)
            loss = F.nll_loss(preds, targets)
            epoch_test_loss.append(loss.cpu().item())
            epoch_test_loss.append(loss.item())
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / float(batch_size)
            mb.write('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
                     % (epoch,
                        np.mean(epoch_train_loss),
                        np.mean(epoch_test_loss),
                        np.mean(epoch_train_acc),
                        np.mean(epoch_test_acc)))
            if test_acc and np.mean(epoch_test_acc) > np.max(test_acc):
                torch.save(model.state_dict(), os.path.join(output_dir, 'shapenet_classification_model.pth'))

            with open(os.path.join(output_dir, 'training_log.csv'), 'a') as fid:
                fid.write('%s,%s,%s,%s,%s\n' % (epoch,
                                                np.mean(epoch_train_loss),
                                                np.mean(epoch_test_loss),
                                                np.mean(epoch_train_acc),
                                                np.mean(epoch_test_acc)))
            train_loss.append(np.mean(epoch_train_loss))
            test_loss.append(np.mean(epoch_test_loss))
            train_acc.append(np.mean(epoch_train_acc))
            test_acc.append(np.mean(epoch_test_acc))

            # implemented in utils using matplot
            # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_dir, 'loss_plot.png'))
            # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_dir, 'accuracy_plot.png'))

if __name__ == '__main__':
    print("main***********")
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('output_dir', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2500, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    train(dataset_dir=args.dataset_folder,
          num_of_points=args.number_of_points,
          batch_size=args.batch_size,
          epochs=args.epochs,
          learning_rate=args.learning_rate,
          output_dir=args.output_dir)
