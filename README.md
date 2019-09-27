https://github.com/romaintha/pytorch_pointnet

Training

Use the following script for training

``python train.py dataset dataset_folder output_folder 
   --number_of_points 2048 
   --batch_size 32
   --epochs 50
   --learning_rate 0.001
   ``
   
where:

`dataset`: is one of the available datasets (e.g. shapenet)

`dataset_folder`: is the path to the root dataset folder


`output_folder`: is the output_folder path where the training logs and model checkpoints will be stored

`number_of_points`: is the amount of points per cloud

`batch_size`: is the batch size

`epochs`: is the number of training epochs

`learning_rate`: is the optimizer learning rate

`model_checkpoint`: is the path to a checkpoint that is loaded before the training begins.