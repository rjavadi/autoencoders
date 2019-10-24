import matplotlib.pyplot as plt


# TODO: plot train/test loss using csv output
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()