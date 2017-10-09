from network import Network
from utils import LOG_INFO, beautiful_dict
from layers import Relu, Sigmoid, Linear
# from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from scheduler import EmptyScheduler, StepScheduler
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='2-layer',
        help="""\
        The model used to train\
        """
    )
    parser.add_argument(
        '--name',
        type=str,
        default='model',
        help="""\
        Model name\
        """
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help="""\
        Initial learning rate\
        """
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.0,
        help="""\
        Momentum\
        """
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=128,
        help="""\
        Batch size\
        """
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        help="""\
        Epoch number\
        """
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0,
        help="""\
        Weight decay\
        """
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='empty',
        help="""\
        Learning rate scheduler\
        """
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        help="""\
        Activation function\
        """
    )
    args = parser.parse_args()

    # use model name to determine log filename and plot filename
    model_name = args.name
    log_file = model_name + '.log'
    loss_plot_file = model_name + '_loss.png'
    acc_plot_file = model_name + '_acc.png'

    # read train and test data
    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    # Your model defintion here
    # You should explore different model architecture
    model_str = args.model
    activation_str = args.activation
    model = Network()
    # choose different model based on command line argument
    if model_str == '2-layer':
        model.add(Linear('fc1', 784, 512, 0.01))
        model.add(Relu('relu1') if activation_str == 'relu' else Sigmoid('sigmoid1'))
        model.add(Linear('fc2', 512, 10, 0.01))
    elif model_str == '3-layer':
        model.add(Linear('fc1', 784, 512, 0.01))
        model.add(Relu('relu1') if activation_str == 'relu' else Sigmoid('sigmoid1'))
        model.add(Linear('fc2', 512, 32, 0.01))
        model.add(Relu('relu2') if activation_str == 'relu' else Sigmoid('sigmoid2'))
        model.add(Linear('fc3', 32, 10, 0.01))
    else:
        raise Exception('Model named {} not found.'.format(model_str))
    # write model architecture to log file
    LOG_INFO('Model architecture:\r\n' + str(model) + '\r\n', time=False, to_file=log_file)

    # use euclidean loss function
    loss = EuclideanLoss(name='loss')
    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.
    # initialize training configuration based on command line arguments
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.wd,
        'momentum': args.momentum,
        'batch_size': args.batch,
        'max_epoch': args.epoch,
        'disp_freq': 0,
        'test_epoch': 1
    }
    # write configurations to log file
    LOG_INFO('Configurations:\r\n' + beautiful_dict(config) + '\r\n', time=False, to_file=log_file)

    # choose learning rate scheduler based on command line argument
    scheduler_str = args.scheduler
    if scheduler_str == 'empty':
        # EmptyScheduler does nothing to learning rate
        scheduler = EmptyScheduler(config['learning_rate'])
    elif scheduler_str == 'step':
        # StepScheduler decay learning rate by 0.1 every 10 epochs
        scheduler = StepScheduler(config['learning_rate'], step_size=50, decay=0.5, min_lr=1e-4)
    else:
        raise Exception('Scheduler named {} not found.'.format(scheduler_str))

    # loss list for plot
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # start training procedure
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch), to_file=log_file)
        config['learning_rate'] = scheduler.step(epoch)
        train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        msg = '    Training, total mean loss %.5f, total acc %.5f' % (train_loss, train_acc)
        LOG_INFO(msg, to_file=log_file)

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch), to_file=log_file)
            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            msg = '    Testing, total mean loss %.5f, total acc %.5f' % (test_loss, test_acc)
            LOG_INFO(msg, to_file=log_file)

    # plot train and test loss using matplotlib
    x = range(1, config['max_epoch'] + 1)
    plt.title('Train/Test Loss v.s. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, train_loss_list, 'r', label='Train')
    plt.plot(x, test_loss_list, 'b', label='Test')
    plt.legend(loc='upper right')
    plt.savefig(loss_plot_file)
    plt.clf()
    plt.title('Train/Test Accuracy v.s. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(x, train_acc_list, 'r', label='Train')
    plt.plot(x, test_acc_list, 'b', label='Test')
    plt.legend(loc='upper left')
    plt.savefig(acc_plot_file)