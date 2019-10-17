import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt

def run_logistic_regression(l=0, type='train'):
    if type == 'train':
        train_inputs, train_targets = load_train()
    elif type=='train_small':
        train_inputs, train_targets = load_train_small()
    else:
        return
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.05,
                    'weight_regularization': True,
                    'num_iterations': 1000,
                    'weight_decay': l
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.uniform(-0.5,0.5,(M + 1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)


    # Begin learning with gradient descent
    ces_train = []
    ces_valid = []
    xs = []

    acc_train = []
    acc_valid = []
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N
        # weights = weights - hyperparameters['learning_rate'] * df
        
        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        xs.append(t)
        ces_train.append(cross_entropy_train)
        ces_valid.append(cross_entropy_valid)
        acc_train.append(frac_correct_train)
        acc_valid.append(frac_correct_valid)
        
        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
    plt.plot(xs, ces_train, 'r',label="train ce")
    plt.plot(xs, ces_valid,'g', label="validation ce")
    plt.xlabel('iteration')
    plt.ylabel('cross entropy')
    plt.title('cross entropy for lambda={}, learning_rate={}, iteration={}'.format(l, hyperparameters['learning_rate'],hyperparameters['num_iterations']))
    plt.legend(loc='upper right')
    plt.savefig('{} pen{}.png'.format(type, str(l)))
    plt.clf()
    plt.plot(xs, acc_train, 'r',label="train accuracy")
    plt.plot(xs, acc_valid,'g', label="validation accuracy")
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('accuracy for lambda={}, learning_rate={}, iteration={}'.format(l, hyperparameters['learning_rate'],hyperparameters['num_iterations']))
    plt.legend(loc='upper right')
    plt.savefig('{} accuracy {}.png'.format(type, str(l)))
    plt.clf()
    # plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    lrs = [0,0.001,0.01,0.1,1.0]
    for lr in lrs:
        run_logistic_regression(lr,'train')

