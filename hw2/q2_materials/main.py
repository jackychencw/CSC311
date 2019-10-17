import run_knn as rk
import utils
import numpy as np
import matplotlib.pyplot as plt
def evaluate(y, target):
    correct = zip(*np.where(target==y))
    return len(correct) * 1.0/len(target)

def q21_script1():
    train_data, train_labels = utils.load_train()
    valid_data, valid_labels = utils.load_valid()
    k_list = [1,3,5,7,9]
    result = []
    for k in k_list:
        y = rk.run_knn(k, train_data, train_labels, valid_data)
        score = evaluate(y, valid_labels)
        result.append(score)
    result = np.array(result)
    plt.plot(k_list, result)
    plt.xlabel('K')
    plt.ylabel('Classification rate')
    plt.axis([0,10,.70,1.0])
    plt.show()

def q21_script2(chosen_k=5):
    train_data, train_labels = utils.load_train()
    test_data, test_labels = utils.load_test()
    k_list = [chosen_k-2,chosen_k,chosen_k+2]
    result = []
    for k in k_list:
        y = rk.run_knn(k, train_data, train_labels, test_data)
        score = evaluate(y, test_labels)
        result.append(score)
    result = np.array(result)
    plt.plot(k_list, result)
    plt.xlabel('K')
    plt.ylabel('Classification rate')
    plt.axis([0,10,.70,1.0])
    plt.show()

if __name__ == "__main__":
    q21_script2()
        
        
