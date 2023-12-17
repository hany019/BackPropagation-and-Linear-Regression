import numpy as np
import sys
import math
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import timeit
import argparse


class Dataset:
    nf: int #number of features
    no: int #number of ouputs
    ns: int #number of samples
    xtable: list = [] #array of arrays of features
    ytable: list = [] # array of arrays of outputs
    xmin: list #array with minimum x for each feature
    ymin: list #array with minimum x for each output
    xmax: list #array with maximum x for each feature
    ymax: list #array with maximum x for each output

class NN:
    L: int # number of layers
    n: list # array with the number of units in each layer (including the input and output layers)
    h: list # array of arrays for the fields (h)
    xi: list # array of arrays for the activations (ξ)
    w: list # n array of matrices for the weights (w)
    theta: list # array of arrays for the thresholds (θ)
    delta: list # array of arrays for the propagation of errors (Δ)
    d_w: list # array of matrices for the changes of the weights (δw)
    d_theta: list # array of arrays for the changes of the thresholds (δθ)
    d_w_prev: list # array of matrices for the previous changes of the weights, used for the momentum term (δw (prev))
    d_theta_prev: list # array of arrays for the previous changes of the thresholds, used for the momentum term (δθ(prev))
    fact: str # the name of the activation function that it will be used. It can be one of these four: sigmoid, relu, linear, tanh.



def read_dataset(file_name):
    dataset = Dataset()
    with open(file_name) as f:
        _,dataset.nf,dataset.no = map(int,f.readline().split(' '))
        data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
        dataset.ns = len(data)
        for row in data:
            dataset.xtable.append(row[:dataset.nf])
            dataset.ytable.append(row[dataset.nf:dataset.nf+dataset.no])
    return dataset

def print_dataset():
    for m in range(0,dataset.ns):
        for n in range(0,dataset.nf):
            print("{}".format(dataset.xtable[m][n]),end='\t')
        for n in range(0,dataset.no):
            print("{}".format(dataset.ytable[m][n]),end='\t')
        print()

def scale_dataset(s_min,s_max):
    # Initialize variables 
    dataset.xmin = np.zeros(dataset.nf)
    dataset.xmax = np.zeros(dataset.nf)
    dataset.ymin = np.zeros(dataset.no)
    dataset.ymax = np.zeros(dataset.no)

    #Iterate over features 
    for n in range(0,dataset.nf):
        max = float('-inf')
        min = float('inf')
        for m in range(0,dataset.ns):
            if dataset.xtable[m][n] > max:
                max = dataset.xtable[m][n]
            if dataset.xtable[m][n] < min:
                min = dataset.xtable[m][n]
        dataset.xmin[n] = min
        dataset.xmax[n] = max

        #Scale feature values 
        for m in range(0, dataset.ns):
            dataset.xtable[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.xtable[m][n] - min)

    #Iterate over outputs
    for n in range(0,dataset.no):
        max = float('-inf')
        min = float('inf')
        for m in range(0,dataset.ns):
            if dataset.ytable[m][n] > max:
                max = dataset.ytable[m][n]
            if dataset.ytable[m][n] < min:
                min = dataset.ytable[m][n]
        dataset.ymin[n] = min
        dataset.ymax[n] = max

        #Scale output values
        for m in range(0,dataset.ns):
            dataset.ytable[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.ytable[m][n] - min)

    return dataset

def descale_dataset(s_min,s_max):
    # iterate over feature nodes
    for n in range(0,dataset.nf):
        min = dataset.xmin[n] # min value for nth feature
        max = dataset.xmax[n] # max value for nth feature
        # iterate over samples
        for m in range(0,dataset.ns):
            dataset.xtable[m][n] = min + (max - min)/(s_max - s_min)*(dataset.xtable[m][n] - s_min) # scaled feature value
    # iterate over output nodes
    for n in range(0,dataset.no):
        min = dataset.ymin[n] # min value for nth output
        max = dataset.ymax[n] # max value for nth output
        # iterate over samples
        for m in range(0,dataset.ns):
            dataset.ytable[m][n] = min + (max - min)/(s_max - s_min)*(dataset.ytable[m][n] - s_min) # scaled output value
    return dataset

def descale_y_value(value,feature,s_min,s_max):
    # calculate y min and y max values for the given feature
    y_min = dataset.ymin[feature]
    y_max = dataset.ymax[feature]
    return ( y_min + (y_max - y_min)/(s_max - s_min)*(value - s_min) )


def initialize_nn():
    # initialize neural network weights
    nn.w = []
    for l in range(nn.L):
        nn.w.append([])
        for i in range(nn.n[l]):
            nn.w[l].append([])
            for _ in range(dataset.nf if l == 0 else nn.n[l-1]):
                nn.w[l][i].append(random.randint(-50,50)/100)

    # initialize neural network weight gradients
    nn.d_w = []
    for l in range(nn.L):
        nn.d_w.append([])
        for i in range(nn.n[l]):
            nn.d_w[l].append([])
            for _ in range(dataset.nf if l == 0 else nn.n[l-1]):
                nn.d_w[l][i].append(0)

    # initialize neural network weight gradients from previous iteration
    nn.d_w_prev = []
    for l in range(nn.L):
        nn.d_w_prev.append([])
        for i in range(nn.n[l]):
            nn.d_w_prev[l].append([])
            for _ in range(dataset.nf if l == 0 else nn.n[l-1]):
                nn.d_w_prev[l][i].append(0)

    # initialize neural network bias
    nn.theta = []
    for l in range(nn.L):
        nn.theta.append([])
        for _ in range(nn.n[l]):
            nn.theta[l].append(random.random())

    # initialize neural network bias gradients
    nn.d_theta = []
    for l in range(nn.L):
        nn.d_theta.append([])
        for _ in range(nn.n[l]):
            nn.d_theta[l].append(0)

    # initialize neural network bias gradients from previous iteration
    nn.d_theta_prev = []
    for l in range(nn.L):
        nn.d_theta_prev.append([])
        for _ in range(nn.n[l]):
            nn.d_theta_prev[l].append(0)

    # initialize neural network hidden layers and activation
    nn.h = []
    nn.h.append([])
    for _ in range(dataset.nf):
        nn.h[0].append(0)
    for l in range(1,nn.L):
        nn.h.append([])
        for _ in range(nn.n[l]):
            nn.h[l].append(0)

    # initialize neural network input values for each layer
    nn.xi = []
    nn.xi.append([])
    for _ in range(dataset.nf):
        nn.xi[0].append(0)
    for l in range(1,nn.L):
        nn.xi.append([])
        for _ in range(nn.n[l]):
            nn.xi[l].append(0)

    # initialize neural network delta values
    nn.delta = []
    for l in range(nn.L):
        nn.delta.append([])
        for _ in range(nn.n[l]):
            nn.delta[l].append(0)


def feed_forward_propagation():
    #Calculate outputs of all layers
    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            aux = 0
            for j in range(0,nn.n[l-1]):
                aux += nn.w[l][i][j] * nn.xi[l-1][j]
            nn.h[l][i] = aux - nn.theta[l][i]
            nn.xi[l][i] = a(nn.h[l][i])
    return nn.xi[nn.L-1]

def a(h):
    #Sigmoid/tanh activation function
    if activation == 0:
        return 1 / (1 + math.exp(-h))
    if activation == 1:
        return math.tanh(h)

def derivate_a(h):
    # calculate gradient for sigmoid
    if activation == 0:
        g = 1 / (1 + math.exp(-h))
        return (g*(1-g))
    # calculate gradient for tanh
    if activation == 1:
        return 1 - pow(math.tanh(h),2)

def back_propagation(y):
    # calculate error at output layer
    l = nn.L-1
    for i in range(0,nn.n[l]):
        nn.delta[l][i] = derivate_a(nn.h[l][i]) * (nn.xi[l][i] - y[i])
    # calculate error at hidden layers
    for l in range(nn.L-2,0,-1):
        for i in range(0,nn.n[l-1]):
            aux = 0
            for j in range(0,nn.n[l]):
                aux += nn.delta[l][j] * nn.w[l][j][i]
                nn.delta[l-1][i] = derivate_a(nn.h[l-1][i]) * aux

def online_update_nn(n,alpha):
    for l in range(nn.L-1,0,-1): # Loop through layers in reverse
        for i in range(0,nn.n[l]): # Loop through neurons in layer
            for j in range(0,nn.n[l-1]): #Loop through inputs to neuron
                nn.d_w[l][i][j] = -n*nn.delta[l][i]*nn.xi[l-1][j] + alpha * nn.d_w_prev[l][i][j]
                nn.d_w_prev[l][i][j] = nn.d_w[l][i][j]
                nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
            nn.d_theta[l][i] = n*nn.delta[l][i] + alpha * nn.d_theta_prev[l][i]
            nn.d_theta_prev[l][i] = nn.d_theta[l][i]
            nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]
    
def main_online_BP(params):
    errors = []
    errors1 = []
    if params == "orig_0":
        outFile = open('outputs-{}/_errors_{}.csv'.format(setname,params),'w',encoding='utf-8')
    else:
        outFile = open('outputs-{}/errors_{}.csv'.format(setname,params),'w',encoding='utf-8')
    for epoch in range(0,epochs):
        error = error1 = 0
        treated = []
        # Update NN for all patients in the training set
        for pat in range(0,trainset_size-validation_size):
            r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            while r in treated:
                r = random.randint(0,sys.maxsize) % (trainset_size-validation_size)
            nn.xi[0] = dataset.xtable[r]
            treated.append(r)
            feed_forward_propagation()
            back_propagation(dataset.ytable[r])
            online_update_nn(n,alpha)
        # Calculate the average error for the training set
        for pat in range(0,trainset_size-validation_size):
            nn.xi[0] = dataset.xtable[pat]
            feed_forward_propagation()
            for i in range(0,dataset.no):
                error += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error = error/2
        errors.append(error)
        # Predict for the validation set
        Y_PredScaled = []
        for pat in range(trainset_size-validation_size,trainset_size):
            nn.xi[0] = dataset.xtable[pat]
            Y_Pred = feed_forward_propagation()
            Y_PredScaled.append(np.copy(Y_Pred))
            for i in range(0,dataset.no):
                error1 += pow(nn.xi[nn.L-1][i] - dataset.ytable[pat][i],2)
        error1 = error1/2
        errors1.append(error1)
        # Write errors to the output file
        outFile.write("{}, {}, {}\n".format(epoch, error, error1 ))
        # Print progress to the console
        if decision == "main":
            print("Finished BP {}%".format(round(100*(epoch/epochs),2)),end='\r',flush=True)
    outFile.close()
    if decision == "main":
        print("Finished BP 100%")
    return Y_PredScaled

def predict_BP(X_Test):
    Y_PredScaled = []
    for x in X_Test:
        nn.xi[0] = x
        Y_Pred = feed_forward_propagation()
        Y_PredScaled.append(np.copy(Y_Pred))
    return Y_PredScaled

def Test_Algorithm(Y_Pred, Y_Scaled,s_min,s_max, output_file):
    outFile = open('outputs-{}/{}'.format(setname,output_file), "w")
    aux1 = aux2 = 0 # Initialize variables
    output = []
    for i in range(len(Y_Test)):
        y=z=0
        for j in range(0,dataset.no):
            y += descale_y_value( Y_Pred[i][j], j, s_min, s_max ) # Calculate descaled values
            z += descale_y_value( Y_Scaled[i][j], j, s_min, s_max ) # Calculate descaled values
        output.append(y) # Store calculated value
        outFile.write("{}, {}, {}\n".format(z, y, abs(z - y))) # Write output to file
        aux1 += abs(z - y) # Calculate error sum
        aux2 += z # Calculate actual value sum
    outFile.close()
    return output,aux1/aux2*100 # Return calculated values and error percentage

def run_all_params(start=1,end=None):
    # Define global variables
    global activation, epochs, alpha, n, nn
    
    # Open file with parameters
    f = open('params.csv',encoding='utf-8')
    lines = f.readlines()
    
    # If end parameter is not given, use total number of lines
    if end is None:
        end = len(lines)
    
    # Loop through parameters lines
    for item in lines[start:end]:
        # Split line into items
        item = item.replace('\n','').split(',')
        
        # Set parameters from file
        activation = activations.index(item[1])
        epochs = int(item[2])
        n = float(item[3])
        alpha = float(item[4])
        nn.L = int(item[5])
        nn.n = layers[int(item[5])-4]
        
        # Initialize neural network
        initialize_nn()
        
        # Loop through tests for each parameter set
        for i in range(0,1):
            print('Running Params ID {} with Test ID {}'.format(item[0],i))
            main_online_BP('{}_{}'.format(item[0],i))
            
            # Reinitialize neural network
            initialize_nn()


def plot_error(id):
    #Plot training and validation error during training.
    if id == "orig_0":
        h = open('outputs-{}/_errors_{}.csv'.format(setname,id))
    else:
        h = open('outputs-{}/errors_{}.csv'.format(setname,id))
    # Create empty lists for errors and epochs
    errors = []
    errors1 = []
    epochs = []
    # Loop through the lines of the file and extract errors and epochs
    for item in h.readlines():
        item = item.replace('\n','').split(',')
        errors.append(float(item[1]))
        errors1.append(float(item[2]))
        epochs.append(int(item[0]))
    # Plot the training and validation errors over epochs
    plt.plot(epochs, errors, color="blue", linewidth=3)
    plt.plot(epochs, errors1, color="orange", linewidth=3)
    plt.show()

def plot_results(Y_res,Y_Pred,color):
    # Creating a scatter plot
    plt.scatter(Y_res, Y_Pred, color=color)
    plt.show()
    
def parse_arguments():
    #Define arguments and parse.
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument("params_file", type=str, nargs=1, help="Give the params_file")
    args = parser.parse_args()
    return args

def read_params(args):
    # Load parameters from file
    global activation, epochs, alpha, n, alpha, nn, trainset_size_perc, dataset_name, s_max, s_min, decision
    with open(args.params_file[0]) as f:
        params = f.readlines()
    params = [param.replace('\n', '') for param in params]
    dataset_name = params[0] # Set dataset name
    trainset_size_perc = int(params[1]) # Set trainset size
    activation = activations.index(params[2]) # Set activation function
    nn.L = int(params[3]) # Set number of layers
    nn.n = [int(i) for i in params[4].split(' ')] # Set layer sizes
    epochs = int(params[5]) # Set number of epochs
    n = float(params[6]) # Set learning rate
    alpha = float(params[7]) # Set regularization parameter
    s_min, s_max = [float(i) for i in params[8].split(' ')] # Set range for output values
    decision = params[9] # Set decision type

def main():
    initialize_nn() # initialize neural network

    start = timeit.default_timer() # start timer
    Y_PredScaled = main_online_BP('orig_0') # perform online BP
    stop = timeit.default_timer() # stop timer
    print(' ONLINE BP   \t Time (s): ', round(stop - start,2)) # print time
    Y_Pred,result = Test_Algorithm(Y_PredScaled, Y_ValScaled,s_min, s_max, on_val) # test BP validation
    print("Percentage of error over the BP ValidationSet: {}".format(result)) # print error percentage
    print("Mean squared error: {}".format(mean_squared_error(Y_ValScaled, Y_PredScaled))) # print mean squared error
    Y_PredScaled = predict_BP(dataset.xtable[trainset_size:]) # predict BP test set
    Y_Pred,result = Test_Algorithm(Y_PredScaled, Y_Test_scaled,s_min, s_max, on_test) # test BP test set
    print("Percentage of error over the BP TestSet: {}".format(result)) # print error percentage
    print("Mean squared error: {}".format(mean_squared_error(Y_Test_scaled, Y_PredScaled))) # print mean squared error
    plot_results(Y_Test, Y_Pred, "blue") # plot BP results

    initialize_nn() # reinitialize neural network
    start = timeit.default_timer() # start timer
    regr = linear_model.LinearRegression() # create linear regression model
    regr.fit(dataset.xtable[:trainset_size-validation_size], dataset.ytable[:trainset_size-validation_size]) # train linear regression model
    stop = timeit.default_timer() # stop timer
    print(' MLR \tTime (s): ', round(stop - start,2)) # print time
    Y_PredScaled = regr.predict(dataset.xtable[trainset_size-validation_size:trainset_size]) # predict MLR validation
    Y_Pred,result = Test_Algorithm(Y_PredScaled, Y_ValScaled,s_min, s_max, mlr_val) # test MLR validation
    print("Percentage of error over the MLR ValidationSet: {}".format(result)) # print error percentage
    print("Mean squared error: {}".format(mean_squared_error(Y_ValScaled,Y_PredScaled))) # print mean squared error
    Y_PredScaled = regr.predict(dataset.xtable[trainset_size:]) # predict MLR test set
    Y_Pred,result = Test_Algorithm(Y_PredScaled, Y_Test_scaled,s_min, s_max, mlr_test) # test MLR test set
    print("Percentage of error over the MLR TestSet: {}".format(result)) # print error percentage
    print("Mean squared error: {}".format(mean_squared_error(Y_Test_scaled,Y_PredScaled))) # print mean squared error
    plot_results(Y_Test, Y_Pred, "red") # plot MLR results

    plot_error('orig_0') # plot error graph


if __name__ == "__main__":
    # Define the list of activation functions
    activations = ['sigmoid','tanh']
    nn = NN()
    on_test = "_BP_test.csv"
    on_val = "_BP_val.csv"
    mlr_test = "_MLR_test.csv"
    mlr_val = "_MLR_val.csv"
    activation = epochs = alpha = n = alpha = trainset_size_perc = dataset_name = s_max = s_min = decision = None
    args = parse_arguments()
    #Read the dataset and scale the data
    read_params(args)
    setname = dataset_name.split('-')[1].split('.')[0]
    dataset = read_dataset(dataset_name)
    layers = [[dataset.nf,dataset.nf+2,dataset.nf+4,dataset.no],[dataset.nf,dataset.nf+2,dataset.nf+4,dataset.nf+2,dataset.no],[dataset.nf,dataset.nf+2,dataset.nf+4,dataset.nf+2,dataset.nf,dataset.no]]
    trainset_size = int(dataset.ns * trainset_size_perc / 100)
    validation_size = trainset_size // 4
    Y_Test = np.copy(dataset.ytable[trainset_size:])
    y_val = np.copy(dataset.ytable[trainset_size-validation_size:trainset_size])
    scale_dataset(s_min,s_max)
    Y_Test_scaled = np.copy(dataset.ytable[trainset_size:])
    Y_ValScaled = np.copy(dataset.ytable[trainset_size-validation_size:trainset_size])
    #Run the appropriate function based on the decision variable
    if decision == "main":
        main()
    elif decision == "run_all_params":
        run_all_params()
    elif decision == "find_best_params":
        find_best_params()
    
