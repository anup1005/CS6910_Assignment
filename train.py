import argparse
import wandb

#
#numpy for all operations
import numpy as np

#Some metrics from sklearn
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

#keras for loading dataset
from keras.datasets import fashion_mnist,mnist
import warnings
#For confusion matrix
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")   


parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='myprojectname')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='myname')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=1)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=4)
parser.add_argument('-l','--loss', help = 'choices: ["squared_error", "cross-entropy"]' , type=str, default='cross-entropy')
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'sgd')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.1)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', type=str, default='random')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=1)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=4, required=False)
parser.add_argument('-a', '--activation', help='choices: ["sigmoid", "tanh", "relu"]', type=str, default='sigmoid')
# parser.add_argument('--hlayer_size', type=int, default=32)
parser.add_argument('-oa', '--output_activation', help = 'choices: ["softmax"]', type=str, default='softmax')
# parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()
if(arguments.dataset=="fashion_mnist"):    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
elif(arguments.dataset=="mnist"):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
labels = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
N , a , b = train_images.shape
#Dimesion of each Datapoint will be a*b
x = a*b
d = x
nl = len(labels)

#Flattening we are converting the 28*28 data point as 784*1 data point
flatted_train_images =  np.array([train_images[i].flatten() for i in range(len(train_images))])
#Same transition for testdata
flatted_test_images =  np.array([test_images[i].flatten() for i in range(len(test_images))])

#getting train data and validation data from training data
from sklearn.model_selection import train_test_split
train_x, valid_x, cat_train_y, cat_val_y = train_test_split(flatted_train_images, train_labels, test_size=0.1, stratify = train_labels ,random_state=42)

#One hot encoding of categorical labels 
def encode(data , nl):
  encode_data = np.array( [ [0]*nl for i in range(len(data))] )
  for i in range(len(data)):
    actual_label = data[i]
    encode_data[i][actual_label] = 1
  return encode_data

#One hot ecoding 
train_y = encode(cat_train_y , nl)
valid_y = encode(cat_val_y , nl)
test_y = encode(test_labels , nl)

#Normalizing the data
mean = np.mean(train_x , axis = 0)
train_x = (train_x - mean) /255
valid_x = (valid_x - mean)/255
test_x = (flatted_test_images - mean)/255


#Sigmoid 
def sigmoid(x):
    temp = []
    for i in range(len(x)):
      temp.append(1/(1 + np.exp(-(float(x[i])))))
    return np.array(temp)

#Tanh
def tanh(x):
    temp = []
    for i in range(len(x)):
      temp.append(np.tanh(x[i]))
    return np.array(temp)

#ReLU
def relu(x):
    temp = []
    for i in range(len(x)):
      temp.append(max(0,x[i]))
    return np.array(temp)

#Softmax
def softmax(x):
    temp = []
    sum = 0
    for i in range(len(x)):
      sum += np.exp(float(x[i]))
    for i in range(len(x)):
      temp.append( np.exp(float(x[i])) / sum)
    return np.array(temp)

#Derivative of Sigmoid
def derivative_sigmoid(x):
    return x*(1 - x)

#Derivative of Tanh
def derivative_tanh(x):
    return 1-np.square(x)

#Derivative of ReLU
def derivative_relu(x):
    temp = []
    for i in range(len(x)):
      if x[i]>0:
        temp.append(1)
      else:
        temp.append(0)
    return np.array(temp)

#Intialize w's and b's with zeroes 
def initialize_zeros(dim,hl,ol):
  #declare w and b
  w = [np.array([])]
  b = [np.array([])]
  for i in range(len(hl)):
    b.append(np.zeros(hl[i]))
    if(i == 0):
      w.append(np.zeros((hl[i],d)))
    else:
      w.append(np.zeros((hl[i],hl[i - 1])))

  for i in range(len(ol)):
    b.append(np.zeros(ol[i]))
    w.append(np.zeros((ol[i],hl[-1])))
  return w , b

  #Initialize Network 
def initialize_network(dim, hl, ol, method):
  W = [np.array([])]
  b = [np.array([])]
  
  #Setting up the random seed
  np.random.seed(2)

  #Random Intialization
  if(method=='random'):
    for i in range(len(hl)):
      b.append(np.random.randn(hl[i]))
      if(i == 0):
        W.append(np.random.randn(hl[i],dim))
      else:
        W.append(np.random.randn(hl[i],hl[i - 1]))
    for i in range(len(ol)):
      b.append(np.random.randn(ol[i]))
      W.append(np.random.randn(ol[i],hl[-1]))
      
  #Xavier Initialization
  else:
    for i in range(len(hl)):
      b.append(np.random.randn(hl[i]))
      if(i == 0):
        W.append(np.random.randn(hl[i],dim ) * np.sqrt(1/dim))
      else:
        W.append(np.random.randn(hl[i],hl[i-1]) * np.sqrt(1/hl[i-1]))
    for i in range(len(ol)):
      b.append(np.random.randn(ol[i]))
      W.append(np.random.randn(ol[i],hl[-1]) * np.sqrt(1/hl[-1]))

  return W,b


#Forward Propagation Framework
def forward_propagation(W,b,x,method):
    a = [[]]
    h = [[]]
    h[0] = x 
    num_layers = (len(W)-1)
    c = []
    d = []
    #Sigmoid as activation function in every hidden layer
    if method=='sigmoid':
        for i in range(1 , num_layers):
            c = np.dot( W[i], h[i-1] ) + b[i]
            a.append(c)
            d = sigmoid(c)
            h.append(d)
    #Tanh as activation function in every hidden layer
    elif method=='tanh':
        for i in range(1 , num_layers):
            c = np.dot( W[i], h[i-1] ) + b[i]
            a.append(c)
            d = tanh(c)
            h.append(d)
    #ReLU as activation function in every hidden layer
    elif method=='relu':
        for i in range(1 , num_layers):
            c = np.dot( W[i], h[i-1] ) + b[i]
            a.append(c)
            d = relu(c)
            h.append(d)
    #Softmax at output Layer
    c = np.dot( W[num_layers], h[num_layers-1] ) + b[num_layers]
    a.append(c)
    d = softmax(c)
    h.append(d)

    return a,h

#Backpropagation Framework
def back_prop(W,h,x,y,y_pred,act_fun,loss_fun):
  del_W,del_b=[[]],[[]]

  #Computing output grad wrt Cross Entropy Loss function
  if loss_fun == "cross_entropy" :
    del_a = (y_pred - y)
    
  #Computing output grad wrt Squared Error Loss function
  else:
    y_label = y_pred[np.argmax(y)]
    del_a = 2 * (y_label - 1) * y_label * ( y - y_pred )

  for i in range(len(W)-1, 0, -1):

    #computing gradients wrt parameters W,b
    dW = np.array(np.dot(np.matrix(del_a).T , np.matrix(h[i-1])))
    db = np.array( del_a )

    #computing gradients wrt below layer activation function
    dh = np.dot( np.transpose(W[i]), del_a )

    #computing gradients wrt below layer pre-activation function
    if act_fun == "sigmoid":
      del_a = dh * derivative_sigmoid( h[i - 1] )
    
    elif act_fun == "tanh":
      del_a = dh * derivative_tanh( h[i - 1] )
    
    elif act_fun == "relu":
      del_a = dh * derivative_relu( h[i - 1] )

    del_W.insert(1, dW)
    del_b.insert(1, db)

  return del_W, del_b


def stochastic_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  #Initializing the weights and biases based on given strategy
  W,b = initialize_network(d, hl, ol, strat)
  dw , db = initialize_zeros(784,hl,ol)

  seen = 0
  for e in range(epochs):
    
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen += 1
      a,h = forward_propagation(W, b, x, act_fun)
      y_pred=h[len(h)-1]
      tdw,tdb=back_prop(W, h, x, y, y_pred, act_fun, loss_fun)

      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]

      if(seen==batch or it==len(train_x)-1):
        seen = 0
        #Update weights and biases
        for i, (weight, deriv) in enumerate(zip(W, dw)):
          W[i] = weight - eta * np.array(deriv)

        for i, (bias, deriv) in enumerate(zip(b, db)):
          b[i] = bias - eta * np.array(deriv)
        
        dw , db = initialize_zeros(784,hl,ol)

    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b

def momentum_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  W,b=initialize_network(d, hl, ol, strat)
  dw , db = initialize_zeros(784,hl,ol)
  prev_w , prev_b = initialize_zeros(784,hl,ol)
  gamma=0.9
  seen=0

  for e in range(epochs):
    #dw , db = initialize_zeros(784,hl,ol)
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen+=1
      a,h = forward_propagation(W, b, x, act_fun)
      y_pred=h[len(h)-1]
      
      tdw,tdb=back_prop(W, h, x, y, y_pred, act_fun, loss_fun)
  
      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]
    
      if(seen==batch or it==len(train_x)-1):
        seen=0
        #Update weights and biases
        for i, (weight, deriv) in enumerate(zip(W, dw)):
          W[i] = weight - eta * np.array(deriv) - gamma*prev_w[i]
          prev_w[i] = eta * np.array(deriv) + gamma*prev_w[i]

        for i, (bias, deriv) in enumerate(zip(b, db)):
          b[i] = bias - eta * np.array(deriv) - gamma*prev_b[i]
          prev_b[i] = eta * np.array(deriv) + gamma*prev_b[i]

        dw , db = initialize_zeros(784,hl,ol)

    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b

def nesterov_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  W,b=initialize_network(d, hl, ol, strat)

  gamma=0.9
  seen=0

  
  prev_w , prev_b = initialize_zeros(784,hl,ol)
  v_w , v_b =initialize_zeros(784,hl,ol)

  dw , db = initialize_zeros(784,hl,ol)
  tw , tb = initialize_zeros(784,hl,ol)

  for e in range(epochs):

    for i in range(len(W)):
      v_w[i] = gamma*prev_w[i]
      v_b[i] = gamma*prev_b[i]
    

    for i in range(len(W)):
      tw[i] = W[i] - v_w[i]
      tb[i] = b[i] - v_b[i]
  
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen+=1

      a,h = forward_propagation(tw, tb, x, act_fun)
      y_pred=h[len(h)-1]
      
      tdw,tdb=back_prop(tw, h, x, y, y_pred, act_fun, loss_fun)
      
      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]

      if seen==batch or it == len(train_x)-1:
        seen=0
        #Update weights and biases
        for i, (weight, deriv) in enumerate(zip(W, dw)):
          v_w[i] = gamma*prev_w[i] + eta*np.array(deriv)
          W[i] = weight - v_w[i]
          tw[i] = W[i]
          prev_w = v_w

        for i, (bias, deriv) in enumerate(zip(b, db)):
          v_b[i] = gamma*prev_b[i] + eta*np.array(deriv)
          b[i] = bias - v_b[i]
          tb[i] = b[i]
          prev_b = v_b
        
        dw , db = initialize_zeros(784,hl,ol)

    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b

def rmsprop(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  W,b=initialize_network(d, hl, ol, strat)
  seen=0
  
  prev_w , prev_b = initialize_zeros(784,hl,ol)
  dw , db = initialize_zeros(784,hl,ol)
  eps , beta = 1e-8 , 0.9

  for e in range(epochs):
  
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen+=1
      a,h = forward_propagation(W, b, x, act_fun)
      y_pred=h[len(h)-1]
      
      tdw,tdb=back_prop(W, h, x, y, y_pred, act_fun, loss_fun)
      
      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]

      if seen==batch or it == len(train_x)-1:

        seen=0
        #Update weights and biases
        for i in range(len(W)):
          prev_w[i] = beta*prev_w[i] + (1-beta)*(dw[i]**2)
          prev_b[i] = beta*prev_b[i] + (1-beta)*(db[i]**2)

        for i, (weight, deriv) in enumerate(zip(W, dw)):
          W[i] = weight - (eta / np.sqrt(prev_w[i] + eps)) * np.array(deriv)

        for i, (bias, deriv) in enumerate(zip(b, db)):
          b[i] = bias - (eta / np.sqrt(prev_b[i] + eps)) * np.array(deriv)
        
        dw , db = initialize_zeros(784,hl,ol)
        
    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b



def adaptive_moments(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  W,b=initialize_network(d, hl, ol, strat)
  seen=0
  c=0
  v_w , v_b = initialize_zeros(784,hl,ol)
  dw , db = initialize_zeros(784,hl,ol)
  m_w , m_b = initialize_zeros(784,hl,ol)
  eps , beta1 , beta2 = 1e-8 , 0.9 , 0.99

  m_w_hat , m_b_hat = initialize_zeros(784,hl,ol)
  v_w_hat , v_b_hat = initialize_zeros(784,hl,ol)

  for e in range(epochs):
  
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen+=1
      a,h = forward_propagation(W, b, x, act_fun)
      y_pred=h[len(h)-1]
      
      tdw,tdb=back_prop(W, h, x, y, y_pred, act_fun, loss_fun)
      
      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]

      if seen==batch or it == len(train_x)-1:

        seen=0
        c+=1
        #Update weights and biases
        for i in range(len(W)):
          m_w[i] = beta1 * m_w[i] + (1-beta1) * dw[i]
          m_b[i] = beta1 * m_b[i] + (1-beta1) * db[i]

          v_w[i] = beta2 * v_w[i] + (1-beta2) * (dw[i]**2)
          v_b[i] = beta2 * v_b[i] + (1-beta2) * (db[i]**2)

        for i in range(len(W)):
          m_w_hat[i] = m_w[i] / (1 - np.power(beta1,c))
          m_b_hat[i] = m_b[i] / (1 - np.power(beta1,c))

          v_w_hat[i] = v_w[i] / (1 - np.power(beta2,c))
          v_b_hat[i] = v_b[i] / (1 - np.power(beta2,c))

        for i, (weight, deriv) in enumerate(zip(W, dw)):
          W[i] = weight - (eta / np.sqrt(v_w_hat[i] + eps)) * m_w_hat[i]

        for i, (bias, deriv) in enumerate(zip(b, db)):
          b[i] = bias - (eta / np.sqrt(v_b_hat[i] + eps)) * m_b_hat[i]
        
        dw , db = initialize_zeros(784,hl,ol)

    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b


def nadam(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch):
  W,b=initialize_network(d, hl, ol, strat)
  seen=0
  c=0
  v_w , v_b = initialize_zeros(784,hl,ol)
  dw , db = initialize_zeros(784,hl,ol)
  m_w , m_b = initialize_zeros(784,hl,ol)
  eps , beta1 , beta2 = 1e-8 , 0.9 , 0.99

  m_w_hat , m_b_hat = initialize_zeros(784,hl,ol)
  v_w_hat , v_b_hat = initialize_zeros(784,hl,ol)

  for e in range(epochs):
  
    for it, (x, y) in enumerate(zip(train_x, train_y)):
      seen+=1
      a,h = forward_propagation(W, b, x, act_fun)
      y_pred=h[len(h)-1]
      
      tdw,tdb=back_prop(W, h, x, y, y_pred, act_fun, loss_fun)
      
      for i in range(len(tdw)):
        dw[i] += tdw[i]
        db[i] += tdb[i]

      if seen==batch or it == len(train_x)-1:

        seen=0
        c+=1
        #Update weights and biases
        for i in range(len(W)):
          m_w[i] = beta1 * m_w[i] + (1-beta1) * dw[i]
          m_b[i] = beta1 * m_b[i] + (1-beta1) * db[i]

          v_w[i] = beta2 * v_w[i] + (1-beta2) * (dw[i]**2)
          v_b[i] = beta2 * v_b[i] + (1-beta2) * (db[i]**2)

        for i in range(len(W)):
          m_w_hat[i] = ( (beta1 * m_w[i]) + ( (1 - beta1) * dw[i] ) )/ (1 - np.power(beta1,c))
          m_b_hat[i] = ( (beta1 * m_b[i]) + ( (1 - beta1) * db[i] ) )/ (1 - np.power(beta1,c))

          v_w_hat[i] = v_w[i] / (1 - np.power(beta2,c))
          v_b_hat[i] = v_b[i] / (1 - np.power(beta2,c))

        for i, (weight, deriv) in enumerate(zip(W, dw)):
          W[i] = weight - (eta / np.sqrt(v_w_hat[i] + eps)) * m_w_hat[i]

        for i, (bias, deriv) in enumerate(zip(b, db)):
          b[i] = bias - (eta / np.sqrt(v_b_hat[i] + eps)) * m_b_hat[i]
        
        dw , db = initialize_zeros(784,hl,ol)

    #Getting train,val,test accuracies and losses and predictions with true labels
    val_acc, val_loss,yt_val,ypred_train = get_predictions_accuracy(W, b, valid_x, valid_y, act_fun, loss_fun)
    train_acc, train_loss,yt_train,ypred_train = get_predictions_accuracy(W, b, train_x, train_y, act_fun, loss_fun)
    test_acc, test_loss,yt_test,ypred_test = get_predictions_accuracy(W, b, test_x, test_y, act_fun, loss_fun)

    print("epoch:" , e+1 , " "+loss_fun+" ", "train_acc :" , train_acc , "valid_acc :" , val_acc , "test_acc :" , test_acc)
    
    if loss_fun=="cross_entropy":
      wandb.log({
          "Epoch": e+1,
          "Train Loss": train_loss,
          "Train Acc": train_acc,
          "Valid Loss": val_loss,
          "Valid Acc": val_acc})
    else:
      wandb.log({
          "Train Loss (squared_error)": train_loss,
          "Train Acc (squared_error)": train_acc,
          "Valid Loss (squared_error)": val_loss,
          "Valid Acc (squared_error)": val_acc})
  return W,b


def get_predictions_accuracy(W, b, X, y, method, loss_fun):
  sum,loss=0,0
  yhat = []
  yt = []
  for dp in range(len(X)):
    a = []
    h = []
    h = X[dp] 
    num_layers = (len(W)-1)
    if method=='sigmoid':
      for i in range(1 , num_layers):
        a = np.dot( W[i], h ) + b[i]
        h = sigmoid(a)

    elif method=='tanh':
      for i in range(1 , num_layers):
        a = np.dot( W[i], h ) + b[i]
        h = tanh(a)

    elif method=='relu':
      for i in range(1 , num_layers):
        a = np.dot( W[i], h ) + b[i]
        h = relu(a)

    a = np.dot( W[num_layers], h ) + b[num_layers]
    y_pred = softmax(a)

    ytrue = y[dp]
    if(ytrue[np.argmax(y_pred)]==1):
      sum=sum+1
    
    if loss_fun == "cross_entropy":
      loss += -np.sum(ytrue*np.log(y_pred))
    else:
      loss += np.sum((ytrue-y_pred)**2)

    yhat.append(labels[np.argmax(y_pred)])
    yt.append(labels[np.argmax(ytrue)])

  acc=sum/len(X)
  loss=loss/len(X)

  return acc,loss,yt,yhat

wandb.login(key="1ee7845713d1303ac1abff70cf959518e1ae311c")
wandb.init(project=arguments.wandb_project,entity="cs22m017")
  
hidden_layer_size = arguments.hidden_size
hidden_layers = arguments.num_layers
hl = [hidden_layer_size]*hidden_layers
ol = [len(train_y[0])]
n_hl = len(hl)
act_fun =arguments.activation
loss_fun=arguments.loss
eta=arguments.learning_rate
alpha=arguments.weight_decay
strat=arguments.weight_init
epochs=arguments.epochs
batch = arguments.batch_size
if(arguments.optimizer=="nadam"):
    nadam(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
if(arguments.optimizer=="adam"):
    adaptive_moments(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
if(arguments.optimizer=="nag"):
    nesterov_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
if(arguments.optimizer=="mgd"):
    momentum_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
if(arguments.optimizer=="rmsprop"):
    rmsprop(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
if(arguments.optimizer=="sgd"):
    stochastic_gradient_descent(train_x,train_y,valid_x,valid_y,d,hl,ol,act_fun,loss_fun,epochs,eta,strat,alpha,batch)
