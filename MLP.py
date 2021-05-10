import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from algebra import  normalize
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def sqr(x):
    return (x*x)

#load dataset
def load_dataset(traindatapath,testdatapath):
    train = pd.read_csv(traindatapath, sep='\t')  # Retrive train data
    Train_x = train.iloc[:, [0,1]]
    Train_y = train.iloc[:, 2]
    test = pd.read_csv(testdatapath, sep='\t')  # Retrive test data
    Test_x = test.iloc[:, [0, 1]]
    Test_y = test.iloc[:, 2]
    return Train_x,Train_y,Test_x,Test_y

#convert labels to binary digits
def convertlabels(data):
    y = np.zeros ((len (data), 7))
    for i, label in enumerate (data):
        if label == 'HasProperty':
            y[i] = [1, 0, 0, 0, 0, 0,0]
        elif label == 'Antonym':
            y[i] = [0, 1, 0, 0, 0, 0,0]
        elif label == 'IsA':
            y[i] = [0, 0, 1, 0, 0, 0,0]
        elif label == 'MadeOf':
            y[i] = [0, 0, 0, 1, 0, 0,0]
        elif label == 'PartOf':
            y[i] = [0, 0, 0,0, 1, 0, 0]
        elif label == 'Synonym':
            y[i] = [ 0, 0,0, 0, 0, 1,0]
        elif label == 'HasA':
            y[i] = [0, 0,0, 0, 0, 0,1]
    return y

def EVALution_dataset_statistics(Train_x,Train_y,Test_x,Test_y):
	#number and type of classes
	print("number of class ",len(Train_y.value_counts()))
	listofclass = Train_y.values
	list = []
	for i in listofclass:
		if i not in list:
			list.append(i)
	print("type of class", (list))
	#number of training instances per class
	print("total number of training instances  ",len(Train_x))
	print("number of training instances per class \n",Train_y.value_counts())
	#number of testing instances per class
	print("total number of testing instances ", len(Test_x))
	print("number of testing instances per class \n",Test_y.value_counts())


def Load_Word_Embeddings(file):
    print("Retrieving words embeddings...")
    F = open (file)  # Open the file
    dim = 50  # Dimensionality (i.e., features) of word embeddings. Each word is represented by 50 dimensions
    # read the vectors.
    vects = {}  # dictionary of word embedding in the form vects={'word1':[..,..,..],'word2':[..,..,..]}
    vocab = []  # list of the words in the vocabulary vocab=['word1','word2','word3',...]
    line = F.readline ( )  # Read the file line by line using readline() method
    while len (line) != 0:
        p = line.split (" " )
        word = p[0]
        v = np.zeros (dim, float)
        for i in range (0, dim):
            v[i] = float(p[i + 1])
        # If you want to normalize the vectors, then call the normalize function.
        vects[word] = normalize (v)
        vocab.append (word)
        line = F.readline ( )  # Read the next line
    # print("Number of words in the vocabulary is: ", len(vocab))
    F.close ( )  # close the file

    return vects, vocab

def Encoder(data,vects, vocab):
		vectdata = np.zeros((len(data),100))
		count = 0
		for i in data.itertuples():
			a= i[1]
			b= i[2]
			if (a in vocab) and (b in vocab):
				x= vects[a]
				y= vects[b]
				vectdata[count] = np.concatenate((x, y), axis=0)
			count= count+1
		return vectdata

def sigmoid_fun(x):
	return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
	return x*(1-x)

def comput_acc_loss(correctprd,X,Y,acc_list,loss_list,sumofloss):
    acc = comput_acc (correctprd, Y)
    acc_list.append (acc)
    loss = (sumofloss) / float (X.shape[0])
    loss_list.append (loss)
    return acc, loss, acc_list, loss_list

def plotting(a,b,epochs,str):
    # Plot train/test accuracy/loss among epochs
    x = [i for i in range (epochs)]
    plt.plot (x, a, label="train " + str)
    plt.plot (x, b, label="test " + str)
    plt.xticks (x)
    plt.legend ( )
    plt.show ( )

def performance(predict, Test_y):
    y_pred = predict
    y_test = Test_y
    f1_score = metrics.f1_score (y_test, y_pred, labels=np.unique (y_pred), pos_label=1, average='macro')
    # precision and recall
    recall = metrics.recall_score (y_test, y_pred, labels=np.unique (y_pred), average='macro')
    precision = metrics.precision_score (y_test, y_pred, labels=np.unique (y_pred), average='macro')
    accu = metrics.accuracy_score (y_test, y_pred)
    print("accu", accu)
    print ("f1 score: ", f1_score*100)
    print ("precision: ", precision*100)
    print ("recall: ", recall*100)

def count_correct_pred(target,actual):
    correct = 0
    for n in range (7):
        if actual[n] >= .5:
            actual[n] = 1.0
        else:
            actual[n] = 0.0
    if np.array_equal (target, actual) == True:
        correct += 1

    return correct

def comput_acc(correct,y):
    acc = (correct / float (len (y))) * 100
    return acc

def Forwardpass(X,Y,H_dim,O_dim,w_ItoH,w_HtO):
    sumofloss = 0
    correctprd = 0
    net_H = np.zeros (H_dim)
    out_H = np.zeros (H_dim)
    net_O = np.zeros (O_dim)
    out_O = np.zeros (O_dim)
    O_matrixout = np.zeros ((X.shape[0], O_dim))
    H_matrixout = np.zeros ((X.shape[0], H_dim))
    E_matrix = np.zeros((X.shape[0], O_dim))

    for i in range (X.shape[0]):
        net_H = np.dot (X[i, :], w_ItoH)
        out_H = sigmoid_fun (net_H)
        net_O = np.dot (out_H, w_HtO)
        out_O = sigmoid_fun (net_O)
        E_matrix[i,:] = out_O - Y[i, :]
        Error = 0.5 * (sqr(E_matrix[i,:]))
        E_total = Error[0] + Error[1] + Error[2] +Error[3] + Error[4] + Error[5]+ Error[6]
        sumofloss = sumofloss + E_total
        # check that if the predicted y is equal to the actual y , if so then increase the count by 1
        correctprd = correctprd + count_correct_pred (Y[i], out_O)
        O_matrixout[i, :] = out_O
        H_matrixout[i, :] = out_H

    return H_matrixout, O_matrixout, E_matrix, E_total, sumofloss, correctprd

def backwardpass(X,out_H,out_O,E_matrix,I_dim,H_dim,O_dim,w_ItoH,w_HtO):
    learning_rate = 0.1
    #delta_out = np.zeros (O_dim)
    delta_hid = np.zeros (H_dim)
    O2H_gradients = np.zeros ((H_dim, O_dim))
    H2I_gradients = np.zeros ((I_dim, H_dim))
    delta_out = sigmoid_deriv (out_O) * E_matrix
    for i in range (X.shape[0]):
        for node in range (O_dim):
            O2H_gradients[:, node] = delta_out[i,node] * out_H[i ,node]
        for node in range (H_dim):
            delta = sigmoid_deriv (out_H[i ,node]) * np.dot (delta_out[i,:], w_HtO[node, :])
            delta_hid[node] = delta
            H2I_gradients[:, node] = delta * X[i, :]

        w_HtI = w_ItoH - (H2I_gradients * learning_rate)
        w_HtO = w_HtO - (O2H_gradients * learning_rate)
    return w_HtI, w_HtO

def Training(train_X, train_Y,test_X, test_Y):
    # Define the trainable parameters
    epochs = 20
    I_dim = 100
    H_dim = 50
    O_dim = 7
    w_ItoH = np.random.normal (0, 1, (I_dim, H_dim))
    # b_ItoH = np.random.rand(1)
    w_HtO = np.random.normal (0, 1, (H_dim, O_dim))
    # b_HtoO = np.random.rand(1)
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for k in range (epochs):
        # Random shuffuling train_X and train_Y together
        randomize = np.arange (train_X.shape[0])
        random.shuffle (randomize)
        train_X = train_X[randomize]
        train_Y = train_Y[randomize]

        # feedforward pass for train data
        out_H, tr_out_O, E_matrix, E_total, tr_sumofloss, tr_correctprd = Forwardpass (train_X, train_Y, H_dim, O_dim,
                                                                                    w_ItoH, w_HtO)
        # backpropagation
        w_HtI, w_HtO = backwardpass (train_X, out_H, tr_out_O, E_matrix, I_dim, H_dim, O_dim, w_ItoH, w_HtO)

        # feedforward pass for test data using the final weightes
        out_H, out_O, E_matrix, E_total, te_sumofloss, te_correctprd = Forwardpass (test_X, test_Y, H_dim, O_dim,
                                                                                    w_ItoH, w_HtO)
        # compute the accuracy & loss for train data
        tr_acc, tr_loss, tr_acc_list, tr_loss_list = comput_acc_loss (tr_correctprd, train_X, train_Y, train_acc,
                                                                      train_loss, tr_sumofloss)
        # compute the accuracy & loss for test data
        te_acc, te_loss, te_acc_list, te_loss_list = comput_acc_loss (te_correctprd, test_X, test_Y, test_acc,
                                                                      test_loss, te_sumofloss)
        print ("epoch", k, "Tr_acc", tr_acc, "Tr_loss", tr_loss,"--", "Test_acc", te_acc, "Test_loss", te_loss)
    print("Train performance:")
    performance (tr_out_O, Train_y)
    print("Test performance:")
    performance (out_O, Test_y)

    # plot acc & loss
    plotting (tr_acc_list, te_acc_list, epochs, 'acc')
    plotting (tr_loss_list, te_loss_list, epochs, 'loss')
    # print confusion matrix
    print("confusion matrix: \n",confusion_matrix(Test_y.argmax(axis=1), out_O.argmax(axis=1) ))
    cm = confusion_matrix(Test_y.argmax(axis=1), out_O.argmax(axis=1) )
    plt.clf ( )
    plt.imshow (cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['HasProperty', 'Antonym', 'IsA', 'PartOf', 'MadeOf', 'Synonym', 'HasA']
    plt.title ('Confusion Matrix ')
    plt.ylabel ('True label')
    plt.xlabel ('Predicted label')
    tick_marks = np.arange (len (classNames))
    plt.xticks (tick_marks, classNames, rotation=45)
    plt.yticks (tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range (7):
        for j in range (7):
            plt.text (j, i, str (cm[i][j]))
    plt.show ( )



#________________


if __name__ == "__main__":
    Train_x, Train_y, Test_x, Test_y = load_dataset ("train.tsv", "test.tsv")
    EVALution_dataset_statistics (Train_x, Train_y, Test_x, Test_y)
    Train_y = convertlabels (Train_y)
    Test_y = convertlabels (Test_y)
    vects, vocab = Load_Word_Embeddings ('../ANN2ndProject/glove.6B.50d.txt')
    Tr_x = Encoder (Train_x, vects, vocab)
    Te_x = Encoder (Test_x, vects, vocab)
    Training (Tr_x, Train_y, Te_x, Test_y)



