from random import shuffle
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def normalize(x): #scaling vector based on length of a vector
    norm_x = np.linalg.norm(x)
    return x if norm_x == 0 else (x / norm_x) #to avoid division to zero

#count number of instances
def countinstances(path0,path1,path2,path3):
    list = [path0,path1,path2,path3]
    total_instances_No = 0
    for f in list:
        file = open(f, "r")
        Counter = 0
        # Reading from file
        Content = file.read()
        CoList = Content.split("\n")
        for i in CoList:
            if i:
                Counter += 1
        total_instances_No += Counter
    return total_instances_No

#load dataset
def load_dataset(path0,path1,path2,path3):
    data = []
    label = []
    list = [path0,path1,path2,path3]
    for f in list:
        with open(f, "r") as a_file:
            for line in a_file:
                data.append(line.strip())
                label.append(f)
    return data,label

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
        vects[word] = normalize(v)
        vocab.append (word)
        line = F.readline ( )  # Read the next line
    F.close ( )  # close the file
    return vects, vocab

# convert label into numerical value
def label_encoder(label):
    y = []
    for i in range(len(label)):
        if label[i] == 'animals.txt':
            y.append(0)
        elif label[i] == 'countries.txt':
            y.append(1)
        elif label[i] == 'fruits.txt':
            y.append(2)
        elif label[i] == 'veggies.txt':
            y.append(3)
    return y

# convert the dataset into vectors
def Encoder(total_instances_No,data,vects, vocab):
        vectdata = np.zeros((total_instances_No,50))
        count = 0
        for i in data:
            if (i in vocab):
                x = vects[i]
                vectdata[count,:]= x
            count +=1
        return vectdata

# calculate euclidean distance
def distnce(x,y):
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x - y)

# Training SOM algorithm
def som(O_dim,data,count):
    #initialize
    regulating_constant = 10
    initial_neighborhood_width = 2.5
    learning_rate = 0.01
    epochs = 20
    output_dim = O_dim
    weights =  np.random.rand(50, output_dim)
    distance_vect = np.zeros((output_dim))

    for k in range(epochs):
        print("epoch :_______________" , k)
        # Random shuffuling data
        randomize = np.arange (data.shape[0])
        shuffle (randomize)
        data = data[randomize]

        for i in range(count):
            # Competitive step
            for node in range(output_dim):
                dis = distnce(data[i],weights[:,node])
                distance_vect[node] = dis
            #Determine the winner node
            BMU = distance_vect.tolist().index(min(distance_vect))

            # Cooperation step
            for node in range(output_dim):
                dis_BMU_neighbor = distnce(weights[:,node],weights[:,BMU])
                neighborhood_size = initial_neighborhood_width*math.exp(-(epochs/regulating_constant))
                T = math.exp(-(dis_BMU_neighbor**2)/(2*(neighborhood_size**2)))

                #Synaptic adaptation (update the weights)
                weights[:, node]=weights[:, node]+learning_rate*T*(data[i]-weights[:,node])
                weights = np.round(weights,3)
            learning_rate = 0.5 * learning_rate

    return weights

# Testing SOM algorithm
def test_som(weights,data,count,output_dim,labels):
    distance_vect = np.zeros((output_dim))
    labels_pred= []
    for i in range(count):
        for node in range(output_dim):
            dis = distnce(data[i], weights[:, node])
            distance_vect[node] = dis
        BMU = distance_vect.tolist().index(min(distance_vect))
        labels_pred.append(BMU)
    recall, precision, f1_score = performance(labels_pred, labels)
    return  recall,precision,f1_score

# Calculate performance of model
def performance(predict, Test_y):
    y_pred = predict
    y_test = Test_y
    print(y_pred)
    print(y_test)
    f1_score = metrics.f1_score (y_test, y_pred, labels=np.unique (y_pred), pos_label=1, average='macro')
    recall = metrics.recall_score (y_test, y_pred, labels=np.unique (y_pred), average='macro')
    precision = metrics.precision_score (y_test, y_pred, labels=np.unique (y_pred), average='macro')
    accu = metrics.accuracy_score (y_test, y_pred)
    print("accuracy: ", accu*100)
    print ("f1 score: ", f1_score*100)
    print ("precision: ", precision*100)
    print ("recall: ", recall*100)
    return recall*100,precision*100,f1_score*100


def plotting(list_a,list_1b,list_2b,list_3b):
    plt.plot(list_a, list_1b,'o' , label='f1_score')
    plt.plot(list_a, list_2b, 'o',label='recall')
    plt.plot(list_a, list_3b, 'o',label='precision')
    plt.xlabel('k values')
    plt.xticks(list_a)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    total_instances_No = countinstances('animals.txt','countries.txt', 'fruits.txt','veggies.txt')
    data,label = load_dataset('animals.txt','countries.txt', 'fruits.txt','veggies.txt')
    vects, vocab = Load_Word_Embeddings('glove.6B.50d.txt')
    vectdata = Encoder(total_instances_No, data, vects, vocab)
    encodedlabel = label_encoder(label)

    list_k = [2,3,4] # number of output neurons
    list_f1_score = []
    list_recall = []
    list_precision= []
    for k in list_k:
        weights = som(k, vectdata,total_instances_No)
        recall,precision,f1_score= test_som(weights, vectdata, total_instances_No, k, encodedlabel)
        list_f1_score.append(f1_score)
        list_precision.append(precision)
        list_recall.append(recall)
    plotting(list_k,list_f1_score, list_recall, list_precision)



