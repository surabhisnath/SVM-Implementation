#Surabhi S Nath
#2016271

import matplotlib.pyplot as plt
import h5py
import numpy
import os
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib.pyplot import imread
from sklearn.metrics import accuracy_score
import glob
import math
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import itertools
from sklearn.metrics import roc_curve, auc
import pandas


def kernel1(x, y):
        return numpy.dot(x,y.T)


def kernel2(x, y, p=2):
        return (1 + numpy.dot(x,y.T))**p


def kernel3(x, y, p=3):
        return (1 + numpy.dot(x,y.T))**p

def calc_acc(cnt,y_train,predictions_train,y_test,predictions_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    true = 0
    false = 0
    #F1 score biased towards +ve
    print("Dataset: ",cnt)
    
    print("Train")
    for j in range(0,len(y_train)):
        if(y_train[j] == predictions_train[j]):
            true = true + 1
        else:
            false = false + 1
    
    acc = float(true)/(true + false)
    print(acc)
    
    
    print("Test")
    for j in range(0,len(y_test)):
        if(y_test[j] == predictions_test[j]):
            true = true + 1
        else:
            false = false + 1
    
    acc = float(true)/(true + false)
    print(acc)


def binclf_linear(c):
    clf = svm.SVC(kernel='linear', C=c, probability = True)
    clf.fit(x_train, y_train)
        
    plot(x_train,y_train,clf)
        
    w = clf.coef_[0]
    b = clf.intercept_[0]
        
    predictions_test = []
    predictions_train = []
        
    for i in range(0,len(x_test)):    
        val = numpy.dot(w,x_test[i]) + b
        if(val<0):
            predictions_test.append(0)
        else:
            predictions_test.append(1)
                
    for i in range(0,len(x_train)):
        val = numpy.dot(w,x_train[i]) + b
        if(val<0):
            predictions_train.append(0)
        else:
            predictions_train.append(1)
            
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))
    roc(clf, x_test, y_test)



def binclf_rbf(c):
    clf = svm.SVC(kernel='rbf', C=c, probability = True)
    clf.fit(x_train, y_train)

    plot(x_train,y_train,clf)

    sv = clf.support_vectors_
    alphay = clf.dual_coef_.reshape(-1,1)
    b = clf.intercept_

    predictions_test = []
    predictions_train = []

    for i in range(0,len(x_test)):
        sum = 0
        for m in range(0,len(sv)):
            sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_test[i])**2))
        val = sum + b

        if(val<0):
            predictions_test.append(0)
        else:
            predictions_test.append(1)

    for i in range(0,len(x_train)):
        sum = 0
        for m in range(0,len(sv)):
            sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_train[i])**2))
        val = sum + b

        if(val<0):
            predictions_train.append(0)
        else:
            predictions_train.append(1)
            
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))
    roc(clf, x_test, y_test)



def ovr_linear(num_classes,c,x_train, x_test, y_train, y_test):
    print("ONE VS REST")
    predictions_onevsall_train = numpy.zeros(shape=(len(x_train), num_classes))
    predictions_onevsall_test = numpy.zeros(shape=(len(x_test), num_classes))

    for i in range(0,num_classes):

        temp_y_train = numpy.copy(y_train)

        eq = numpy.where(temp_y_train==i)
        neq = numpy.where(temp_y_train!=i)
        numpy.put(temp_y_train,eq[0],[1]*len(eq[0]))
        numpy.put(temp_y_train,neq[0],[0]*len(neq[0]))    

        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(x_train, temp_y_train)

        plt.figure(cnt)
        plot(x_train, y_train, clf)

        w = clf.coef_[0]
        b = clf.intercept_[0]

        for k in range(0, len(x_train)):
            val = numpy.dot(w,x_train[k]) + b
            predictions_onevsall_train[k,i] = val

        for k in range(0, len(x_test)):
            val = numpy.dot(w,x_test[k]) + b
            predictions_onevsall_test[k,i] = val

    predictions_test = numpy.argmax(predictions_onevsall_test, axis = 1)
    predictions_train = numpy.argmax(predictions_onevsall_train, axis = 1)
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))



def ovo_linear(num_classes,c,x_train, x_test, y_train, y_test):
    print("ONE VS ONE")
    predictions_onevsone_train = numpy.zeros(shape=(len(x_train), num_classes))
    predictions_onevsone_test = numpy.zeros(shape=(len(x_test), num_classes))

    for i in range(0,num_classes):

        for j in range(i+1,num_classes):

            temp_y_train = numpy.copy(y_train)
            temp_x_train = numpy.copy(x_train)

            one = numpy.where(temp_y_train==i)
            two = numpy.where(temp_y_train==j)

            numpy.put(temp_y_train,one[0],[0]*len(one[0]))
            numpy.put(temp_y_train,two[0],[1]*len(two[0]))

            indices = one[0].tolist()
            indices.extend(two[0].tolist())
            indices.sort()

            temp_x_train = temp_x_train[indices]
            temp_y_train = temp_y_train[indices]


            clf = svm.SVC(kernel='linear', C=c)
            clf.fit(temp_x_train, temp_y_train)

            plt.figure(cnt)

            plot(x_train,y_train,clf)

            w = clf.coef_[0]
            b = clf.intercept_[0]

            for k in range(0, len(x_train)):
                val = numpy.dot(w,x_train[k]) + b
                
                if(val<0):
                    predictions_onevsone_train[k,i] = predictions_onevsone_train[k,i] + 1
                else:
                    predictions_onevsone_train[k,j] = predictions_onevsone_train[k,j] + 1

            for l in range(0, len(x_test)):
                val = numpy.dot(w,x_test[l]) + b
                #What to assign
                if(val<0):
                    predictions_onevsone_test[l,i] = predictions_onevsone_test[l,i] + 1
                else:
                    predictions_onevsone_test[l,j] = predictions_onevsone_test[l,j] + 1


    predictions_test = numpy.argmax(predictions_onevsone_test, axis = 1)
    predictions_train = numpy.argmax(predictions_onevsone_train, axis = 1)
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))  



def ovr_rbf(num_classes, c, x_train, x_test, y_train, y_test):
    print("ONE VS REST")
    predictions_onevsall_train = numpy.zeros(shape=(len(x_train), num_classes))
    predictions_onevsall_test = numpy.zeros(shape=(len(x_test), num_classes))

    for i in range(0,num_classes):

        temp_y_train = numpy.copy(y_train)

        eq = numpy.where(temp_y_train==i)
        neq = numpy.where(temp_y_train!=i)
        numpy.put(temp_y_train,eq[0],[1]*len(eq[0]))
        numpy.put(temp_y_train,neq[0],[0]*len(neq[0]))    

        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(x_train, temp_y_train)

        plot(x_train, y_train, clf)

        sv = clf.support_vectors_
        alphay = clf.dual_coef_.reshape(-1,1)
        b = clf.intercept_

        #Default gamma = 1/numfeatures = 0.5
        for k in range(0, len(x_test)):
            sum = 0
            for m in range(0,len(sv)):
                sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_test[k])**2))
            val = sum + b
            predictions_onevsall_test[k,i] = val

        for k in range(0, len(x_train)):
            sum = 0
            for m in range(0,len(sv)):
                sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_train[k])**2))
            val = sum + b
            predictions_onevsall_train[k,i] = val

    predictions_test = numpy.argmax(predictions_onevsall_test, axis = 1)
    predictions_train = numpy.argmax(predictions_onevsall_train, axis = 1)
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))



def ovo_rbf(num_classes, c, x_train, x_test, y_train, y_test):
    print("ONE VS ONE")
    predictions_onevsone_train = numpy.zeros(shape=(len(x_train), num_classes))
    predictions_onevsone_test = numpy.zeros(shape=(len(x_test), num_classes))

    for i in range(0,num_classes):

        for j in range(i+1,num_classes):

            temp_y_train = numpy.copy(y_train)
            temp_x_train = numpy.copy(x_train)

            one = numpy.where(temp_y_train==i)
            two = numpy.where(temp_y_train==j)

            numpy.put(temp_y_train,one[0],[0]*len(one[0]))
            numpy.put(temp_y_train,two[0],[1]*len(two[0]))

            indices = one[0].tolist()
            indices.extend(two[0].tolist())
            indices.sort()

            temp_x_train = temp_x_train[indices]
            temp_y_train = temp_y_train[indices]


            clf = svm.SVC(kernel='rbf', C=c)
            clf.fit(temp_x_train, temp_y_train)

            plot(x_train,y_train,clf)

            sv = clf.support_vectors_
            alphay = clf.dual_coef_.reshape(-1,1)
            b = clf.intercept_


            for k in range(0, len(x_train)):
                sum = 0
                for m in range(0,len(sv)):
                    sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_train[k])**2))
                val = sum + b

                if(val<0):
                    predictions_onevsone_train[k,i] = predictions_onevsone_train[k,i] + 1
                else:
                    predictions_onevsone_train[k,j] = predictions_onevsone_train[k,j] + 1

            for l in range(0, len(x_test)):
                sum = 0
                for m in range(0,len(sv)):
                    sum = sum + (alphay[m,0]*math.exp(-0.5*numpy.linalg.norm(sv[m] - x_test[l])**2))
                val = sum + b

                if(val<0):
                    predictions_onevsone_test[l,i] = predictions_onevsone_test[l,i] + 1
                else:
                    predictions_onevsone_test[l,j] = predictions_onevsone_test[l,j] + 1

    predictions_test = numpy.argmax(predictions_onevsone_test, axis = 1)
    predictions_train = numpy.argmax(predictions_onevsone_train, axis = 1)
    confmat = make_confmat(predictions_test, y_test, num_classes)
    #confusion_matrix(confmat, classes=[0,1],title='Confusion matrix')
    calc_acc(cnt,y_train,predictions_train,y_test,predictions_test)
    print("F1 score: ",f1_score(y_test, predictions_test, average=None))



#Code Reference: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
def roc(clf, x_test, y_test):
    arr = clf.predict_proba(x_test)
    prob = arr[:,1]
    fpr, tpr, threshold = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc)



#Code Reference: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



#Code reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def multiclass(X, Y, num_classes,s):
    plt.figure()
    y = label_binarize(Y, classes=numpy.arange(num_classes).tolist())
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = OneVsRestClassifier(svm.SVC(kernel=s, probability=True))
    clf.fit(x_train, y_train)
    prob = clf.decision_function(x_test)
    fpr = dict()
    tpr = dict()
    area_under_curve = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prob[:, i])
        area_under_curve[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prob.ravel())
    area_under_curve["micro"] = auc(fpr["micro"], tpr["micro"])
    
    for i in range(0,num_classes):
        plt.plot(fpr[i], tpr[i],lw=2, label='ROC curve (area = %0.2f)' % area_under_curve[i])
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



def plot(x_train,y_train,clf):
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = numpy.linspace(xlim[0], xlim[1], 30)
    yy = numpy.linspace(ylim[0], ylim[1], 30)
    YY, XX = numpy.meshgrid(yy, xx)
    xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')



def make_confmat(y_pred, y_act,num_classes):
    mat = numpy.zeros(shape = (num_classes,num_classes))
    for i in range(0,len(y_act)):
        mat[y_act[i],y_pred[i]]+=1
    return mat


# def confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = numpy.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()


#PART 1
os.chdir('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/data')
cnt = 0
for file in os.listdir('.'):
    cnt = cnt + 1
    with h5py.File(file) as data:
        x = data['x'][:]
        y = data['y'][:]
        
    numclasses = max(y) + 1
    arr0 = []
    arr1 = []
    arr2 = []
    for i in range(0,len(y)):
        if y[i] == 0:
            arr0.append(x[i].tolist())
        if y[i] == 1:
            arr1.append(x[i].tolist())
        if y[i] == 2:
            arr2.append(x[i].tolist())
    
    arr0 = numpy.array(arr0)
    arr1 = numpy.array(arr1)
    arr2 = numpy.array(arr2)
    
    x0 = arr0[:,0]
    y0 = arr0[:,1]
    x1 = arr1[:,0]
    y1 = arr1[:,1]
    plt.figure(cnt)
    plt.scatter(x0,y0)
    plt.figure(cnt)
    plt.scatter(x1,y1)
    if len(arr2)!=0:
        x2 = arr2[:,0]
        y2 = arr2[:,1]
        plt.figure(cnt)
        plt.scatter(x2,y2)
        x2 = x2.tolist()
        y2 = y2.tolist()
    
    plt.show()
    


#PART 2
#Code Reference: http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html
cnt = 0
for file in os.listdir('.'):
    cnt = cnt + 1
    with h5py.File(file) as data:
        x = data['x'][:]
        y = data['y'][:]
    
    if(cnt == 1 or cnt == 4):
        clf = svm.SVC(kernel = kernel2)
    elif(cnt == 2):
        clf = svm.SVC(kernel = kernel3)
    elif(cnt == 3):
        clf = svm.SVC(kernel = kernel1)
        
    clf.fit(x,y)
    
    h = 0.02
    x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    
    plt.axis('tight')
    plt.show()
    

#PART 3
#LINEAR KERNEL
#ONE VS REST, ONE VS ONE
cnt = 0
os.chdir('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/data')
for file in os.listdir('.'):
    cnt = cnt + 1
    with h5py.File(file) as data:
        x = data['x'][:]
        y = data['y'][:]
    
    num_classes = len(numpy.unique(y))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)
    
    parameters = {'C': [0.1,0.5,1,2.5,5,10,20,50], 'kernel': ['linear']}
    classifier = GridSearchCV(svm.SVC(), parameters)
    classifier.fit(x_train,y_train)
    c = classifier.best_params_.get('C')
    print("C ",c)
    
    if(num_classes>2):
        
        ovr_linear(num_classes,c,x_train, x_test, y_train, y_test)
        ovo_linear(num_classes,c,x_train, x_test, y_train, y_test)
        multiclass(x,y,num_classes,'linear')
        
    else:
        
        binclf_linear(c)
    
    plt.show()
    


#PART 4
#RBF
#ONE VS REST, ONE VS ONE

cnt = 0
os.chdir('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/data')
for file in os.listdir('.'):
    cnt = cnt + 1
    with h5py.File(file) as data:
        x = data['x'][:]
        y = data['y'][:]
    
    num_classes = len(numpy.unique(y))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)
    
    parameters = {'C': [0.5,1,2.5,5,10,20,50], 'kernel': ['rbf']}
    classifier = GridSearchCV(svm.SVC(), parameters)
    classifier.fit(x_train,y_train)
    c = classifier.best_params_.get('C')
    print("c = ", c)
    
    if(num_classes>2):
        
        ovr_rbf(num_classes,c,x_train, x_test, y_train, y_test)
        ovo_rbf(num_classes,c,x_train, x_test, y_train, y_test)
        multiclass(x,y,num_classes,'rbf')
    
    else:
        
        binclf_rbf(c)
    
    plt.show()
    


#PART 5

images = []
os.chdir('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/Train_val')
files = glob.glob('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/Train_val' + '/**/*.png', recursive=True)
for f in files:
    im = imread(f)
    im = im.reshape((im.shape[0]*im.shape[1]))
    images.append(im)
    
labels = []
sizes = [1700,1700,1700,1700,1700]
ka = [0]*sizes[0]
kha = [1]*sizes[1]
ga = [2]*sizes[2]
gha = [3]*sizes[3]
nga = [4]*sizes[4]
labels.extend(ka)
labels.extend(kha)
labels.extend(ga)
labels.extend(gha)
labels.extend(nga)

test_images = []
os.chdir('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/Test')
files = glob.glob('C:/Users/Surabhi/Desktop/IIITD/5th SEM/ML/Assignments/Assignment2/Test' + '/**/*.png', recursive=True)
for f in files:
    im = imread(f)
    im = im.reshape((im.shape[0]*im.shape[1]))
    test_images.append(im)

classes = 5
test_labels = []
sizes = [300,300,300,300,300]
ka = [0]*sizes[0]
kha = [1]*sizes[1]
ga = [2]*sizes[2]
gha = [3]*sizes[3]
nga = [4]*sizes[4]
test_labels.extend(ka)
test_labels.extend(kha)
test_labels.extend(ga)
test_labels.extend(gha)
test_labels.extend(nga)

allimages = []
alllabels = []
allimages.extend(images)
allimages.extend(test_images)
alllabels.extend(labels)
alllabels.extend(test_labels)

tsne_images = TSNE(n_components=2).fit_transform(images)
print(tsne_images)


#PART 5
clf = GridSearchCV(svm.SVC(), {'C': [0.1,0.3,1,2.5,5,10,50,100,500,1000], 'gamma': [0.001,0.01,0.1,0.5,1], 'kernel': ['rbf']}, cv=5)
clf.fit(images, labels)
predictions = clf.predict(test_images)
print(accuracy_score(test_labels, predictions))


clf2 = svm.SVC(kernel = 'rbf', C = 2.5, gamma = 0.001)
clf2.fit(tsne_images, labels)


dictionary = clf.cv_results_
trainerrors = dictionary.get('mean_train_score')
testerrors = dictionary.get('mean_test_score')
print(trainerrors)
print(testerrors)


plot(tsne_images,labels,clf2)


#Make conf mat for Hindi Dataset
print(predictions)
confmat = make_confmat(predictions, test_labels, classes)
#confusion_matrix(confmat, classes=[0,1,2,3,4],title='Confusion matrix')


#Make ROC Curve for Hindi Dataset
multiclass(allimages,alllabels,classes,'rbf')
