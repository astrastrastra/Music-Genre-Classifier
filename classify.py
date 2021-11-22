import numpy as np
import os
import matplotlib.pyplot as plt
from feature_generator import sets_generator
from feature_extractor import extractMFCC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def main():
    genres_num = 7
	
	X, y = sets_generator()
    #X = np.load('modele/cechy' + str(genres_num) + '.npy')
    #y = np.load('modele/gatunki' + str(genres_num) + '.npy')
    
    model = GaussianNB()
    model.fit(X, np.ravel(y))

    directory='/validation'
    gat_true = []
    gat_pred = []
    i=0
    j=0

    for filename in os.listdir(directory):
        i+=1
        for name in os.listdir(directory+'/'+filename):
            j+=1
            try:
                fea = extractMFCC(directory + '/' + filename + '/' + name)
                Xnew = np.array(fea).reshape(1, -1)
                ynew = model.predict(Xnew)
                gat_true.append(i)
                gat_pred.append(int(ynew[0]))
            except:
                continue
        if i==genres_num: break;

    correct = sum(x == y for x, y in zip(gat_true, gat_pred))

    sklearn = confusion_matrix(gat_true, gat_pred)

    display_labels = ["Classical", "Pop", "Jazz", "Metal", "Hip hop", "Rock", "Country"]

    disp = ConfusionMatrixDisplay(confusion_matrix=sklearn, display_labels=display_labels[0:(genres_num - 1)])
    disp.plot(include_values=True, cmap='PuBu')
    plt.title("Wspolczynnik zgodnosci: %.2f" % (correct/j))

    plt.show()


if __name__ == "__main__":
    main()