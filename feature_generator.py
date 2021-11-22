import numpy as np
from feature_extractor import extractMFCC


def sets_generator(bayes_input=30):
    """
    Function responsible for aggregation of labels (music genres) and corresponding computed features (MFCCs).
	Can be used as a part of the pipeline or a standalone script that saves the arrays into .npy files. 
	Intentionally left in this non-optimal fashion for easier tuning.

    :bayes_input: the number of songs destined to use as learning data 

    :return: numpy array of labels, numpy array of MFCCs
    """ 
	auDir = '/learning'
   
    genres = np.zeros((bayes_input*7,1))
    extracted_features = np.zeros((bayes_input*7, 156))


    for i in range (0, bayes_input):
        genres[i][0] = 1 # Classical
    for i in range(bayes_input, 2 * bayes_input):
        genres[i][0] = 2  # Pop
    for i in range (2*bayes_input, 3*bayes_input):
        genres[i][0] = 3 # Jazz
    for i in range (3*bayes_input, 4*bayes_input):
        genres[i][0] = 4 # Metal
    for i in range (4*bayes_input, 5*bayes_input):
        genres[i][0] = 5 # Hiphop
    for i in range(5 * bayes_input, 6 * bayes_input):
        genres[i][0] = 6  # Rock
    for i in range(6 * bayes_input, 7 * bayes_input):
        genres[i][0] = 7  # Country

    for i in  range(0, bayes_input):
        myfilename = '/classical/classical.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i][:] = fea
    for i in range(0, bayes_input):
        myfilename = '/dpop/pop.000{}{}.wav'.format(int(np.floor(i / 10)), (i % 10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i+bayes_input][:] = fea
    for i in  range(0, bayes_input):
        myfilename = '/jazz/jazz.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i+2*bayes_input][:] = fea
    for i in  range(0, bayes_input):
        myfilename = '/metal/metal.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i+3*bayes_input][:] = fea
    for i in  range(0, bayes_input):
        myfilename = '/rhiphop/hiphop.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i+4*bayes_input][:] = fea
    for i in  range(0, bayes_input):
        myfilename = '/xrock/rock.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i+5*bayes_input][:] = fea
    for i in range(0, bayes_input):
        myfilename = '/zcountry/country.000{}{}.wav'.format(int(np.floor(i/10)), (i%10))
        auFile = auDir + myfilename
        fea = extractMFCC(auFile)
        extracted_features[i + 6 * bayes_input][:] = fea

    np.save('features7.npy', extracted_features)
    np.save('genres7.npy', genres)

    return extracted_features, genres

sets_generator()
