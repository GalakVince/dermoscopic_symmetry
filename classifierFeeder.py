from skimage.io import *
from skimage.exposure import *
from skimage.feature import *
from skimage.color import *
from skimage import img_as_ubyte

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib
import pandas as pd


def aSimilarLoader(patchNumber):
    """Load a "a" patch from the patchesDataSet into "Similar" folder.

    # Arguments :
        patchNumber: Int. The number of the patch to be loaded.

    # Outputs :
        patch: The loaded patch.
    """

    filename = "patchesDataSet/Similar/patch" + str(patchNumber) + "a.bmp"
    patch = imread(filename)
    return patch

def bSimilarLoader(patchNumber):
    """Load a "b" patch from the patchesDataSet into "Similar" folder.

    # Arguments :
        patchNumber: Int. The number of the patch to be loaded.

    # Outputs :
        patch: The loaded patch.
    """

    filename = "patchesDataSet/Similar/patch" + str(patchNumber) + "b.bmp"
    patch = imread(filename)
    return patch

def aNonSimilarLoader(patchNumber):
    """Load a "a" patch from the patchesDataSet into "nonSimilar" folder.

    # Arguments :
        patchNumber: Int. The number of the patch to be loaded.

    # Outputs :
        patch: The loaded patch.
    """

    filename = "patchesDataSet/nonSimilar/patch" + str(patchNumber) + "a.bmp"
    patch = imread(filename)
    return patch

def bNonSimilarLoader(patchNumber):
    """Load a "b" patch from the patchesDataSet into "nonSimilar" folder.

    # Arguments :
        patchNumber: Int. The number of the patch to be loaded.

    # Outputs :
        patch: The loaded patch.
    """

    filename = "patchesDataSet/nonSimilar/patch" + str(patchNumber) + "b.bmp"
    patch = imread(filename)
    return patch

def listCreator(nbBins):
    """Create the necessary number of lists to compute colors feature extraction.

    # Arguments :
        nbBins: Int. The number of bins wanted for color histogram.

    # Outputs :
        lists: A list of 2*3*`nbBins` empty lists.
    """

    lists = []
    nbLists = 2*3*nbBins
    for k in range(nbLists):
        lists.append([])
    return lists

def dataExtractorForTraining(patchesPerImage, nbImages, nbBins):
    """Create a file "features.csv" containing glcm features and colors feature extracted from all patches of the
       "patchesDataSet" folder. This file is saved in this "patchesDtaSet folder.

    # Arguments :
        patchesPerImage: The amount of patches wanted for each image.
        nbImages:        Int. The amout of images from the PH2Dataset used to extract features.
        nbBins:          Int. The number of bins wanted for color histogram. For example, 4 bins means the color
                         histogram will be divided into 4 parts.

    # Outputs :
        Only save the "features.csv" file.
    """

    dissimilarityLista = []
    correlationLista = []
    energyLista = []
    contrastLista = []
    homogeneityLista = []

    dissimilarityListb = []
    correlationListb = []
    energyListb = []
    contrastListb = []
    homogeneityListb = []

    lists = listCreator(nbBins)

    resultList = []

    for patchCount in range(0, nbImages*patchesPerImage):

        crit = patchCount // int(patchesPerImage / 2)

        if crit%2 == 0 :

            patcha = aSimilarLoader(patchCount)

            reda = patcha[:, :, 0]
            greena = patcha[:, :, 1]
            bluea = patcha[:, :, 2]

            patcha = rgb2gray(patcha)
            patcha = img_as_ubyte(patcha)

            patchb = bSimilarLoader(patchCount)

            redb = patchb[:, :, 0]
            greenb = patchb[:, :, 1]
            blueb = patchb[:, :, 2]

            patchb = rgb2gray(patchb)
            patchb = img_as_ubyte(patchb)

            # Compute glcm features extraction
            glcma = greycomatrix(patcha, [2], [0])
            glcmb = greycomatrix(patchb, [2], [0])

            dissimilaritya = greycoprops(glcma, 'dissimilarity')[0, 0]
            correlationa = greycoprops(glcma, 'correlation')[0, 0]
            energya = greycoprops(glcma, 'energy')[0, 0]
            contrasta = greycoprops(glcma, 'contrast')[0, 0]
            homogeneitya = greycoprops(glcma, 'homogeneity')[0, 0]
            dissimilarityb = greycoprops(glcmb, 'dissimilarity')[0, 0]
            correlationb = greycoprops(glcmb, 'correlation')[0, 0]
            energyb = greycoprops(glcmb, 'energy')[0, 0]
            contrastb = greycoprops(glcmb, 'contrast')[0, 0]
            homogeneityb = greycoprops(glcmb, 'homogeneity')[0, 0]

            dissimilarityLista.append(dissimilaritya)
            correlationLista.append(correlationa)
            energyLista.append(energya)
            contrastLista.append(contrasta)
            homogeneityLista.append(homogeneitya)
            dissimilarityListb.append(dissimilarityb)
            correlationListb.append(correlationb)
            energyListb.append(energyb)
            contrastListb.append(contrastb)
            homogeneityListb.append(homogeneityb)

            # Compute colors feature extraction (color histograms)
            histo_ra = histogram(reda)
            histo_ga = histogram(greena)
            histo_ba = histogram(bluea)
            histo_rb = histogram(redb)
            histo_gb = histogram(greenb)
            histo_bb = histogram(blueb)

            numPixelsReda = [0] * 256
            cReda = 0
            for index in histo_ra[1]:
                numPixelsReda[index] = histo_ra[0][cReda]
                cReda += 1
            for k in range(0,nbBins) :
                lists[k].append(sum(numPixelsReda[k*int(256/nbBins):(k+1)*int(256/nbBins)-1]))

            numPixelsGreena = [0] * 256
            cGreena = 0
            for index in histo_ga[1]:
                numPixelsGreena[index] = histo_ga[0][cGreena]
                cGreena += 1
            for k in range(nbBins,2*nbBins) :
                lists[k].append(sum(numPixelsGreena[(k-nbBins)*int(256/nbBins):(k+1-nbBins)*int(256/nbBins)-1]))

            numPixelsBluea = [0] * 256
            cBluea = 0
            for index in histo_ba[1]:
                numPixelsBluea[index] = histo_ba[0][cBluea]
                cBluea += 1
            for k in range(2*nbBins,3*nbBins) :
                lists[k].append(sum(numPixelsBluea[(k-2*nbBins)*int(256/nbBins):(k+1-2*nbBins)*int(256/nbBins)-1]))

            numPixelsRedb = [0] * 256
            cRedb = 0
            for index in histo_rb[1]:
                numPixelsRedb[index] = histo_rb[0][cRedb]
                cRedb += 1
            for k in range(3*nbBins,4*nbBins) :
                lists[k].append(sum(numPixelsRedb[(k-3*nbBins)*int(256/nbBins):(k+1-3*nbBins)*int(256/nbBins)-1]))

            numPixelsGreenb = [0] * 256
            cGreenb = 0
            for index in histo_gb[1]:
                numPixelsGreenb[index] = histo_gb[0][cGreenb]
                cGreenb += 1
            for k in range(4*nbBins,5*nbBins) :
                lists[k].append(sum(numPixelsGreenb[(k-4*nbBins)*int(256/nbBins):(k+1-4*nbBins)*int(256/nbBins)-1]))

            numPixelsBlueb = [0] * 256
            cBlueb = 0
            for index in histo_bb[1]:
                numPixelsBlueb[index] = histo_bb[0][cBlueb]
                cBlueb += 1
            for k in range(5*nbBins,6*nbBins) :
                lists[k].append(sum(numPixelsBlueb[(k-5*nbBins)*int(256/nbBins):(k+1-5*nbBins)*int(256/nbBins)-1]))

            resultList.append(1)

        else :

            patcha = aNonSimilarLoader(patchCount)

            reda = patcha[:, :, 0]
            greena = patcha[:, :, 1]
            bluea = patcha[:, :, 2]

            patcha = rgb2gray(patcha)
            patcha = img_as_ubyte(patcha)

            patchb = bNonSimilarLoader(patchCount)

            redb = patchb[:, :, 0]
            greenb = patchb[:, :, 1]
            blueb = patchb[:, :, 2]

            patchb = rgb2gray(patchb)
            patchb = img_as_ubyte(patchb)

            # Compute glcm features extraction
            glcma = greycomatrix(patcha, [2], [0])
            glcmb = greycomatrix(patchb, [2], [0])

            dissimilaritya = greycoprops(glcma, 'dissimilarity')[0, 0]
            correlationa = greycoprops(glcma, 'correlation')[0, 0]
            energya = greycoprops(glcma, 'energy')[0, 0]
            contrasta = greycoprops(glcma, 'contrast')[0, 0]
            homogeneitya = greycoprops(glcma, 'homogeneity')[0, 0]

            dissimilarityb = greycoprops(glcmb, 'dissimilarity')[0, 0]
            correlationb = greycoprops(glcmb, 'correlation')[0, 0]
            energyb = greycoprops(glcmb, 'energy')[0, 0]
            contrastb = greycoprops(glcmb, 'contrast')[0, 0]
            homogeneityb = greycoprops(glcmb, 'homogeneity')[0, 0]

            dissimilarityLista.append(dissimilaritya)
            correlationLista.append(correlationa)
            energyLista.append(energya)
            contrastLista.append(contrasta)
            homogeneityLista.append(homogeneitya)
            dissimilarityListb.append(dissimilarityb)
            correlationListb.append(correlationb)
            energyListb.append(energyb)
            contrastListb.append(contrastb)
            homogeneityListb.append(homogeneityb)

            # Compute color feature extraction
            histo_ra = histogram(reda)
            histo_ga = histogram(greena)
            histo_ba = histogram(bluea)
            histo_rb = histogram(redb)
            histo_gb = histogram(greenb)
            histo_bb = histogram(blueb)

            numPixelsReda = [0] * 256
            cReda = 0
            for index in histo_ra[1]:
                numPixelsReda[index] = histo_ra[0][cReda]
                cReda += 1
            for k in range(0, nbBins):
                lists[k].append(sum(numPixelsReda[k * int(256 / nbBins):(k + 1) * int(256 / nbBins) - 1]))

            numPixelsGreena = [0] * 256
            cGreena = 0
            for index in histo_ga[1]:
                numPixelsGreena[index] = histo_ga[0][cGreena]
                cGreena += 1
            for k in range(nbBins, 2 * nbBins):
                lists[k].append(
                    sum(numPixelsGreena[(k - nbBins) * int(256 / nbBins):(k + 1 - nbBins) * int(256 / nbBins) - 1]))

            numPixelsBluea = [0] * 256
            cBluea = 0
            for index in histo_ba[1]:
                numPixelsBluea[index] = histo_ba[0][cBluea]
                cBluea += 1
            for k in range(2 * nbBins, 3 * nbBins):
                lists[k].append(
                    sum(numPixelsBluea[(k - 2 * nbBins) * int(256 / nbBins):(k + 1 - 2 * nbBins) * int(256 / nbBins) - 1]))

            numPixelsRedb = [0] * 256
            cRedb = 0
            for index in histo_rb[1]:
                numPixelsRedb[index] = histo_rb[0][cRedb]
                cRedb += 1
            for k in range(3 * nbBins, 4 * nbBins):
                lists[k].append(
                    sum(numPixelsRedb[(k - 3 * nbBins) * int(256 / nbBins):(k + 1 - 3 * nbBins) * int(256 / nbBins) - 1]))

            numPixelsGreenb = [0] * 256
            cGreenb = 0
            for index in histo_gb[1]:
                numPixelsGreenb[index] = histo_gb[0][cGreenb]
                cGreenb += 1
            for k in range(4 * nbBins, 5 * nbBins):
                lists[k].append(
                    sum(numPixelsGreenb[(k - 4 * nbBins) * int(256 / nbBins):(k + 1 - 4 * nbBins) * int(256 / nbBins) - 1]))

            numPixelsBlueb = [0] * 256
            cBlueb = 0
            for index in histo_bb[1]:
                numPixelsBlueb[index] = histo_bb[0][cBlueb]
                cBlueb += 1
            for k in range(5 * nbBins, 6 * nbBins):
                lists[k].append(
                    sum(numPixelsBlueb[(k - 5 * nbBins) * int(256 / nbBins):(k + 1 - 5 * nbBins) * int(256 / nbBins) - 1]))

            resultList.append(0)

    df = pd.DataFrame({"Dissimilarity a": dissimilarityLista, "Correlation a": correlationLista, "Energy a": energyLista,
                               "Contrast a": contrastLista, "Homogeneity a": homogeneityLista,
                               "Dissimilarity b": dissimilarityListb, "Correlation b": correlationListb,
                               "Energy b": energyListb, "Contrast b": contrastListb,
                               "Homogeneity b": homogeneityListb})

    for k in range(0, len(lists)):
        if k < nbBins :
            df["red " + str(k + 1) + "/" + str(nbBins) + " a"] = lists[k]
        elif k >= nbBins and k < 2*nbBins:
            df["green " + str(k + 1 - nbBins) + "/" + str(nbBins) + " a"] = lists[k]
        elif k >= 2*nbBins and k < 3*nbBins:
            df["blue " + str(k + 1 - 2*nbBins) + "/" + str(nbBins) + " a"] = lists[k]
        elif k >= 3*nbBins and k < 4*nbBins:
            df["red " + str(k + 1 - 3*nbBins) + "/" + str(nbBins) + " b"] = lists[k]
        elif k >= 4*nbBins and k < 5*nbBins:
            df["green " + str(k + 1 - 4*nbBins) + "/" + str(nbBins) + " b"] = lists[k]
        else :
            df["blue " + str(k + 1 - 5*nbBins) + "/" + str(nbBins) + " b"] = lists[k]

    df["Result"] = resultList

    df.to_csv("patchesDataSet/features.csv")

    return 0

def classifierTrainer(maxLeafNodes):
    """Train a random forest classifier with data from the patchesDataSet.

    # Arguments :
        maxLeafNodes: Int or None. Grow trees with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None then unlimited number of leaf nodes (scikit-learn
        RandomForestClassifier() documentation).

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """

    data = pd.read_csv("patchesDataSet/features.csv")

    features = list(data)
    del features[0]
    del features[-1]

    trainX = data[features][500:]
    trainy = data.Result[500:]
    valX = data[features][:500]
    valy = data.Result[:500]

    clf = RandomForestClassifier(max_leaf_nodes=maxLeafNodes, random_state=2)
    clf.fit(trainX, trainy)

    preds = clf.predict(valX)

    acc = accuracy_score(valy, preds)

    return (clf, acc)

#-----------------EXAMPLE-----------------------
# dataExtractorForTraining(10,199,4)
# classifier, accScore = classifierTrainer(200)
# print(accScore)
# joblib.dump(classifier, "similarityModel.pkl")
#-----------------------------------------------