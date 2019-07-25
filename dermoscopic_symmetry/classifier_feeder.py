import joblib
import pandas as pd
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dermoscopic_symmetry.utils import package_path, save_model


def example():
    """Usage example of the main functionalities within this file. """
    dataExtractorForTraining(10, 199, 4)
    classifier, accScore = classifierTrainer(200)
    print(f'Accuracy score: {accScore}')
    save_model(classifier, "SimilarPatchClassifier")


def load_patches(patchNumber, subdir):
    """Load pairs of patches from the data/patchesDataSet from the specified subfolder.

    # Arguments :
        patchNumber: Int. The number of the patch to be loaded.
        subdir: Either 'Similar' or 'nonSimilar'

    # Outputs :
        patch_a: First patch.
        patch_b: Second patch.
    """

    filename_a = f"{package_path()}/data/patchesDataSet/{subdir}/patch{patchNumber}a.bmp"
    patch_a = imread(filename_a)
    filename_b = f"{package_path()}/data/patchesDataSet/{subdir}/patch{patchNumber}b.bmp"
    patch_b = imread(filename_b)
    return patch_a, patch_b


def list_creator(nbBins):
    """Create the necessary number of lists to compute colors feature extraction.

    # Arguments :
        nbBins: Int. The number of bins wanted for color histogram.

    # Outputs :
        lists: A list of 2*3*`nbBins` empty lists.
    """
    nbLists = 2*3*nbBins
    return [[] for _ in range(nbLists)]


def dataExtractorForTraining(patchesPerImage, nbImages, nbBins=4,
                             filename_to_save='FeaturesForTraining'):
    """Create a file "features.csv" containing glcm features and colors feature extracted from all patches of the
       "patchesDataSet" folder. This file is saved in the "patchesDataSet folder.

    # Arguments :
        patchesPerImage: The amount of patches wanted for each image.
        nbImages:        Int. The amout of images from the PH2Dataset used to extract features.
        nbBins:          Int. The number of bins wanted for color histogram. For example, 4 bins means the color
                         histogram will be divided into 4 parts.
        filename_to_save: String or None.

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

    lists = list_creator(nbBins)

    resultList = []

    for patchCount in range(0, nbImages*patchesPerImage):

        crit = patchCount // int(patchesPerImage / 2)

        if crit%2 == 0 :

            patch_a, patch_b = load_patches(patchCount, subdir='Similar')

            reda = patch_a[:, :, 0]
            greena = patch_a[:, :, 1]
            bluea = patch_a[:, :, 2]

            patch_a = rgb2gray(patch_a)
            patch_a = img_as_ubyte(patch_a)

            redb = patch_b[:, :, 0]
            greenb = patch_b[:, :, 1]
            blueb = patch_b[:, :, 2]

            patch_b = rgb2gray(patch_b)
            patch_b = img_as_ubyte(patch_b)

            # Compute glcm features extraction
            glcma = greycomatrix(patch_a, [2], [0])
            glcmb = greycomatrix(patch_b, [2], [0])

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

            patch_a, patch_b = load_patches(patchCount, subdir='nonSimilar')

            reda = patch_a[:, :, 0]
            greena = patch_a[:, :, 1]
            bluea = patch_a[:, :, 2]

            patch_a = rgb2gray(patch_a)
            patch_a = img_as_ubyte(patch_a)

            redb = patch_b[:, :, 0]
            greenb = patch_b[:, :, 1]
            blueb = patch_b[:, :, 2]

            patch_b = rgb2gray(patch_b)
            patch_b = img_as_ubyte(patch_b)

            # Compute glcm features extraction
            glcma = greycomatrix(patch_a, [2], [0])
            glcmb = greycomatrix(patch_b, [2], [0])

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

    if filename_to_save:
        df.to_csv(f"{package_path()}/data/patchesDataSet/{filename_to_save}.csv")

    return df


def classifierTrainer(maxLeafNodes, data=None, data_backup_file='patchesDataSet/Features',
                      filename_to_save_model='PatchClassifierModel'):
    """Train a random forest classifier with data from the patchesDataSet.

    # Arguments :
        maxLeafNodes: Int or None. Grow trees with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None then unlimited number of leaf nodes (scikit-learn
        RandomForestClassifier() documentation).

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """
    if data is None:
        data = pd.read_csv(f"{package_path()}/data/patchesDataSet/{data_backup_file}.csv", index_col=False)
        features = list(data)
        del features[0]
    else:
        features = list(data)

    del features[-1]     # Remove `Result` colname.

    trainX = data[features][500:]
    trainy = data.Result[500:]
    valX = data[features][:500]
    valy = data.Result[:500]

    clf = RandomForestClassifier(max_leaf_nodes=maxLeafNodes, random_state=2)
    clf.fit(trainX, trainy)

    if filename_to_save_model:
        save_model(clf, filename_to_save_model)

    preds = clf.predict(valX)

    acc = accuracy_score(valy, preds)

    return clf, acc


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()
