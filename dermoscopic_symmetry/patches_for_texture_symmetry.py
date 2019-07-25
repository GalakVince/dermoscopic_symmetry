import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.draw import rectangle
from skimage.exposure import histogram
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import find_contours, regionprops
from skimage.segmentation import join_segmentations

from dermoscopic_symmetry.classifier_feeder import list_creator
from dermoscopic_symmetry.utils import load_segmentation, load_dermoscopic, package_path


def example():
    im = load_dermoscopic("IMD400")
    segIm = load_segmentation("IMD400")
    patchesUsed, points, reference, data = texture_symmetry_features(im, segIm, 32, 4)


def withinLesionPatchesExtractor(image, segImage, patchSize):
    """Extract patches only taken within the lesion.

    # Arguments :
        image:     The dermoscopic image where the patches are taken.
        segImage:  The corresponding segmented image.
        patchSize: Int. The size of the patches taken. For example, if `patchSize` = 32, the function takes
                   32*32 patches.

    # Outputs :
        k:       The number of patches created.
        points:  The list of points used to create the patches. Each point corresponds to the patch's upper left
                 corner
        patches: The list of patches created.
    """

    histoSeg = histogram(segImage)
    numPix = histoSeg[0][-1]

    numPatchLine = np.shape(image)[1] // patchSize
    numPatchCol = np.shape(image)[0] // patchSize

    points = []
    blk = np.zeros(np.shape(segImage))
    k = 0

    patches = []

    for countLine in range(0, numPatchLine):
        for countCol in range(0, numPatchCol):

            start = (countCol * patchSize, countLine * patchSize)
            extent = (patchSize, patchSize)
            rr, cc = rectangle(start, extent=extent)
            blk[rr, cc] = 1

            join = join_segmentations(blk, segImage)
            histoJoin = histogram(join)

            # Extract patches beginning at 0;0.
            if histoJoin[0][-1] == patchSize * patchSize and histoJoin[0][1] != numPix:
                points.append([countCol*patchSize, countLine * patchSize])
                patch = img_as_ubyte(image[countCol * patchSize: countCol * patchSize + patchSize,
                                     countLine * patchSize: countLine * patchSize + patchSize])
                patches.append(patch)
                k += 1
            blk[rr, cc] = 0


            # Extract patches beginning at 0 + patchSize/2;0 + patchSize/2.
            if countCol * patchSize + int(3*patchSize / 2) < np.shape(image)[0] and countLine * patchSize + int(
                    3*patchSize / 2) < np.shape(image)[1]:

                start = (countCol * patchSize + int(patchSize / 2), countLine * patchSize + int(patchSize / 2))
                extent = (patchSize, patchSize)
                rr, cc = rectangle(start, extent=extent)
                blk[rr, cc] = 1

                join = join_segmentations(blk, segImage)
                histoJoin = histogram(join)

                if histoJoin[0][-1] == patchSize * patchSize and histoJoin[0][1] != numPix:
                    points.append([countCol * patchSize + int(patchSize / 2), countLine * patchSize + int(patchSize / 2)])
                    patch = img_as_ubyte(image[countCol * patchSize + int(patchSize / 2): countCol * patchSize + patchSize + int(patchSize / 2),
                                             countLine * patchSize + int(patchSize / 2): countLine * patchSize + patchSize + int(patchSize / 2)])
                    patches.append(patch)
                    k += 1
            blk[rr, cc] = 0

    return (k, points, patches)

def patchesForClassifier(im, segIm, patchSize):
    """Extract patches from a dermoscopic image to prepare texture features extraction.

    # Arguments :
        im:        The dermoscopic image where the patches are taken. Note that the image will be divided into
                   upper and lower part according the horizontal axis going through the lesion's center of mass.
        segIm:     The corresponding segmented image.
        patchSize: Int. The size of the patches taken. For example, if `patchSize` = 32, the function takes
                   32*32 patches.

    # Outputs :
        n:           The number of patches created with upper part.
        pointsUsed:  The list of points used to create the patches. Each point corresponds to the patch's
                     upper left corner
        indexes:     The indexes of the points kept to extract the symmetric patches within the lower part.
        reference:   The part taken as reference ("Upper" or "Lower").
        patches:     The list of patches created.
    """

    blkSeg = np.zeros((np.shape(segIm)[0] + 2, np.shape(segIm)[1] + 2))
    blkSeg[1:np.shape(blkSeg)[0] - 1, 1:np.shape(blkSeg)[1] - 1] = segIm
    segIm = blkSeg
    contour = find_contours(segIm, 0)
    cnt = contour[0]
    minx = min(cnt[:, 1])
    maxx = max(cnt[:, 1])
    miny = min(cnt[:, 0])
    maxy = max(cnt[:, 0])
    segIm = segIm[max(0, int(miny)-1):int(maxy)+1, max(0, int(minx)-1): int(maxx)+1]
    im = im[max(0, int(miny)-1):int(maxy) + 1, max(0, int(minx)-1):int(maxx)+1]

    segIm = img_as_ubyte(segIm / 255)
    properties = regionprops(segIm)
    originalCentroid = properties[0].centroid
    im2compare = im[0:int(originalCentroid[0]),0:np.shape(im)[1]]
    segIm2compare = segIm[0:int(originalCentroid[0]),0:np.shape(segIm)[1]]

    im2flip = im[int(originalCentroid[0]):np.shape(im)[0],0:np.shape(im)[1]]
    flipIm = np.flip(im2flip, 0)
    segIm2flip = segIm[int(originalCentroid[0]):np.shape(segIm)[0],0:np.shape(segIm)[1]]
    flipSegIm = np.flip(segIm2flip, 0)

    diff = np.shape(im2compare)[0] - np.shape(flipIm)[0]
    if diff < 0:
        flipIm = flipIm[abs(diff):np.shape(flipIm)[0],0:np.shape(flipIm)[1]]
        flipSegIm = flipSegIm[abs(diff):np.shape(flipSegIm)[0], 0:np.shape(flipSegIm)[1]]
    else :
        im2compare = im2compare[abs(diff):np.shape(im2compare)[0], 0:np.shape(im2compare)[1]]
        segIm2compare = segIm2compare[abs(diff):np.shape(segIm2compare)[0], 0:np.shape(segIm2compare)[1]]

    n1, points1, patches1 = withinLesionPatchesExtractor(im2compare, segIm2compare, patchSize)
    n2, points2, patches2 = withinLesionPatchesExtractor(flipIm, flipSegIm, patchSize)
    references = ["Upper","Lower"]
    if n2<=n1 :
        n, points, patchesa = withinLesionPatchesExtractor(im2compare, segIm2compare, patchSize)
        imTreat = flipIm
        reference = references[0]
    else:
        n=n2
        points=points2
        patchesa = patches2
        imTreat = im2compare
        reference = references[1]

    indexes = [k for k in range(0,len(points))]
    originalLen = len(indexes)
    indcount = 0
    patchesb = []

    for point in points :
        if point[0] + patchSize < np.shape(imTreat)[0] and point[1] + patchSize < np.shape(imTreat)[1]:

            patch = img_as_ubyte(imTreat[point[0]: point[0] + patchSize,
                                             point[1]: point[1] + patchSize])
            patchesb.append(patch)

        else :
            sub = originalLen - len(indexes)
            del indexes[indcount - sub]
            del patchesa[indcount-sub]

        indcount += 1

    pointsUsed = []
    for index in indexes :
        pointsUsed.append(points[index])

    patches = patchesa + patchesb

    return (n,pointsUsed,indexes,reference, patches)


def texture_symmetry_features(im, segIm, patchSize, nbBins, filename_to_save='FeaturesForPreds'):
    """Extract gray level co-occurence matrix's features (dissimilarity, correlation, energy, contrast and homogeneity)
       and color feature (color histogram for each RGB's channel) from patches taken in a dermoscopic image and stored
       in the "patchesDataSet" folder and store them ("featuresForPreds.csv" in the same folder).

    # Arguments :
        im:        The dermoscopic image.
        segIm:     The corresponding segmented image.
        patchSize: Int. The size of the patches taken. For example, if `patchSize` = 32, the function takes
                   32*32 patches.
        nbBins:    Int. The number of bins wanted to divide color histogram.
        filename_to_save: String or None.

    # Outputs :
        patchesUsed: The number of patches effectively kept during the use of `patchesForClassifier()`
                     function.
        pointsUsed:  The corresponding list of points used. Each point corresponds to the patch's upper
                     left corner.
        reference:   The part of the image taken as reference ("Upper" or "Lower").
        data:        The dataframe that stores the .csv information.
    """

    num, points, indexes, reference, patches = patchesForClassifier(im, segIm, patchSize)
    patchesUsed = len(indexes)

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

    for index in range(0, int(len(patches)/2)) :

        patcha = patches[index]
        reda = patcha[:, :, 0]
        greena = patcha[:, :, 1]
        bluea = patcha[:, :, 2]
        patcha = rgb2gray(patcha)
        patcha = img_as_ubyte(patcha)

        patchb = patches[int(len(patches)/2) + index]
        redb = patchb[:, :, 0]
        greenb = patchb[:, :, 1]
        blueb = patchb[:, :, 2]
        patchb = rgb2gray(patchb)
        patchb = img_as_ubyte(patchb)

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
        # Add red feature from "a" patch in the previously created lists
        for k in range(0, nbBins):
            lists[k].append(sum(numPixelsReda[k * int(256 / nbBins):(k + 1) * int(256 / nbBins) - 1]))

        numPixelsGreena = [0] * 256
        cGreena = 0
        for index in histo_ga[1]:
            numPixelsGreena[index] = histo_ga[0][cGreena]
            cGreena += 1
        # Add green feature from "a" patch in the previously created lists
        for k in range(nbBins, 2 * nbBins):
            lists[k].append(sum(numPixelsGreena[(k - nbBins) * int(256 / nbBins):(k + 1 - nbBins) * int(256 / nbBins) - 1]))

        numPixelsBluea = [0] * 256
        cBluea = 0
        for index in histo_ba[1]:
            numPixelsBluea[index] = histo_ba[0][cBluea]
            cBluea += 1
        # Add blue feature from "a" patch in the previously created lists
        for k in range(2 * nbBins, 3 * nbBins):
            lists[k].append(
                sum(numPixelsBluea[(k - 2 * nbBins) * int(256 / nbBins):(k + 1 - 2 * nbBins) * int(256 / nbBins) - 1]))

        numPixelsRedb = [0] * 256
        cRedb = 0
        for index in histo_rb[1]:
            numPixelsRedb[index] = histo_rb[0][cRedb]
            cRedb += 1
        # Add red feature from "b" patch in the previously created lists
        for k in range(3 * nbBins, 4 * nbBins):
            lists[k].append(
                sum(numPixelsRedb[(k - 3 * nbBins) * int(256 / nbBins):(k + 1 - 3 * nbBins) * int(256 / nbBins) - 1]))

        numPixelsGreenb = [0] * 256
        cGreenb = 0
        for index in histo_gb[1]:
            numPixelsGreenb[index] = histo_gb[0][cGreenb]
            cGreenb += 1
        # Add green feature from "b" patch in the previously created lists
        for k in range(4 * nbBins, 5 * nbBins):
            lists[k].append(
                sum(numPixelsGreenb[(k - 4 * nbBins) * int(256 / nbBins):(k + 1 - 4 * nbBins) * int(256 / nbBins) - 1]))

        numPixelsBlueb = [0] * 256
        cBlueb = 0
        for index in histo_bb[1]:
            numPixelsBlueb[index] = histo_bb[0][cBlueb]
            cBlueb += 1
        # Add blue feature from "b" patch in the previously created lists
        for k in range(5 * nbBins, 6 * nbBins):
            lists[k].append(
                sum(numPixelsBlueb[(k - 5 * nbBins) * int(256 / nbBins):(k + 1 - 5 * nbBins) * int(256 / nbBins) - 1]))

    df = pd.DataFrame({"Dissimilarity a": dissimilarityLista, "Correlation a": correlationLista, "Energy a": energyLista,
                                   "Contrast a": contrastLista, "Homogeneity a": homogeneityLista,
                                   "Dissimilarity b": dissimilarityListb, "Correlation b": correlationListb,
                                   "Energy b": energyListb, "Contrast b": contrastListb,
                                   "Homogeneity b": homogeneityListb})

    # Store colors feature
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

    # Create .csv file
    if filename_to_save:
        df.to_csv(f"{package_path()}/data/patchesDataSet/{filename_to_save}.csv")

    return (patchesUsed, points, reference, df)


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()