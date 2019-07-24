import os
import shutil
from random import randint, choice

import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage.draw import rectangle
from skimage.exposure import histogram
from skimage.io import imsave
from skimage.measure import find_contours
from skimage.segmentation import join_segmentations

from dermoscopic_symmetry.utils import load_dermoscopic, load_segmentation, package_path


def randomPatchForDataset(image, segImage, patchSize, num, index):
    """Create a dataset of "a" randomly taken patches in a dermoscopic image and save them in the folder
       "patchesDataSet". They will be used to create pairs of "a" and "b" patches.
       Note that patches over borders are ignored.

            # Arguments :
                image:     The image in which the patches are randomly taken.
                segImage:  The corresponding segmented image.
                patchSize: Int. The size of the patches taken. For example, if `patchSize` = 32, the function takes
                           32*32 patches.
                num:       Int. The number of patches wanted.
                index:     Int. This parameter permits to save patches with their right name.

            # Outputs :
                points:  The list of points randomly taken to create the patches. Each point coresponds to the patch's
                         upper left corner.
                inOrOut: A list of integer coefficients corresponding to each point randomly taken.
                                    1: the corresponding patch is within the lesion
                                    0: the corresponding patch is out of the lesion

            # Note on folders organisation :
                A folder named "patchesDataSet" has to already exist.
            """

    histoSeg = histogram(segImage)
    numPix = histoSeg[0][-1]
    points = []
    inOrOut = []

    blk = np.zeros(np.shape(segImage))

    k = index
    while k != num + index:
        ligne = randint(0, np.shape(image)[0] - patchSize - 1)
        col = randint(0, np.shape(image)[1] - patchSize - 1)

        start = (ligne, col)
        extent = (patchSize, patchSize)
        rr, cc = rectangle(start, extent=extent)
        blk[rr,cc] = 1

        join = join_segmentations(blk, segImage)
        histoJoin = histogram(join)

        conditionNumPix = histoJoin[0][-1]==patchSize*patchSize

        if conditionNumPix :

            if histoJoin[0][1] == numPix:
                inOrOut.append(0)

            else :
                inOrOut.append(1)

            points.append([ligne, col])
            rdpatch = img_as_ubyte(image[ligne: ligne + patchSize, col: col + patchSize])
            imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k) + "a.bmp", rdpatch)
            k += 1

        blk[rr,cc]=0

    return (points, inOrOut)

def datasetCreator(patchesPerImage, patchSize, overlap):
    """Create a dataset of patches. This dataset is composed of similar and non similar pairs of "a" and "b" patches.

            # Arguments :
                patchesPerImage: The amount of patches wanted for each image.
                patchSize:       Int. The size of the patches taken. For example, if `patchSize` = 32, the function takes
                                 32*32 patches.
                overlap:         Int. Similar pairs of patched are created by shifting "a" patch from `overlap` pixels
                                 to have the "b" patch.

            # Outputs :
                Only save the patches into a folder named "patchesDataSet". Then order the patches created into two new
                folders (within the "patchesDataSet" folder) : "Similar" and "nonSimilar".
            """
    os.makedirs(f'{package_path()}/data/patchesDataSet/', exist_ok=True)   # Make sure dirs exist.
    df = pd.read_excel(f"{package_path()}/symtab.xlsx")
    ims = df["Image Name"]
    ims = list(ims)

    # ---------------Creation of the "a" patches-----------------
    images = ims
    index = 0
    allPoints = []
    allInorOut = []

    for img in images:
        segIm = load_segmentation(img)

        contour = find_contours(segIm, 0)
        cnt = contour[0]
        minx = min(cnt[:, 1])
        maxx = max(cnt[:, 1])
        maxy = min(cnt[:, 0])
        miny = max(cnt[:, 0])
        segIm = segIm[int(maxy):int(miny), int(minx):int(maxx)]

        im = load_dermoscopic(img)

        imCrop = im[int(maxy):int(miny), int(minx):int(maxx)]

        points, inOrOut = randomPatchForDataset(imCrop, segIm, patchSize, patchesPerImage, index)
        allPoints += [points]
        allInorOut += [inOrOut]

        index += patchesPerImage

    #---------------Creation of the "b" patches (to have pairs of patches "a" and "b")-----------------
    for countIndex in range(len(images)):

        segIm = load_segmentation(images[countIndex])

        contour = find_contours(segIm, 0)
        cnt = contour[0]
        minx = min(cnt[:, 1])
        maxx = max(cnt[:, 1])
        maxy = min(cnt[:, 0])
        miny = max(cnt[:, 0])
        segIm = segIm[int(maxy):int(miny), int(minx):int(maxx)]

        im = load_dermoscopic(images[countIndex])

        imCrop = im[int(maxy):int(miny), int(minx):int(maxx)]

        pts = allPoints[countIndex]
        ioo = allInorOut[countIndex]

        histoSeg = histogram(segIm)
        numPix = histoSeg[0][-1]

        blk00 = np.zeros(np.shape(segIm))
        blk01 = np.zeros(np.shape(segIm))
        blk10 = np.zeros(np.shape(segIm))
        blk11 = np.zeros(np.shape(segIm))


        k=0
        for c in range(int(len(pts)/2)) :

            start00 = (pts[c][0] + overlap, pts[c][1])
            start01 = (pts[c][0] - overlap, pts[c][1])
            start10 = (pts[c][0], pts[c][1] + overlap)
            start11 = (pts[c][0], pts[c][1] - overlap)
            extent = (patchSize, patchSize)

            if pts[c][0]+overlap+patchSize < np.shape(imCrop)[0]:
                rr, cc = rectangle(start00, extent=extent)
                blk00[rr, cc] = 1
            if pts[c][0] - overlap > 0:
                rr, cc = rectangle(start01, extent=extent)
                blk01[rr, cc] = 1
            if pts[c][1] + overlap + patchSize < np.shape(imCrop)[1]:
                rr, cc = rectangle(start10, extent=extent)
                blk10[rr, cc] = 1
            if pts[c][1] - overlap > 0:
                rr, cc = rectangle(start11, extent=extent)
                blk11[rr, cc] = 1

            join00 = join_segmentations(blk00, segIm)
            histoJoin00 = histogram(join00)
            join01 = join_segmentations(blk01, segIm)
            histoJoin01 = histogram(join01)
            join10 = join_segmentations(blk10, segIm)
            histoJoin10 = histogram(join10)
            join11 = join_segmentations(blk11, segIm)
            histoJoin11 = histogram(join11)

            if histoJoin00[0][-1] == patchSize * patchSize :
                patch = img_as_ubyte(imCrop[pts[c][0] + overlap: pts[c][0] + overlap + patchSize, pts[c][1]: pts[c][1] + patchSize])
                imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k + countIndex * patchesPerImage) + "b.bmp", patch)
            elif histoJoin01[0][-1] == patchSize * patchSize :
                patch = img_as_ubyte(imCrop[pts[c][0] - overlap: pts[c][0] - overlap + patchSize, pts[c][1]: pts[c][1] + patchSize])
                imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k + countIndex * patchesPerImage) + "b.bmp", patch)
            elif histoJoin10[0][-1] == patchSize * patchSize :
                patch = img_as_ubyte(imCrop[pts[c][0]: pts[c][0] + patchSize, pts[c][1] + overlap: pts[c][1] + overlap + patchSize])
                imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k + countIndex * patchesPerImage) + "b.bmp", patch)
            elif histoJoin11[0][-1] == patchSize * patchSize :
                patch = img_as_ubyte(imCrop[pts[c][0]: pts[c][0] + patchSize, pts[c][1] - overlap: pts[c][1] - overlap + patchSize])
                imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k + countIndex * patchesPerImage) + "b.bmp", patch)
            else :
                patch = img_as_ubyte(imCrop[pts[c][0]: pts[c][0] + patchSize, pts[c][1]: pts[c][1] + patchSize])
                imsave(f"{package_path()}/data/patchesDataSet/patch" + str(k + countIndex * patchesPerImage) + "b.bmp", patch)

            blk00[rr, cc] = 0
            blk01[rr, cc] = 0
            blk10[rr, cc] = 0
            blk11[rr, cc] = 0

            k += 1

        for idx in range(int(patchesPerImage/2), int(len(ioo))):

            indexesZero = []
            indexesOne = []
            coeff = ioo[idx]

            ind=0

            for digit in ioo :

                if digit==0 :
                    indexesZero.append(ind)

                else :
                    indexesOne.append(ind)
                ind += 1

            # If the "a" patch treated is within the lesion, then the corresponding "b" patch is chosen out of it. If there
            # are only in or only out patches, then take randomly a patch from another image
            #TODO : take care about the random choice in lines 236 and 249 (must have enough patches to make choice, eg : if
            # patchesPerImage=10 and have to make a choice for the 6th patch then you make a random choice in (0;-4) which
            # is impossible.
            if coeff == 1 :
                if indexesZero != []:
                    rdInd = choice(indexesZero)
                    rdPt = pts[rdInd]
                    patch = img_as_ubyte(imCrop[rdPt[0]: rdPt[0] + patchSize, rdPt[1]: rdPt[1] + patchSize])
                    imsave(f"{package_path()}/data/patchesDataSet/patch" + str(idx+countIndex*patchesPerImage) + "b.bmp", patch)
                else :
                    actual = idx + countIndex*patchesPerImage
                    rdIdx = randint(0, actual-patchesPerImage)
                    shutil.copyfile(f"{package_path()}/data/patchesDataSet/patch" + str(rdIdx) + "a.bmp", f"{package_path()}/data/patchesDataSet/patch" + str(actual) + "b.bmp" )

            else :
                if indexesOne != []:
                    rdInd = choice(indexesOne)
                    rdPt = pts[rdInd]
                    patch = img_as_ubyte(imCrop[rdPt[0]: rdPt[0] + patchSize, rdPt[1]: rdPt[1] + patchSize])
                    imsave(f"{package_path()}/data/patchesDataSet/patch" + str(idx+countIndex*patchesPerImage) + "b.bmp", patch)
                else :
                    actual = idx + countIndex * patchesPerImage
                    rdIdx = randint(0, actual - patchesPerImage)
                    shutil.copyfile(f"{package_path()}/data/patchesDataSet/patch" + str(rdIdx) + "a.bmp",
                                           f"{package_path()}/data/patchesDataSet/patch" + str(actual) + "b.bmp")

    #---------------Move created pairs of patches to the Similar or nonSimilar folder-----------------
    for compteur in range(0 , (len(ims)*patchesPerImage)):
        crit = compteur//int(patchesPerImage/2)
        if (crit%2 == 0):
            shutil.move(f"{package_path()}/data/patchesDataSet/patch" + str(compteur) + "a.bmp", f"{package_path()}/data/patchesDataSet/Similar/patch" + str(compteur) + "a.bmp")
            shutil.move(f"{package_path()}/data/patchesDataSet/patch" + str(compteur) + "b.bmp",
                        f"{package_path()}/data/patchesDataSet/Similar/patch" + str(compteur) + "b.bmp")

        else :
            shutil.move(f"{package_path()}/data/patchesDataSet/patch" + str(compteur) + "a.bmp",
                        f"{package_path()}/data/patchesDataSet/nonSimilar/patch" + str(compteur) + "a.bmp")
            shutil.move(f"{package_path()}/data/patchesDataSet/patch" + str(compteur) + "b.bmp",
                        f"{package_path()}/data/patchesDataSet/nonSimilar/patch" + str(compteur) + "b.bmp")
