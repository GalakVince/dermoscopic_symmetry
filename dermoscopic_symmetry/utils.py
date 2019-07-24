import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.measure import regionprops, find_contours


def display_symmetry_axes(img, segmentation, symmetry_info):
    """Display the axis of symmetry of an image, considering shape symmetry.

    # Arguments :
        img:       The image whose shape symmetry has been evaluated with the symmetryShapeEval() function.
        segmentation:    The corresponding segmented image.
        symmetry_info: The output of the symmetryShapeEval() function used on `im`.
    """

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle("Shape Symmetry", fontsize=20)

    properties = regionprops(segmentation)
    centroid = properties[0].centroid



    if symmetry_info[0] == 0:
        pente = np.tan(symmetry_info[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segmentation)[1])
        y = pente * x + ordOrig

        penteOrtho = np.tan(symmetry_info[2][0] * np.pi / 180)
        ordOrigOrtho = centroid[0] - penteOrtho * centroid[1]
        xOrtho = np.linspace(0, np.shape(segmentation)[1])
        yOrtho = penteOrtho * x + ordOrigOrtho

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r", linewidth=2)
        axs[2].plot(xOrtho, yOrtho, "-r", linewidth=0.8)
        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Main and secondary symmetry axes")
        axs[2].axis("off")
        plt.show()

    elif symmetry_info[0] == 1:
        pente = np.tan(symmetry_info[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segmentation)[1])
        y = pente * x + ordOrig

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r")
        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Main symmetry axis")
        axs[2].axis("off")
        plt.show()

    elif symmetry_info[0] == 2:

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("No symmetry axis")
        axs[2].axis("off")
        plt.show()

    else:
        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Lesion too large: no symmetry axis")
        axs[2].axis("off")
        plt.show()


def displayTextureSymmetry(im, segIm, symmetry):
    """Display the axis of symmetry of an image, considering textures symmetry.

    # Arguments :
        im:       The image whose textures symmetry has been evaluated.
        segIm:    The corresponding segmented image.
        symmetry: The output of the `symmetryTextureEval()` function.

    # Outputs :
        Display axis. Returns 0 if no error occured.
    """

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle("Texture Symmetry", fontsize=20)

    blkSeg = np.zeros((np.shape(segIm)[0] + 2, np.shape(segIm)[1] + 2))
    blkSeg[1:np.shape(blkSeg)[0] - 1, 1:np.shape(blkSeg)[1] - 1] = segIm
    segIm = blkSeg
    contour = find_contours(segIm, 0)
    cnt = contour[0]
    minx = min(cnt[:, 1])
    maxx = max(cnt[:, 1])
    miny = min(cnt[:, 0])
    maxy = max(cnt[:, 0])
    segIm = segIm[max(0, int(miny) - 1):int(maxy) + 1, max(0, int(minx) - 1):int(maxx) + 1]
    im = im[max(0, int(miny) - 1):int(maxy), max(0, int(minx) - 1):int(maxx) + 1]

    # Compute center of mass
    segIm = img_as_ubyte(segIm / 255)
    properties = regionprops(segIm)
    centroid = properties[0].centroid

    if symmetry[0] == 0:
        pente = np.tan(symmetry[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segIm)[1])
        y = pente * x + ordOrig

        penteOrtho = np.tan(symmetry[2][0] * np.pi / 180)
        ordOrigOrtho = centroid[0] - penteOrtho * centroid[1]
        xOrtho = np.linspace(0, np.shape(segIm)[1])
        yOrtho = penteOrtho * x + ordOrigOrtho

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].plot(x, y, "-r", linewidth=2)
        axs[1].plot(xOrtho, yOrtho, "-r", linewidth=0.8)
        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("Main symmetry axis")
        axs[1].axis("off")
        plt.show()

    elif symmetry[0] == 1:
        pente = np.tan(symmetry[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segIm)[1])
        y = pente * x + ordOrig

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].plot(x, y, "-r")
        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("Main symmetry axis")
        axs[1].axis("off")
        plt.show()

    else:

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("No symmetry axis")
        axs[1].axis("off")
        plt.show()

    return 0


def load_dermoscopic(imNumber) :
    """Load a dermoscopic image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """
    filename = f"{package_path()}/data/PH2Dataset/PH2 Dataset images/{imNumber}/{imNumber}_Dermoscopic_Image/{imNumber}.bmp"
    try:
        im = imread(filename)
    except FileNotFoundError as exc:
        raise RuntimeError('Plase copy the PH2 dataset in /data/PH2Dataset/') from exc
    return im


def load_segmentation(imNumber):
    """Load a segmented image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """
    filename = f"{package_path()}/data/PH2Dataset/PH2 Dataset images/{imNumber}/{imNumber}_lesion/{imNumber}_lesion.bmp"
    try:
        im = imread(filename)
    except FileNotFoundError as exc:
        raise RuntimeError('Plase copy the PH2 dataset in /data/PH2Dataset/') from exc
    return im


def save_model(model, name):
    """Save a (Random Forest) classifier into a pickled file.

    # Arguments :
        model: Random Forest classifier.
        name: String. Filename, without extension.

    # Outputs :
        im: The loaded image.
    """
    dir = f'{package_path()}/data/models/'
    os.makedirs(dir, exist_ok=True)
    joblib.dump(model, f"{dir}/{name}.pkl")


def load_model(name):
    dir = f'{package_path()}/data/models/'
    return joblib.load(f"{dir}/{name}.pkl")


def load_PH2_asymmetry_GT():
    df = pd.read_excel(f"{package_path()}/data/symtab.xlsx")
    return df["Image Name"], df["Asymmetry"]


def package_path():
    return os.path.dirname(os.path.abspath(__file__))
