import pandas as pd

from code.shapeSymmetry import symmetryRatios, seg2load, pig2load
from code.textureSymmetry import symmetryTextureEval


def shapeScoresSaver(ims, asymCoefs, stepAngle):
    """Create a "shapeScores.csv" file containing all shape symmetry scores over angles for each image of the PH2
       Dataset.

    # Arguments :
        ims: The list of considered images.
        asymCoefs: The corresponding coefficients of asymmetry in the PH2 Dataset.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                   corresponding orientation.
    # Outputs :
        Returns 0 if no error occured.
    """

    labels = []
    for coef in asymCoefs :
        if coef == 2:
            labels.append(2)
        elif coef == 1:
            labels.append(1)
        else :
            labels.append(0)

    lists = []
    for k in range(int(180/stepAngle)+1):
        lists.append([])
    c = 0
    for im in ims:
        print(c+1,"/200")
        ratios = symmetryRatios(im,stepAngle)
        for k in range(len(ratios)):
            lists[k].append(ratios[k])
        c+=1

    df = pd.DataFrame({"Labels": labels})
    for k in range(int(180/stepAngle)+1):
        df["Shape score " + str(k)] = lists[k]
    df.to_csv("../../ShapeScores.csv")

    return 0

def textureScoresSaver(ims, asymCoefs, stepAngle):
    """Create a "textureScores.csv" file containing all texture symmetry scores over angles for each image of the PH2
           Dataset.

    # Arguments :
        ims: The list of considered images.
        asymCoefs: The corresponding coefficients of asymmetry in the PH2 Dataset.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                    corresponding orientation.
    # Outputs :
        Returns 0 if no error occured.
    """

    labels = []
    for coef in asymCoefs:
        if coef == 2:
            labels.append(2)
        elif coef == 1:
            labels.append(1)
        else:
            labels.append(0)

    lists = []
    for k in range(int(180/stepAngle)+1):
        lists.append([])
    c = 0
    for im in ims:
        print(im, " : ",c + 1, "/200")
        segIm = seg2load(im)
        im = pig2load(im)
        res, ratios = symmetryTextureEval(im, segIm, stepAngle)
        for k in range(len(ratios)):
            lists[k].append(ratios[k])
        c += 1

    df = pd.DataFrame({"Labels": labels})
    for k in range(int(180/stepAngle)+1):
        df["Texture score " + str(k)] = lists[k]
    df.to_csv("../../TextureScores.csv")

    return 0

#---------------EXAMPLE------------------------------
# df = pd.read_excel("../../symtab.xlsx")
# asymCoefs = df["Asymmetry"]
# ims = df["Image number"]
# shapeScoresSaver(ims,asymCoefs,9)
# textureScoresSaver(ims,asymCoefs,9)
#----------------------------------------------------