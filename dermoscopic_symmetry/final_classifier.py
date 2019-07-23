import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def shapeBasedClassifier():
    """Train a random forest classifier with data from the "shapeScores.csv" file following the expert diagnosis about
       symmetry (PH2 Dataset).

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """

    data = pd.read_csv("../ShapeScores.csv")

    features = list(data)
    del features[0]
    del features[0]

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=3, random_state=1)
    clf.fit(trainX, trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    return (clf,acc)

def textureBasedClassifier():
    """Train a random forest classifier with data from the "textureScores.csv" file following the expert diagnosis about
           symmetry (PH2 Dataset).

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """

    data = pd.read_csv("../TextureScores.csv")

    features = list(data)
    del features[0]
    del features[0]

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=100,max_leaf_nodes=3, random_state=1)
    clf.fit(trainX, trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    return (clf,acc)

def finalClassifier():
    """Train a random forest classifier with data from the "shapeAndTextureScores.csv" (created by merging shape scores
       and texture scores) file following the expert diagnosis about symmetry (PH2 Dataset).

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """

    data = pd.read_csv("../ShapeAndTextureScores.csv")

    features = list(data)
    del features[0]
    del features[0]

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=10,max_leaf_nodes=3,random_state=2)
    clf.fit(trainX,trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    return (clf,acc)

def modelsSaver():
    """Use the trained random forests classifiers (based on shape, textures and both) and save the resulting models.

    # Outputs :
        accShape:   The accuracy of the shape-based classifier.
        accTexture: The accuracy of the textures-based classifier.
        accFinal:   The accuracy of the both shape and textures based classifier.
    """

    clfShape, accShape = shapeBasedClassifier()
    clfTexture, accTexture = textureBasedClassifier()
    clfFinal, accFinal = finalClassifier()

    joblib.dump(clfShape,"../shapeModel.pkl")
    joblib.dump(clfTexture,"../textureModel.pkl")
    joblib.dump(clfFinal,"../shapeAndTextureModel.pkl")

    return (accShape,accTexture,accFinal)

#----------------EXAMPLE----------------------
# accShape, accTexture, accFinal = modelsSaver()
# print(accShape)
# print(accTexture)
# print(accFinal)
#---------------------------------------------