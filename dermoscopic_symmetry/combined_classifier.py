import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dermoscopic_symmetry.shape_symmetry import shape_symmetry_ratios
from dermoscopic_symmetry.texture_symmetry import texture_symmetry
from dermoscopic_symmetry.utils import package_path, load_PH2_asymmetry_GT, load_segmentation, load_dermoscopic, \
    save_model, load_model


def example():
    """Usage example of the main functionalities within this file. """
    accShape, accTexture, accFinal = train_and_eval_all_models()
    print(f'Accuracy -- only shape: {accShape}; only texture: {accTexture}; combined: {accFinal}.')


def shape_symmetry_scores(stepAngle=9, filename_to_save='ShapeScores'):
    """Create a "shapeScores.csv" file containing all shape symmetry scores over angles for each image of the PH2
       Dataset.

    # Arguments :
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                   corresponding orientation.
        filename_to_save: String or None.

    # Outputs :
        Scores as a pandas dataframe
    """

    ims, asymCoefs = load_PH2_asymmetry_GT()

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
        ratios = shape_symmetry_ratios(im, stepAngle)
        for k in range(len(ratios)):
            lists[k].append(ratios[k])
        c+=1

    df = pd.DataFrame({"Labels": labels})
    for k in range(int(180/stepAngle)+1):
        df["Shape score " + str(k)] = lists[k]

    if filename_to_save:
        df.to_csv(f"{package_path()}/data/{filename_to_save}.csv")

    return df


def texture_symmetry_scores(stepAngle=9, filename_to_save='TextureScores'):
    """Create a "textureScores.csv" file containing all texture symmetry scores over angles for each image of the PH2
           Dataset.

    # Arguments :
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                    corresponding orientation.
        filename_to_save: String or None.

    # Outputs :
        Scores as a pandas dataframe
    """

    ims, asymCoefs = load_PH2_asymmetry_GT()

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
        segIm = load_segmentation(im)
        im = load_dermoscopic(im)
        res, ratios = texture_symmetry(im, segIm, stepAngle)
        for k in range(len(ratios)):
            lists[k].append(ratios[k])
        c += 1

    df = pd.DataFrame({"Labels": labels})
    for k in range(int(180/stepAngle)+1):
        df["Texture score " + str(k)] = lists[k]

    if filename_to_save:
        df.to_csv(f"{package_path()}/data/{filename_to_save}.csv")

    return df


def shape_symmetry_train_classifier(data=None, data_backup_filename='ShapeScores',
                                    filename_to_save_model='ShapeModel'):
    """Train a random forest classifier with data from the "shapeScores.csv" file following the expert diagnosis about
       symmetry (PH2 Dataset).

    # Arguments :
        data:   As returned by the shape_symmetry_scores function (optional).
        data_backup_filename:   Only if data is None, file to load data from.
        filename_to_save_model: String or None.

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """
    if data is None:
        data = pd.read_csv(f"{package_path()}/data/patchesDataSet/{data_backup_filename}.csv", index_col=False)
        features = list(data)
        del features[0]
    else:
        features = list(data)

    del features[0]     # Delete labels too

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=3, random_state=1)
    clf.fit(trainX, trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    if filename_to_save_model:
        save_model(clf, filename_to_save_model)

    return clf, acc


def texture_symmetry_train_classifier(data=None, data_backup_filename='TextureScores',
                                      filename_to_save_model='TextureModel'):
    """Train a random forest classifier with data from the "textureScores.csv" file following the expert diagnosis about
           symmetry (PH2 Dataset).

    # Arguments :
        data:   As returned by the texture_symmetry_scores function (optional).
        backup_filename:   Only if data is None, file to load data from.
        filename_to_save_model: String or None.

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """
    if data is None:
        data = pd.read_csv(f"{package_path()}/data/patchesDataSet/{data_backup_filename}.csv", index_col=False)
        features = list(data)
        del features[0]
    else:
        features = list(data)

    del features[0]     # Delete labels too

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=100,max_leaf_nodes=3, random_state=1)
    clf.fit(trainX, trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    if filename_to_save_model:
        save_model(clf, filename_to_save_model)

    return (clf,acc)


def combined_symmetry_train_classifier(data=None, data_backup_filename='ShapeAndTextureScores',
                                       filename_to_save_model='ShapeAndTextureModel'):
    """Train a random forest classifier with data from the "shapeAndTextureScores.csv" (created by merging shape scores
       and texture scores) file following the expert diagnosis about symmetry (PH2 Dataset).

    # Arguments :
        data:   As returned by the texture_symmetry_scores function (optional).
        backup_filename:   Only if data is None, file to load data from.
        filename_to_save_model: String or None.

    # Outputs :
        clf: The fitted classifier.
        acc: The accuracy score of the classifier
    """
    if data is None:
        data = pd.read_csv(f"{package_path()}/data/patchesDataSet/{data_backup_filename}.csv", index_col=False)
        features = list(data)
        del features[0]
    else:
        features = list(data)

    del features[0]     # Delete labels too

    trainX = data[features][50:]
    trainy = data.Labels[50:]
    valX = data[features][:50]
    valy = data.Labels[:50]

    clf = RandomForestClassifier(n_estimators=10,max_leaf_nodes=3,random_state=2)
    clf.fit(trainX,trainy)

    preds = clf.predict(valX)
    acc = accuracy_score(valy, preds)

    if filename_to_save_model:
        save_model(clf, filename_to_save_model)

    return clf, acc


def train_and_eval_all_models():
    """Train all random forests classifiers (based on shape, textures and both) and save the resulting models.

    Do not provide the features, but automatically collect the ones in files.

    # Outputs :
        accShape:   The accuracy of the shape-based classifier.
        accTexture: The accuracy of the textures-based classifier.
        accFinal:   The accuracy of the both shape and textures based classifier.
    """

    clfShape, accShape = shape_symmetry_train_classifier()
    clfTexture, accTexture = texture_symmetry_train_classifier()
    clfFinal, accFinal = combined_symmetry_train_classifier()

    return accShape, accTexture, accFinal


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()
