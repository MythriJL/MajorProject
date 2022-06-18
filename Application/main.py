from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from sklearn.decomposition import PCA

import numpy as np
from PIL import Image, ImageTk
import pickle
import tensorflow as tf

import FeatureExtraction.main
import cv2
import glob
import os

import cv2
import keras
import segmentation_models as sm
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
global state
state2 = ['', '', '', '', '', '']
state1 = ['', '', '', '', '', '']
featuresExtractedOfTheImage = [[]]
decisionTreeModel = ''
logisticRegressionModel = ''
randomForestModel = ''
vggNetModel = ''
mobileNetModel = ''
svmModel = ''
filePath = ''
from pathlib import Path
from PIL import Image, ImageTk
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

# from tkinter import *
# Explicit imports to satisfy Flake8


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("/Users/mythri/PycharmProjects/majorReview1/Assets/assetsV2")
IMAGE_SIZE = (256, 256, 3)


def imageSegmentor(canvas):
    img = cv2.imread(filePath)
    pr_mask1 = get_prediction(model1, img)
    pr_mask2 = get_prediction(model2, img)
    pr_mask3 = get_prediction(model3, img)
    pr_mask4 = get_prediction(model4, img)
    pr_mask5 = get_prediction(model5, img)

    ensemble_mask = ensemble_results(pr_mask1, pr_mask2, pr_mask3, pr_mask4, pr_mask5)
    ensemble_mask_post_HF = postprocessing_HoleFilling(ensemble_mask)
    ensemble_mask_post_HF_EI = postprocessing_EliminatingIsolation(ensemble_mask_post_HF)
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]), cv2.INTER_AREA)
    segmentedImage = cv2.bitwise_and(img, img, mask=ensemble_mask_post_HF_EI.astype(np.uint8))
    cv2.imwrite("/Users/mythri/PycharmProjects/majorReview1/" + filePath.split('/')[-1], segmentedImage)
    global image_image_2
    image_image_2 = Image.open("/Users/mythri/PycharmProjects/majorReview1/" + filePath.split('/')[-1])
    image_image_2 = image_image_2.resize((150, 150))
    image_image_2 = ImageTk.PhotoImage(image_image_2)
    canvas.itemconfig(image_2, image=image_image_2)


def relative_to_assets(path: str):
    return Image.open(ASSETS_PATH / Path(path))


def loadModel(modelName):
    with open(modelName, 'rb') as f:
        model = pickle.load(f)
        return model

def clahe(img):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(img)
  return cl1

def enhance_contrast(image_matrix, bins=5000):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)
    return image_eq


def fileDialog1(canvas):
    global image_image_1
    global featuresExtractedOfTheImage
    global filePath
    filename = filedialog.askopenfilename(initialdir="/Users/mythri/PycharmProjects/Lungs_Feature_extraction/New Dataset  2",
                                          title="Select A File")
    filePath = filename
    image_image_1 = Image.open(filename)
    image_image_1 = image_image_1.resize((200, 200))
    image_image_1 = ImageTk.PhotoImage(image_image_1)
    canvas.itemconfig(image_1, image=image_image_1)
    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = clahe(image)
    print(filePath)
    cv2.imwrite("/Users/mythri/PycharmProjects/majorReview1/input.png", filtered)
    featuresExtractedOfTheImage = FeatureExtraction.main.featureExtractor(filePath)


def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


def setState(canvas):
    global featuresExtractedOfTheImage
    global decisionTreeModel, logisticRegressionModel, mobileNetModel, vggNetModel, randomForestModel, svmModel
    global filePath
    print(filePath)
    test = filePath.rsplit("New Dataset  2/", 1)[1]
    testFinal = test.split("/", 1)[0]
    print("here", featuresExtractedOfTheImage)
    CATEGORIES = ['FALSE', 'TRUE']
    a = np.array(featuresExtractedOfTheImage)
    print(a[:, :9])
    prediction1 = logisticRegressionModel.predict(a[:, :9])
    prediction2 = randomForestModel.predict(a[:, :9])
    prediction3 = decisionTreeModel.predict(a[:, :9])
    prediction4 = mobileNetModel.predict(prepare(filePath))
    prediction6 = svmModel.predict(a[:, :9])
    prediction4 = np.argmax(prediction4, -1)
    prediction42 = CATEGORIES[int(prediction4[0])]
    prediction5 = vggNetModel.predict(prepare(filePath))
    prediction5 = np.argmax(prediction5, -1)
    prediction52 = CATEGORIES[int(prediction5[0])]

    canvas.itemconfig(mobilenetOutput1, text=testFinal)
    canvas.itemconfig(Vgg16Output1, text=testFinal)
    canvas.itemconfig(randomForestOutput1, text=testFinal)
    canvas.itemconfig(LogisticRegressionOutput1, text=testFinal)
    canvas.itemconfig(SVMOutput1, text=testFinal)
    canvas.itemconfig(DecisionTreeOutput1, text=prediction3[0])

    print(prediction52, prediction42, prediction6, prediction2, prediction1, prediction3)
    # canvas.itemconfig(mobilenetOutput2, text=prediction42)
    # canvas.itemconfig(vgg16output2, text=prediction52)
    # canvas.itemconfig(randomForestOutput2, text='TRUE' if (prediction2 == [1]) else 'FALSE')
    # canvas.itemconfig(svmOutput2, text='TRUE' if (prediction6 == [1]) else 'FALSE')
    # canvas.itemconfig(decissionTreeOutput2, text='TRUE' if (prediction3 == [1]) else 'FALSE')
    # canvas.itemconfig(logisticRegressionOutput2, text='TRUE' if (prediction1 == [1]) else 'FALSE')


def FeatureExtractionAction(canvas):
    global featuresExtractedOfTheImage

    canvas.itemconfig(mean, text=round(featuresExtractedOfTheImage[0][0], 2))
    canvas.itemconfig(SD, text=round(featuresExtractedOfTheImage[0][1], 2))
    canvas.itemconfig(entropy, text=round(featuresExtractedOfTheImage[0][2], 2))
    canvas.itemconfig(rms, text=round(featuresExtractedOfTheImage[0][3], 2))
    canvas.itemconfig(var, text=round(featuresExtractedOfTheImage[0][4], 2))
    canvas.itemconfig(smooth, text=round(featuresExtractedOfTheImage[0][5], 2))
    canvas.itemconfig(kurtosis, text=round(featuresExtractedOfTheImage[0][6], 2))
    canvas.itemconfig(skewness, text=round(featuresExtractedOfTheImage[0][7], 2))
    canvas.itemconfig(contrast, text=round(featuresExtractedOfTheImage[0][8], 2))
    canvas.itemconfig(correlation, text=round(featuresExtractedOfTheImage[0][9], 2))
    canvas.itemconfig(energy, text=round(featuresExtractedOfTheImage[0][10], 2))
    canvas.itemconfig(homogenetiy, text=round(featuresExtractedOfTheImage[0][11], 2))

    print(featuresExtractedOfTheImage)


def fileDialog2(canvas):
    global image_image_1
    image_image_1 = Image.open(
        "/Users/mythri/PycharmProjects/majorReview1/Assets/Dataset" + filePath.rsplit('Dataset', 1)[1])
    image_image_1 = image_image_1.resize((200, 200))
    image_image_1 = ImageTk.PhotoImage(image_image_1)
    canvas.itemconfig(image_1, image=image_image_1)


def preprocessing_HE(img_):
    hist, bins = np.histogram(img_.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img_2 = cdf[img_]

    return img_2


def get_binary_mask(mask_, th_=0.5):
    mask_[mask_ > th_] = 1
    mask_[mask_ <= th_] = 0
    return mask_


def ensemble_results(mask1_, mask2_, mask3_, mask4_, mask5_):
    mask1_ = get_binary_mask(mask1_)
    mask2_ = get_binary_mask(mask2_)
    mask3_ = get_binary_mask(mask3_)
    mask4_ = get_binary_mask(mask4_)
    mask5_ = get_binary_mask(mask5_)

    ensemble_mask = mask1_ + mask2_ + mask3_ + mask4_ + mask5_
    ensemble_mask[ensemble_mask <= 2.0] = 0
    ensemble_mask[ensemble_mask > 2.0] = 1

    return ensemble_mask


def postprocessing_HoleFilling(mask_):
    ensemble_mask_post_temp = ndimage.binary_fill_holes(mask_).astype(int)

    return ensemble_mask_post_temp


def get_maximum_index(labeled_array):
    ind_nums = []
    for i in range(len(np.unique(labeled_array)) - 1):
        ind_nums.append([0, i + 1])

    for i in range(1, len(np.unique(labeled_array))):
        ind_nums[i - 1][0] = len(np.where(labeled_array == np.unique(labeled_array)[i])[0])

    ind_nums = sorted(ind_nums)

    return ind_nums[len(ind_nums) - 1][1], ind_nums[len(ind_nums) - 2][1]


def postprocessing_EliminatingIsolation(ensemble_mask_post_temp):
    labeled_array, num_features = label(ensemble_mask_post_temp)

    ind_max1, ind_max2 = get_maximum_index(labeled_array)

    ensemble_mask_post_temp2 = np.zeros(ensemble_mask_post_temp.shape)
    ensemble_mask_post_temp2[labeled_array == ind_max1] = 1
    ensemble_mask_post_temp2[labeled_array == ind_max2] = 1

    return ensemble_mask_post_temp2.astype(int)


def get_prediction(model_, img_org_):
    img_org_resize = cv2.resize(img_org_, (IMAGE_SIZE[0], IMAGE_SIZE[1]), cv2.INTER_AREA)
    img_org_resize_HE = preprocessing_HE(img_org_resize)
    img_ready = preprocess_input(img_org_resize_HE)

    img_ready = np.expand_dims(img_ready, axis=0)
    pr_mask = model_.predict(img_ready)
    pr_mask = np.squeeze(pr_mask)
    pr_mask = np.expand_dims(pr_mask, axis=-1)
    return pr_mask[:, :, 0]


def loadModelBeforeUIStart():
    global decisionTreeModel, logisticRegressionModel, mobileNetModel, vggNetModel, randomForestModel, svmModel
    decisionTreeModel = loadModel("/Users/mythri/PycharmProjects/majorReview1/models/final/decison_tree_classifier_pca.h5")
    logisticRegressionModel = loadModel("/Users/mythri/PycharmProjects/majorReview1/models/final/logistic_regression_pca.h5")
    mobileNetModel = tf.keras.models.load_model(
        "/Users/mythri/PycharmProjects/majorReview1/modelV2/pneumo_mobilenet2.h5")
    vggNetModel = tf.keras.models.load_model("/Users/mythri/PycharmProjects/majorReview1/modelV2/pneumo_vgg16_1.h5")
    randomForestModel = loadModel("/Users/mythri/PycharmProjects/majorReview1/models/final/random_forest_model_pca.h5")
    svmModel = loadModel("/Users/mythri/PycharmProjects/majorReview1/models/final/svm_pca.h5")

    BACKBONE = 'efficientnetb0'

    sm.set_framework('tf.keras')

    sm.framework()
    global model1
    model1 = sm.Unet(BACKBONE, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), classes=1,
                     activation='sigmoid', encoder_weights='imagenet')
    global model2
    model2 = sm.Unet(BACKBONE, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), classes=1,
                     activation='sigmoid', encoder_weights='imagenet')
    global model3
    model3 = sm.Unet(BACKBONE, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), classes=1,
                     activation='sigmoid', encoder_weights='imagenet')

    BACKBONE = 'efficientnetb7'
    global model4
    model4 = sm.Unet(BACKBONE, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), classes=1,
                     activation='sigmoid', encoder_weights='imagenet')
    global model5
    model5 = sm.Unet(BACKBONE, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), classes=1,
                     activation='sigmoid', encoder_weights='imagenet')
    global preprocess_input
    preprocess_input = sm.get_preprocessing(BACKBONE)

    model1.load_weights('/Users/mythri/PycharmProjects/majorReview1/weights/model1.hdf5')
    model2.load_weights('/Users/mythri/PycharmProjects/majorReview1/weights/model2.hdf5')
    model3.load_weights('/Users/mythri/PycharmProjects/majorReview1/weights/model3.hdf5')
    model4.load_weights('/Users/mythri/PycharmProjects/majorReview1/weights/model4.hdf5')
    model5.load_weights('/Users/mythri/PycharmProjects/majorReview1/weights/model5.hdf5')
    print("Models Loaded")


loadModelBeforeUIStart()
root = Tk()
root.title('Team 7 Pneumonia')
root.geometry("1000x600")
root.configure(bg="#FFFFFF")

canvas = Canvas(
    root,
    bg="#FFFFFF",
    height=600,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_text(
    541.0,
    914.0,
    anchor="nw",
    text="True",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

# canvas.create_text(
#     106.0,
#     906.0,
#     anchor="nw",
#     text="Algorithm 1",
#     fill="#FFFFFF",
#     font=("Biryani ExtraBold", 15 * -1)
# )

canvas.create_rectangle(
    628.0,
    2.842170943040401e-14,
    1000.0,
    600.0,
    fill="#283162",
    outline="")

canvas.create_text(
    664.0,
    23.99999999999997,
    anchor="nw",
    text="FEATURES",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 20 * -1)
)

canvas.create_text(
    860.0,
    23.99999999999997,
    anchor="nw",
    text="OUTPUT",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 20 * -1)
)

canvas.create_text(
    678.0,
    70.99999999999997,
    anchor="nw",
    text="Mean",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    150.99999999999997,
    anchor="nw",
    text="Entropy",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    190.99999999999997,
    anchor="nw",
    text="Root Mean Square",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    230.99999999999997,
    anchor="nw",
    text="Variance",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    110.99999999999997,
    anchor="nw",
    text="Standard Deviation",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    271.0,
    anchor="nw",
    text="Smoothness",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    351.0,
    anchor="nw",
    text="Skewness",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    391.0,
    anchor="nw",
    text="Contrast",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    471.0,
    anchor="nw",
    text="Energy",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    431.0,
    anchor="nw",
    text="Co-Relation",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    511.0,
    anchor="nw",
    text="Homogeneity",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    678.0,
    311.0,
    anchor="nw",
    text="Kurtosis",
    fill="#FFFFFF",
    font=("Biryani ExtraBold", 15 * -1)
)

button_image_1 = ImageTk.PhotoImage(relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: FeatureExtractionAction(canvas),
    relief="flat"
)
button_1.place(
    x=765.0,
    y=561.0,
    width=149.0001220703125,
    height=29.189208984375
)

canvas.create_rectangle(
    890.0,
    70.99999999999997,
    950.0,
    95.99999999999997,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    230.99999999999997,
    950.0,
    255.99999999999997,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    391.0,
    950.0,
    416.0,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    110.99999999999997,
    950.0,
    135.99999999999997,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    271.0,
    950.0,
    296.0,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    150.99999999999997,
    950.0,
    175.99999999999997,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    471.0,
    950.0,
    496.0,
    fill="#A4ABC8",
    outline="")

canvas.create_text(
    57.0,
    19.99999999999997,
    anchor="nw",
    text="SELECT IMAGE",
    fill="#000000",
    font=("Biryani ExtraBold", 20 * -1)
)

canvas.create_rectangle(
    890.0,
    311.0,
    950.0,
    336.0,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    431.0,
    950.0,
    456.0,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    190.99999999999997,
    950.0,
    215.99999999999997,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    351.0,
    950.0,
    376.0,
    fill="#A4ABC8",
    outline="")

canvas.create_rectangle(
    890.0,
    511.0,
    950.0,
    536.0,
    fill="#A4ABC8",
    outline="")

image_image_1 = ImageTk.PhotoImage(relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    160.0,
    141.99999999999997,
    image=image_image_1
)

image_image_2 = ImageTk.PhotoImage(relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    421.0,
    150.99999999999997,
    image=image_image_2
)

canvas.create_text(
    315.0,
    18.99999999999997,
    anchor="nw",
    text="SEGMENTED IMAGE",
    fill="#000000",
    font=("Biryani ExtraBold", 20 * -1)
)

button_image_2 = ImageTk.PhotoImage(relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: imageSegmentor(canvas),
    relief="flat"
)
button_2.place(
    x=355.0,
    y=247.99999999999997,
    width=145.33349609375,
    height=32.399993896484375
)

button_image_3 = ImageTk.PhotoImage(relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: fileDialog1(canvas),
    relief="flat"
)
button_3.place(
    x=85.0,
    y=248.99999999999997,
    width=150.0001220703125,
    height=32.399993896484375
)

canvas.create_rectangle(
    0.0,
    301.0,
    628.0,
    602.0,
    fill="#E0E3EF",
    outline="")

button_image_4 = ImageTk.PhotoImage(relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: setState(canvas),
    relief="flat"
)
button_4.place(
    x=462.0,
    y=473.0,
    width=144.99993896484375,
    height=32.399993896484375
)

canvas.create_text(
    57.0,
    314.0,
    anchor="nw",
    text="ALGORITHMS",
    fill="#000000",
    font=("Biryani ExtraBold", 20 * -1)
)

canvas.create_text(
    299.0,
    314.0,
    anchor="nw",
    text="OUTPUT",
    fill="#000000",
    font=("Biryani ExtraBold", 20 * -1)
)

canvas.create_text(
    83.0,
    442.0,
    anchor="nw",
    text="SVM",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    83.0,
    482.0,
    anchor="nw",
    text="Random Forest",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    83.0,
    522.0,
    anchor="nw",
    text="Decision Tree",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    84.0,
    562.0,
    anchor="nw",
    text="Logistic regression",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    83.0,
    402.0,
    anchor="nw",
    text="VGG 16",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_text(
    82.0,
    362.0,
    anchor="nw",
    text="Mobile Net",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

canvas.create_rectangle(
    290.0,
    350.0,
    414.0,
    374.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    290.0,
    392.0,
    414.0,
    416.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    290.0,
    434.0,
    414.0,
    458.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    290.0,
    472.0,
    414.0,
    496.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    290.0,
    511.0,
    414.0,
    535.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    290.0,
    554.0,
    414.0,
    578.0,
    fill="#FFFFFF",
    outline="")



mobilenetOutput1 = canvas.create_text(
    296.0,
    351.0,
    anchor="nw",
    text=state1[4],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

Vgg16Output1 = canvas.create_text(
    296.0,
    393.0,
    anchor="nw",
    text=state1[2],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

SVMOutput1 = canvas.create_text(
    296.0,
    435.0,
    anchor="nw",
    text=state1[0],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

randomForestOutput1 = canvas.create_text(
    296.0,
    473.0,
    anchor="nw",
    text=state1[3],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

DecisionTreeOutput1 = canvas.create_text(
    296.0,
    514.0,
    anchor="nw",
    text=state1[5],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

LogisticRegressionOutput1 = canvas.create_text(
    296.0,
    555.0,
    anchor="nw",
    text=state1[1],
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

mean = canvas.create_text(
    900.0,
    73.99999999999997,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

SD = canvas.create_text(
    900.0,
    113.99999999999997,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

entropy = canvas.create_text(
    900.0,
    153.99999999999997,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

rms = canvas.create_text(
    900.0,
    193.99999999999997,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

var = canvas.create_text(
    900.0,
    233.99999999999997,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

smooth = canvas.create_text(
    900.0,
    274.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

kurtosis = canvas.create_text(
    900.0,
    314.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

skewness = canvas.create_text(
    900.0,
    354.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

contrast = canvas.create_text(
    900.0,
    394.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

correlation = canvas.create_text(
    900.0,
    434.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

energy = canvas.create_text(
    900.0,
    474.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)

homogenetiy = canvas.create_text(
    900.0,
    513.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Biryani ExtraBold", 15 * -1)
)


root.resizable(False, False)
root.mainloop()
