import sys
sys.path.append('../')
from EmoPy.src.fermodel import FERModel

emotions = [['surprise', 0], ['happiness', 0], ['anger', 0], ['fear', 0], ['disgust', 0], ['calm', 0]]

target_emotions = ['surprise', 'anger', 'fear', 'calm']
target_emotions2 = ['surprise', 'disgust', 'happiness']
target_emotions3 = ['surprise', 'anger', 'fear']
target_emotions4 = ['anger', 'fear', 'calm']
target_emotions5 = ['anger', 'calm', 'happiness']
target_emotions6 = ['anger', 'fear', 'disgust']
target_emotions7 = ['disgust', 'surprise', 'calm']
target_emotions8 = ['surprise', 'disgust', 'sadness']
target_emotions9 = ['anger', 'happiness']

models = [
    FERModel(target_emotions, verbose=True),
    FERModel(target_emotions2, verbose=True),
    FERModel(target_emotions3, verbose=True),
    FERModel(target_emotions4, verbose=True),
    FERModel(target_emotions5, verbose=True),
    FERModel(target_emotions6, verbose=True),
    FERModel(target_emotions7, verbose=True),
    FERModel(target_emotions8, verbose=True),
    FERModel(target_emotions9, verbose=True),
]

print('Predicting on happy image...')
models[0].predict('image_data/sample_happy_image.png')

print('Predicting on disgust image...')
models[0].predict('image_data/sample_disgust_image.png')

print('Predicting on anger image...')
models[0].predict('image_data/sample_anger_image2.png')


def get_score(filepath):
    print('Predicting on ' + filepath)

#   format data
    results = []
    for i in range (len(models)):
        datastrlist = models[i].predict(filepath)
        length = len(datastrlist)
        data = [datastrlist[length - 1]]
        for j in range(length - 1):
            data.append([datastrlist[j].split(": ")[0],
                        float(datastrlist[j].split(": ")[1][:-1])
                         ])
        results.append(data)

#   calculate score


    # for i in range(len(results)):
    #     for j in range(len(results[i])):
            # max(results[i][j])
    print(str(results))



fileA = 'image_data/Angry-face-man.jpg'
get_score(fileA)
#
# fileB = 'image_data/AngryFace2.jpg'
# get_score(fileB)
#
# fileC = 'image_data/Happyman.png'
# get_score(fileC)
#
# fileD = 'image_data/Sad-Face.png'
# get_score(fileD)