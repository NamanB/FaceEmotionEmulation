import sys
sys.path.append('../')
from EmoPy.src.fermodel import FERModel

emotions = [['surprise', 0], ['happiness', 0], ['anger', 0], ['fear', 0], ['disgust', 0], ['calm', 0]]
faces = ['image_data/angry.png','image_data/calm.png','image_data/disgust.png','image_data/fear.png','image_data/happiness.png','image_data/surprise.png']

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


def extract_data(filepath):
    print('Predicting on ' + filepath)

    #   format data
    results = []
    for i in range(len(models)):
        datastrlist = models[i].predict(filepath)
        length = len(datastrlist)
        data = [datastrlist[length - 1]]
        for j in range(length - 1):
            data.append([datastrlist[j].split(": ")[0],
                         float(datastrlist[j].split(": ")[1][:-1])
                         ])
        results.append(data)

    # for i in range(len(results)):
    #     for j in range(len(results[i])):
    # max(results[i][j])
    print(str(results))
    return results


def get_score(result_a, result_b):
    score = 0
    for i in range(len(result_a)):
        if result_a[i][0] == result_b[i][0]:
            score += 100
    return round(score / 9.0)


def calculate_score(filepath_a, index):
    return print(get_score(extract_data(filepath_a), extract_data(faces[index])))

# print('Predicting on happy image...')
# models[0].predict('image_data/sample_happy_image.png')
#
# print('Predicting on disgust image...')
# models[0].predict('image_data/sample_disgust_image.png')
#
# print('Predicting on anger image...')
# models[0].predict('image_data/sample_anger_image2.png')


# fileA = 'image_data/Angry-face-man.jpg'
# fileAb = 'image_data/sample_anger_image2.png'
# resultA = extract_data(fileA)
# resultB = extract_data(fileAb)
# print(get_score(resultA, resultB))

# calculate_score('image_data/AngryFace2.jpg', 'image_data/Sad-Face.png')

#
# fileB = 'image_data/AngryFace2.jpg'
# extract_data(fileB)
#
# fileC = 'image_data/Happyman.png'
# extract_data(fileC)
#
# fileD = 'image_data/Sad-Face.png'
# get_score(fileD)