import sys
sys.path.append('../')
from EmoPy.src.fermodel import FERModel

emotions = [['surprise', 0], ['happiness', 0], ['anger', 0], ['fear', 0], ['disgust', 0], ['calm', 0]]

target_emotions = ['surprise', 'anger', 'fear', 'calm']
model = FERModel(target_emotions, verbose=True)

target_emotions2 = ['surprise', 'disgust', 'happiness']
model2 = FERModel(target_emotions2, verbose=True)

target_emotions3 = ['surprise', 'anger', 'fear']
model3 = FERModel(target_emotions3, verbose=True)

target_emotions4 = ['anger', 'fear', 'calm']
model4 = FERModel(target_emotions4, verbose=True)

target_emotions5 = ['anger', 'calm', 'happiness']
model5 = FERModel(target_emotions5, verbose=True)

target_emotions6 = ['anger', 'fear', 'disgust']
model6 = FERModel(target_emotions6, verbose=True)

target_emotions7 = ['disgust', 'surprise', 'calm']
model7 = FERModel(target_emotions7, verbose=True)

target_emotions8 = ['surprise', 'disgust', 'sadness']
model8 = FERModel(target_emotions8, verbose=True)

target_emotions9 = ['anger', 'happiness']
model9 = FERModel(target_emotions9, verbose=True)

print('Predicting on happy image...')
model.predict('image_data/sample_happy_image.png')

print('Predicting on disgust image...')
model.predict('image_data/sample_disgust_image.png')

print('Predicting on anger image...')
model.predict('image_data/sample_anger_image2.png')

def predictImage(filepath):
    print('Predicting on ' + filepath)
    results = [
        model.predict(filepath).split(';')[:len(target_emotions)],
        model2.predict(filepath).split(';')[:len(target_emotions2)],
        model3.predict(filepath).split(';')[:len(target_emotions3)],
        model4.predict(filepath).split(';')[:len(target_emotions4)],
        model5.predict(filepath).split(';')[:len(target_emotions5)],
        model6.predict(filepath).split(';')[:len(target_emotions6)],
        model7.predict(filepath).split(';')[:len(target_emotions7)],
        model8.predict(filepath).split(';')[:len(target_emotions8)],
        model9.predict(filepath).split(';')[:len(target_emotions9)]
    ]
    # for i in range(len(results)):
    #     for j in range(len(results[i])):
            # max(results[i][j])
    print(str(results))



fileA = 'image_data/Angry-face-man.jpg'
predictImage(fileA)
#
# fileB = 'image_data/AngryFace2.jpg'
# predictImage(fileB)
#
# fileC = 'image_data/Happyman.png'
# predictImage(fileC)
#
# fileD = 'image_data/Sad-Face.png'
# predictImage(fileD)