import time
import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime

path = 'KnownFaces'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name}, {dtString}')

encodeListKnown = findEncodings(images)
print("הפענוח הושלם")
sound = 65

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)



        if matches[matchIndex]:
            name = classNames[matchIndex]
            name_for_print = ""
            error_faceNO = "שגיאה: לא נמצא השם של התלמיד"

            if name == "Ori-Ger":
                name_for_print = "אורי גרשוב"

            elif name == "Matan":
                name_for_print = "מתן דנצין"

            elif name == "Gavriel":
                name_for_print = "גבריאל חנוכייב"

            elif name == "Liam":
                name_for_print = "ליאם סוסלוב"

            elif name == "Elian-Gor":
                name_for_print = "אליאן גורדין"

            elif name == "Erica":
                name_for_print = "אריקה רחמימוב"

            elif name == "Tom":
                name_for_print = "טום רוכמן"

            elif name == "Elian-Dan":
                name_for_print = "אליאן דן"

            elif name == "michelle":
                name_for_print = "מישל לנצמן"

            elif name == "Alin":
                name_for_print = "אלין קורקוס"

            elif name == "Michal":
                name_for_print = "מיכל דגן"

            elif name == "Ediel":
                name_for_print = "אדיאל אברמוב"

            elif name == "Alina":
                name_for_print = "אלינה בוצ'קובה"

            elif name == "Riki":
                name_for_print = "ריקי ממן"

            elif name == "Nazer":
                name_for_print = "נזר ריסוחין"

            elif name == "Alice":
                name_for_print = "אליס יוסילביץ"

            elif name == "Maor":
                name_for_print = "מאור גולדנברג"

            elif name == "Galya":
                name_for_print = "גליה רכס"

            elif name == "Danilo":
                name_for_print = "דנילו אנדרייב"

            elif name == "David":
                name_for_print = "דוד גרינברג"

            elif name == "Michael":
                name_for_print = "מיכאל סמירנוב"

            elif name == "Leol":
                name_for_print = "לאול קבדה"

            elif name == "Roman":
                name_for_print = "רומן לוגשקין"

            elif name == "Hodaya":
                name_for_print = "הודיה מומו"

            elif name == "Daniel":
                name_for_print = "דניאל סולומון"

            elif name == "Ayala":
                name_for_print = "איילה נאוגוקר"

            elif name == "Elad":
                name_for_print = "אלעד רודובסקי"

            elif name == "Yonatan":
                name_for_print = "יונתן שוורצר"

            elif name == "Yan":
                name_for_print = "יאן קוסטנוביץ'"

            elif name == "Yahav":
                name_for_print = "יהב גילילוב"

            elif name == "Ido":
                name_for_print = "עידו מצקפלי"

            #  +============================+
            #  |  כאן נית ן להוסיף תלמידים! |
            #  +============================+

            else:
                name_for_print = error_faceNO


            print(name_for_print) # מה שיהיה כתוב ב LCD
            print("כמות התווים:", len(name_for_print), name_for_print)
            print(name) # באנגלית, מה שכתוב על המסך!



            #name1 = "start " + name + ".mp3"
            #print(name1)
            #os.system(name1)




            # כתיבת שם התלמיד על מסך LCD ברסברי פי 5.




            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

            sound = sound + sound

        if sound > 70:
            name1 = "start " + name + ".mp3"
            print(name1)
            print("מפעיל שמע:", name_for_print)
            os.system(name1)
            sound = 1

    cv2.imshow("Class CV", img)
    cv2.waitKey(1)