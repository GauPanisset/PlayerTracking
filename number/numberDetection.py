import sys
import numpy as np
import cv2


def generate_data(train_set):
    im = cv2.imread(train_set)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if h>13 and h<20:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',im)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print("training complete")

    np.savetxt('generalsamples.data', samples)
    np.savetxt('generalresponses.data', responses)


def setup_model(root):
    samples = np.loadtxt(root + '/generalsamples.data', np.float32)
    responses = np.loadtxt(root + '/generalresponses.data', np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def detect_number(model, img, disp=False):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 13 and h < 20:
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                res.append(int((results[0][0])))
    if disp:
        print("%d%d:%d%d" % (res[3], res[2], res[1], res[0]))
    return res


def array_to_second(array, length=None):
    if length and len(array) != length:
        return 0
    return (10 * array[-1] + array[-2]) * 60 + 10 * array[-3] + array[-4]
