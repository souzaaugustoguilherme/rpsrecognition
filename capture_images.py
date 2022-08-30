#!/usr/bin/env python3
import cv2
import os
import sys

DESC = '''
Usage: python capture_images.py <label_name> <num_samples>

Press 'a' to start/pause the image collecting process.
Press 'q' to quit.
'''

if __name__ == "__main__":
    try:
        label_name = sys.argv[1]
        num_samples = int(sys.argv[2])
    except:
        print("Arguments missing.")
        print(DESC)
        exit(-1)

    IMG_SAVE_PATH = 'rps_images'

    try:
        os.mkdir(IMG_SAVE_PATH)
    except FileExistsError:
        print(f"{IMG_SAVE_PATH} directory already exists, all images will be saved along with existing items in this folder.")

    cap = cv2.VideoCapture(0)

    start = False
    count = 0

    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 0, 0), 2)

        if not ret:
            continue
        if count == num_samples:
            break

        if start:
            roi = frame[100:500, 100:500]
            save_path = os.path.join(IMG_SAVE_PATH, f"{label_name}_{count + 1}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            count += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Collecting images", frame)

        k = cv2.waitKey(10)
        if k == ord('a'):
            start = True
        if k == ord('q'):
            break

    print("\n{} image(s) saved to {}.".format(count, IMG_SAVE_PATH))
    cap.release()
    cv2.destroyAllWindows()
