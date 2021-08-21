# Import python libraries
import numpy as np
import cv2
from droiddekka.simplekalman.kalmanfilter import simplekalmanfilter as skf

debugMode=1

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray, 50, 190, 3)
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    contours, heirarchies = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_radius_thresh = 3 
    max_radius_thresh = 30 

    centers = [] 

    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
        return centers 

def main():
    
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('/home/arkaprabha/droiddekka/tests/video/video_randomball.avi')

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    k = skf(4,2) 
    k.state_process_setter(X=np.array([1, 1, 0.1,0.1]),dt=0.15,process_noise=2.25)
    

    while(True):
        # Read frame
        ret, frame = VideoCap.read()

        # Detect object
        centers = detect(frame)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (X, P) = k.predict()
            x,y = X[0:2]
            # Draw a rectangle as the predicted object position
            frame = cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)
            cv2.putText(frame, 'Prediction', (int(x+10), int(y-35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            # Update
            (x1, y1) = k.update(centers[0])
            print((x1,y1))
            # Draw a rectangle as the estimated object position
            frame = cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)
            cv2.putText(frame, 'Estimation', (int(x+10), int(y+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2) 

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    # execute main
    main()
