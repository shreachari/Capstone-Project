# lets make the client code
import socket, cv2, pickle, struct
import time
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import random
from collections import deque
import json
from finger_counter import landmarker_and_result, draw_landmarks_on_image
import os
from openai import OpenAI
import math
import re

# CONSTANTS
SERVER_SOCKET = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
HOST_NAME  = socket.gethostname()
HOST_IP = socket.gethostbyname(HOST_NAME)
print('HOST IP:',HOST_IP)
PORT = 5050
SOCKET_ADDRESS = (HOST_IP,PORT)
client_socket = None


def create_socket():
    SERVER_SOCKET.bind(SOCKET_ADDRESS)
    SERVER_SOCKET.listen(5)
    print("LISTENING AT:",SOCKET_ADDRESS)

def close_socket():
    client_socket.close()
    SERVER_SOCKET.close()
    print("Closing socket.")
    

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    try:
        # Get Data
        hand_landmarks_list = detection_result.hand_landmarks
        # Counter
        numRaised = 0
        # for each hand...
        for idx in range(len(hand_landmarks_list)):
            # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
            hand_landmarks = hand_landmarks_list[idx]
            # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
            for i in range(8,21,4):
                # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
                tip_y = hand_landmarks[i].y
                dip_y = hand_landmarks[i-1].y
                pip_y = hand_landmarks[i-2].y
                mcp_y = hand_landmarks[i-3].y
                if tip_y < min(dip_y,pip_y,mcp_y):
                    numRaised += 1
            # for the thumb
            # use direction vector from wrist to base of thumb to determine "raised"
            tip_x = hand_landmarks[4].x
            dip_x = hand_landmarks[3].x
            pip_x = hand_landmarks[2].x
            mcp_x = hand_landmarks[1].x
            palm_x = hand_landmarks[0].x
            if mcp_x > palm_x:
                if tip_x > max(dip_x,pip_x,mcp_x):
                    numRaised += 1
            else:
                if tip_x < min(dip_x,pip_x,mcp_x):
                    numRaised += 1

        # display number of fingers raised on the image
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape
        text_x = int(hand_landmarks[0].x * width) - 100
        text_y = int(hand_landmarks[0].y * height) + 50
        cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
                            org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                            fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
        return annotated_image, numRaised
    except:
        return rgb_image, -1

def finger_counting_test(seq_length):
    data = b""
    payload_size = struct.calcsize("Q")
    hand_landmarker = landmarker_and_result()

    sequence = [random.randint(1, 5) for _ in range(seq_length)]
    index = 0
    print(sequence)
    num_tries = 3
    last_num = 0
    previous_elements = deque(maxlen=5)
    previous_elements.append(0)
    previous_elements.append(0)
    previous_elements.append(0)
    previous_elements.append(0)
    previous_elements.append(0)

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024) # 4K
            if not packet: break
            data+=packet
        
        if len(data) < payload_size:
            continue
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q",packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = pickle.loads(frame_data)
                
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_landmarker.detect_async(frame)
        # draw landmarks on frame
        frame = draw_landmarks_on_image(frame,hand_landmarker.result)
        # count number of fingers raised
        frame, num_raised = count_fingers_raised(frame,hand_landmarker.result)
        
        previous_elements.append(num_raised)
        
        if all(item == previous_elements[0] for item in previous_elements) and num_raised != last_num and num_raised != -1:
            # print(num_raised)
            last_num = num_raised
            if num_raised == 0:
                continue
            else:
                if num_raised == sequence[index]:
                    print("PLEASE DISPLAY NEXT VALUE")
                    index+=1
                else:
                    num_tries-=1
                    # print("NUM TRIES LEFT: " + str(num_tries))
                    
        
            if index == len(sequence):                
                # clean up
                hand_landmarker.close()
                cv2.destroyAllWindows()
                
                return True
        
            if num_tries == 0:
                # clean up
                hand_landmarker.close()
                cv2.destroyAllWindows()
                
                return False
        # cv2.imshow("RECIEVING",frame)

        key = cv2.waitKey(1) & 0xFF
        if key  == ord('q'):
            break
    
    # release everything
    hand_landmarker.close()
    cv2.destroyAllWindows()

def distance(x,y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) 

def nodding_test(questions):
    question_list = list(questions.keys())
    answer_list = list(questions.values())
    index = 0
    question = question_list[index]
    answer = answer_list[index]
    print(question)
    prev_gesture = False
    
    data = b""
    payload_size = struct.calcsize("Q")
    #params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #path to face cascde
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #function to get coordinates
    def get_coords(p1):
        try: return int(p1[0][0][0]), int(p1[0][0][1])
        except: return int(p1[0][0]), int(p1[0][1])

    #define font and text color
    font = cv2.FONT_HERSHEY_SIMPLEX


    #define movement thresholds
    max_head_movement = 20
    movement_threshold = 50
    gesture_threshold = 175

    #find the face in the image
    face_found = False
    frame_num = 0
    gesture = False
    x_movement = 0
    y_movement = 0
    gesture_show = 60 #number of frames a gesture is shown

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024) # 4K
            if not packet: break
            data+=packet
        
        if len(data) < payload_size:
            continue
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q",packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = pickle.loads(frame_data)
        
        while not face_found:
            # Take first frame and find corners in it
            frame_num += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face_center = x+w/2, y+h/3
                p0 = np.array([[face_center]], np.float32)
                face_found = True
            cv2.waitKey(1)

        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        cv2.circle(frame, get_coords(p1), 4, (0,0,255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255,0,0))
        
        #get the xy coordinates for points p0 and p1
        a,b = get_coords(p0), get_coords(p1)
        x_movement += abs(a[0]-b[0])
        y_movement += abs(a[1]-b[1])
        
        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame,text,(50,50), font, 0.8,(0,0,255),2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame,text,(50,100), font, 0.8,(0,0,255),2)

        if x_movement > gesture_threshold:
            gesture = 'No'
        if y_movement > gesture_threshold:
            gesture = 'Yes'
        if gesture and gesture_show > 0:
            cv2.putText(frame,'Gesture Detected: ' + gesture,(50,50), font, 1.2,(0,0,255),3)
            if gesture != prev_gesture:
                print(gesture)
                prev_gesture = gesture
                # user answered incorrectly
                if gesture != answer:
                    return False
                else: # user answered correctly
                    index += 1
                    # user has answered all correctly
                    if index == len(questions):
                        return True
                    question = question_list[index]
                    answer = answer_list[index]
                    print(question)
            gesture_show -=1
        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 60 #number of frames a gesture is shown
        if not gesture:
            prev_gesture = gesture
            
        p0 = p1

        # cv2.imshow('RECIEVING',frame)
        key = cv2.waitKey(1) & 0xFF
        if key  == ord('q'):
            break

    cv2.destroyAllWindows()

def receive_user_input_from_server():
    data = client_socket.recv(1024)
    if not data:
        return None
    return json.loads(data.decode())

def get_user_profile(profiles, user_input):
    first_name = user_input.get("first_name")
    last_name = user_input.get("last_name")
    middle_name = user_input.get("middle_name")

    user_profile = None
    for profile in profiles:
        if profile["first_name"].lower() == first_name.lower() and profile["last_name"].lower() == last_name.lower():
            if "middle_name" in profile:
                if profile["middle_name"].lower() != middle_name.lower():
                    return None
            else:
                if middle_name:
                    return None
            user_profile = profile
    return user_profile

def send_response_to_server(response):
    client_socket.sendall(response.encode())


def generate_questions(profile, client):
    prompt = f'''Give me 3 (non rhyme based) simple yes or no questions based on the provided user profile which can also be answered using this user profile. 
    Don't just ask questions from the first three facts, use the entire profile.
    Direct the question toward the user.
    Ensure that some of the answers are yes and some are no. 
    Enclose the questions in quotations and put the answer in parenthesis after the question.
    Ensure all questions and answers are accurate to the profile.
    Profile: {profile}.
    '''
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    question = completion.choices[0].message.content.strip()    
    return question

def clean_questions(input_text):
    # Initialize an empty dictionary to store the questions and answers
    qa_dict = {}

    # Find all question-answer pairs using regex
    pairs = re.findall(r'"(.*?)"\s*\(\s*(.*?)\s*\)', input_text)

    # Iterate over each pair
    for question, answer in pairs:
        # Add the question and answer to the dictionary
        qa_dict[question.strip()] = answer.strip()

    # Return the resulting dictionary
    return qa_dict


# def ask_yes_no_questions(question_dict):
    
#     for question in question_dict:
#         print("hi")
#     return False


def main():
    start_time = time.time()
    create_socket()
    try:
        global client_socket
        while client_socket is None:
            client_socket,addr = SERVER_SOCKET.accept()
            print('GOT CONNECTION FROM:',addr)
            
        user_input = receive_user_input_from_server()

        # Load user profiles from JSON
        with open("user_profiles.json", "r") as file:
            profiles = json.load(file)

        # Process user profile
        user_profile = get_user_profile(profiles, user_input)
        if user_profile:
            response = json.dumps(user_profile["facts"], indent=2)
        else:
            response = "Fail"        
        # Send response back to server
        send_response_to_server(response)
        if response != "Fail":
            # OpenAI setup
            os.environ['OPENAI_API_KEY'] = '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'
            client = OpenAI()
            # Setting environment variable
            
            # Generate questions
            questions = generate_questions(user_profile, client)
            # Clean questions
            question_dict = clean_questions(questions)
            # print(question_dict)
            # create_socket()
            ret = nodding_test(question_dict)
            if ret:
                print("Nodding test has passed.")
            else:
                print("Nodding test has failed")
                close_socket()
                return
            
            ret = finger_counting_test(5)
            if ret:
                print("Finger counting test has passed.")
            else:
                print("Finger counting test has failed.")
                close_socket()
                return
            
            print("APPROVED")
        close_socket()
    except socket.error as e:
        print("Socket error:", e)
        close_socket()
    except KeyboardInterrupt:
        close_socket()
        print("Server stopped by KeyboardInterrupt")
    
    end_time = time.time()
    latency = end_time - start_time
    print("Latency of main function:", latency)

if __name__ == "__main__":
    main()