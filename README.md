# Capstone-Project

## Overview

This repository consists of Shrea Chari's Capstone Project for the UCLA M.S. Computer Science degree. It contains code to run an application that checks the identity of a user based on a provided user profile. The application requires user input in the form of text, head movements, and finger movements.

## Setup

1. Clone the repository
2. Ensure python is installed
3. Install all dependencies by running ```pip install -r requirements.txt```
4. Ensure docker is installed and the daemon is running
5. Open two terminal windows and ensure you are in the repository folder
6. In one window run the following commands:
    a. ```docker build -t capstone .```
    b. ```docker run -p 5050:5050 -it capstone```
7. In the other window, run the command ```python client.py```
8. Follow the prompts:
    - You will first need to enter your full name
    - Next, you will nod "yes" or "no" to respond to the questions provided
    - Lastly, you will need to display the numerical sequence printed by holding up your fingers on one hand. Reset to 0 fingers in between each digit.

## Important notes

- You will need to ensure that the ```HOST_IP``` variable in the file client.py is set to the IP address where the server will be run.
- You will need to ensure that ```OPENAI_API_KEY``` is set in line 385 of server.py: ```os.environ['OPENAI_API_KEY'] = 'api-key'```. You can obtain an API key from [OpenAI](https://openai.com/index/openai-api/)

- The port number can be changed. Ensure that it is changed in the files: client.py, server.py, and Dockerfile. The port number must be changed in the command ```docker run -p 5050:5050 -it capstone``` as well.

## References
- Socket programming to send and receive webcam video. https://pyshine.com/Socket-programming-and-openc/, 2020. Website.
- Docker. https://www.docker.com/, 2024. Website.
- Aws ec2. https://aws.amazon.com/pm/ec2/?gclid=Cj0KCQjwu8uyBhC6ARIsAKwBGpRcNt8GsvFl0U11_hjtWHUh677shJNVwB06mPNQ64gL-v2ctvYlCd0aAtu7EALw_wcB&trk=36c6da98-7b20-48fa8225-4784bced9843&sc_channel=ps&ef_id=Cj0KCQjwu8uyBhC6ARIsAKwBGpRcNt8GsvFl0U11_hjtWHUh677shJNVwB06mPNQ64gL-v2ctvYlCd0aAtu7EALw_wcB:G:s&s_kwcid=AL!4422!3!467723097970!e!!g!!aws%20ec2!11198711716!118263955828, 2024. Website.
- Thiago Avelino. haarcascade frontalface alt. https://github.com/avelino/python-opencv-detect/blob/master/haarcascade_frontalface_alt.xml, 2011. GitHub repository.
- Yunmeng Dong, Gaochao Xu, Meng Zhang, and Xiangyu Meng. A high-efficient joint ’cloud-edge’ aware strategy for task deployment and load balancing. IEEE Access, 2021, 2021. doi:10.1109/ACCESS.2021.3051672. Received December 5, 2020, accepted December 16, 2020, published online January 14, 2021, current version January 22, 2021.
- Stephen Meschke. head nod detection. https://gist.github.com/smeschke/e59a9f5a40f0b0ed73305d34695d916b, 2019. GitHub repository.
- Weisong Shi, Jie Cao, Quan Zhang, Youhuizi Li, and Lanyu Xu. Edge computing: Vision and challenges. IEEE Internet of Things Journal, 3(5):637, 2016.
- Owen Talmo. finger-counter. https://github.com/OwenTalmo/finger-counter/tree/master, 2023. GitHub repository.
- Alexandre da Silva Veith, Marcos Dias de Assuncao, and Laurent Lefèvre. Latency-aware placement of data stream analytics on edge computing. In Service-Oriented Computing, pages 215–229, Hangzhou, Zhejiang, China, 2018. HAL Id: hal-01875936.
- Qi Zhang, Lu Cheng, and Raouf Boutaba. Cloud computing: State-of-the-art and research challenges. Journal of Internet Services and Applications, 1(1):7–18, 2010. doi: 10.1007/s13174-010-0007-6.
