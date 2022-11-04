# text-detection-from-image

## Installation

### Installation on *nix system:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/text-detection-from-image` (current repository URL);
3. Run command `cd ./text-detection-from-image/` (cloned folder);
4. Run command `pip install gdown` (Install google cloud download lib for download weights model from google cloud);
5. Run command `bash ./install_project.sh` (Downloading libs, and models);
6. Run command `pip install -r ./requirements.txt`;
7. Run main.py script, you can chage original image URL in `function print_hi_text_detection():`. Output image will save in local folder `./results_images`.

### Installation and run FastAPI service:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/text-detection-from-image` (current repository URL);
3. Run command `cd ./text-detection-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run command `pip install -r ./requirements-fast-api.txt` (modules for FastAPI service);
7. Run command for start up FastAPI service `uvicorn app:app`;
8. Remove text from image by HTTP request `http://127.0.0.1:8003/text_detection/?url=https://img-9gag-fun.9cache.com/photo/axMNd31_460s.jpg` (IP and port will print in console when you start up service *step 7*. url= -- URL to original image).

### Installation and run FastAPI service with gunicorn:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/text-detection-from-image` (current repository URL);
3. Run command `cd ./text-detection-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run command `pip install -r ./requirements-fast-api.txt` (modules for FastAPI service);
7. Run command for start up FastAPI service `gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --timeout 600 --max-requests 5 --bind 0.0.0.0:8003`;
8. Remove text from image by HTTP request `http://127.0.0.1:8003/text_detection/?url=https://img-9gag-fun.9cache.com/photo/axMNd31_460s.jpg` (IP and port will print in console when you start up service *step 7*. url= -- URL to original image).

Note: FastAPI with unicorn "eat" a lot of memory and have memory leak, thats why you can use gunicorn service, witch will restart and clean memory every `--max-requests COUNT_REQUEST`
