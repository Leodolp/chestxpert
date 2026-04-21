# Chestxpert

This is a personnal MLOps learning project.

## Project Structure  
```
chestxpert/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py
│   ├── predict.py
├── data/
│   └── sample_images
└── notebooks/
    └── messy exploration notebooks
```

## Usage examples
__Command line:__  
``` 
python src/predict.py --image data/00000001_000_Cardiomegalie.png
```
__With docker:__  
Pull image from docker hub  
```
docker pull leopolp/chestxpert:latest
```
_OR_  
Build the image from Dockerfile  
```
docker build -f docker/Dockerfile.cpu -t chestxpert:latest .
```
then  
```
docker run -v ./data/00000001_000_Cardiomegalie.png:/app/data/00000001_000_Cardiomegalie.png:ro chestxpert --image data/00000001_000_Cardiomegalie.png
```
to run the provided example image  