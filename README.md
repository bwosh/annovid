# ANNOVID v.0.2

The projects is set of tools prepared for pre- or postprocessing data in Computer Vision domain. Fow images aw well as videos. Main goal is to have codebase for repetetive tasks in many projects. 

# Demo 

For showcase/reference please go to **[DEMO](./demo.ipynb) notebook**.  
Check samples below for an idea:  

**Fast video annotation**:  
![bbox sample](./assets/video_bbox.png)
*Frame from video: https://www.youtube.com/watch?v=MNn9qKG2UFI which was used during tests in ths project.* 

**Object apperance heatmaps**:  
![bbox sample](./assets/heatmap.png)

**Ultra-quick baseline annotations**:  
![bbox sample](./assets/bbox.png)

# Status/Things to be done
- visualuzation: processing images/folders/videos
- visualuzation:  video **heatmaps**
 - setup file for making library
 - (optionally) publishing to PyPI

 refer do "TODO"-s in code for more.

# Features 
The code addresses taksks like:
- image **representations** (PIL/OpenCV/RGB vs BRG issues)
- image **cropping** & **scaling**
- **bbox** notation standarization
- video heatmaps  

Visual helper methods include:
- bounding box drawing

# Dependencies
Code uses Python 3 syntax.   

**Core code uses:**  
- [OpenCV](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/) 
- [Pillow](https://pypi.org/project/tqdm/)
- [tqdm](https://pypi.org/project/tqdm/)

**Sample are using additionally:**  
- PyTorch (torch, torchvision)

# License
This code is licensed with [Apache License Version 2.0](./LICENSE).  
Original author: Błażej Matuszewski, repository: [bwosh/annovid](https://github.com/bwosh/annovid).  
Used libraries are licensed separately. Please review them as well.