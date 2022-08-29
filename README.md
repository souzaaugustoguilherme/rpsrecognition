# Rock Paper Scissors Recognition
This project uses an CNN (Convolutional Neural Network) with PyTorch framework to recognize the Rock, Paper and Scissor hand gesture.  
## Usage:
### capture_images.py
```
$ ./capture_images.py <label_name> <num_samples>
# label_name: The label e.g. rock, paper, scissor.
# num_samples: How many images will be stored.
```
Note: The sum of images must be an even number.
### create_csv.py
```
$ ./create_csv.py

```
Will read all the files from the rps_images dir and create and csv file e.g.:  
```
Img,Label
rock_1.jpg,0
paper_1.jpg,1
scissor_1.jpg,2
```
