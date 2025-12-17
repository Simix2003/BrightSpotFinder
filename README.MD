To create RESIDUAL :

python main.py --input "D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND" --output "D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND_RESIDUAL"

To Label the residuals:

python yolo_annotator.py --images D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND_RESIDUAL --out "D:\Imix\Lavori\Capgemini\3SUN\BrightSpotFinder\YoloDataset" --classes bright_spot --val-ratio 0.2 --seed 42

Mouse drag: draw a bounding box

d: delete last box

x: clear all boxes for that image

c: change class id

s: save

n: save + next image

p: save + previous image

q: save + quit



# Mirror the dataset (creates 4x the images)
python mirror_dataset.py --dataset "D:\Imix\Lavori\Capgemini\3SUN\BrightSpotFinder\YoloDataset2"

# Visualize the dataset
python visualize_dataset.py --dataset "D:\Imix\Lavori\Capgemini\3SUN\BrightSpotFinder\YoloDataset2"