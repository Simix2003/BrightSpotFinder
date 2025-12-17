To create RESIDUAL :

python main.py --input "D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND" --output "D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND_RESIDUAL"

To Label the residuals:

python yolo_annotator.py --images D:\Imix\Lavori\Capgemini\3SUN\Immagini\FOUND_RESIDUAL --out "D:\Imix\Lavori\Capgemini\3SUN\BrightSpotFinder\YoloDataset" --classes bright_spot --val-ratio 0.2 --seed 42
