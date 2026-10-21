# Automated Document Routing Pipeline

An end-to-end Deep Learning pipeline that classifies document images into 16 categories (RVL-CDIP) and automatically routes them to business departments (e.g., Finance, HR) or a human review queue based on confidence thresholds.

## Project Structure
- `doc_router_new_final.ipynb`: Notebook used for data extraction, transfer learning (ResNet18), and threshold analysis.
- `route_folder.py`: The production script that processes an input folder and routes images.
- `routing_config.json`: Configuration file containing class names, team mappings, and confidence threshold (0.70).
- `resnet18.pth`: The trained model weights (Note: If this is too large, it can be downloaded from Releases).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place document images in an `input/` folder.
3. Run the script: `python route_folder.py --input input`
4. Check the `output/` folder for routed images and the generated CSV report.
