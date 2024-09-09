# AIAC: AI Art Curator

AIAC (AI Art Curator) is a conversational AI that interacts with users to recommend paintings based on their preferences. 
It leverages a hierarchical recommendation system built around art genres and artist nationalities.

## Requirements
To run the code, the following packages are needed:
- **Python**: 3.10.4
- **PyTorch**: 2.4.0+cu121
- **Transformers**: 4.44.2
- **Pandas**: 2.2.2

## Before running the program:
Download the dataset from [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) and generate the hierarchy tree:
```
python3 generatetree.py -d <path to the dataset folder>
```

## To run the program in an interactive environment:

```
python3 interactive.py -o <path_to_log_files> -d <path_to_dataset_folder> -r <path_to_recommended_images_folder> -m <LLM_model>
```

### System-level Commands:

- `clear`: Clears the console output.
- `exit`: Exits the program and saves the conversation history.