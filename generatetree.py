import pandas as pd
import ipdb
import pprint
import torch
import argparse

# Check the structure of the dataset
# print(df_exploded.head())

def build_hierarchy(df, levels):
    tree = {}
    for _, row in df.iterrows():
        current_level = tree
        for level in levels:
            value = row[level]
            if value not in current_level:
                if level == 'name':
                    current_level[value] = f'path/to/{value}-images'
                else:
                    current_level[value] = {}
            current_level = current_level[value]
    return tree

def traverse_tree(tree, user_preferences):
    current_node = tree
    for preference in user_preferences:
        if preference in current_node:
            current_node = current_node[preference]
        else:
            return None
    return current_node

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-folder", type=str, default = '/fs01/home/ama/workspace/aiac/dataset/')
    args = parser.parse_args()

    # Focusing on genre and nationality
    hierarchy_columns = ['genre', 'nationality','name']

    # Load the csv file
    df = pd.read_csv(f'{args.dataset_folder}/artists.csv')

    # Some columns have multiple values separated by commas, we break them into individual rows
    df['genre'] = df['genre'].str.split(',')
    df['nationality'] = df['nationality'].str.split(',')
    df_exploded = df.explode('genre').explode('nationality')

    hierarchy = build_hierarchy(df_exploded, hierarchy_columns)
    torch.save(hierarchy, './hierarchy.pt')
    # hierarchy = torch.load('./hierarchy.pt')
    pprint.pp(hierarchy)

    user_preferences = ['Expressionism', 'Russian']
    recommendations = traverse_tree(hierarchy, user_preferences)
    ipdb.set_trace()
    print(recommendations)