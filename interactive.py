import torch
import time
import logging
import os
import ipdb
import argparse
import copy
import os
import random
from pathlib import Path
from datetime import datetime
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import (
    ALL_GENRE, ALL_COUNTRY, ALL_ARTIST,
    setup_logging, initialize_accelerator, load_model_and_tokenizer,
    print_gpu_memory_usage, generate_response, system_command
)

from generatetree import traverse_tree

# Environment configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def check_criteria(model, tokenizer, convo, max_new_tokens, temperature, criteria):
    # We focus on two criterion genre and country, and only make recommendations(traverse tree) when two criterion are met.
    if criteria == 'genre':
        ALL_AVAILABLE_CRITERIA = ALL_GENRE
        check_message = "From the conversation so far, can we confirm the genre of the painting you are looking for? If no, please answer 'no'. If 'yes', please limit your response to 'yes, \{genre\}.', with no further conversations."
    elif criteria == 'country':
        ALL_AVAILABLE_CRITERIA = ALL_COUNTRY
        check_message = "From the conversation so far, can we confirm the country of the artist you are looking for? If no, please answer 'no'.  If 'yes', please limit your response to 'yes, \{adjectival forms of the country\}.', with no further conversations."
    elif criteria == 'artist':
        ALL_AVAILABLE_CRITERIA = ALL_ARTIST
        check_message = "From the conversation so far, can we confirm the artist? If no, please answer 'no'. If 'yes', please response with 'yes, \{full name of the artist\}.'"

    convo_check = copy.deepcopy(convo)
    convo_check.append({"role": "user", "content": {check_message}})

    response = generate_response(model, tokenizer, convo_check, max_new_tokens, temperature)

    if 'yes' in response.lower():
        match = check_availability(response, ALL_AVAILABLE_CRITERIA)
        return match
    else:
        return None

def check_availability(response, all_criteria):
    response_lower = response.lower()
    if any(criteria.lower() in response_lower for criteria in all_criteria):
        match = [criteria for criteria in all_criteria if criteria.lower() in response_lower][0]
        return match
    else:
        return None

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-folder", type=str, default = '/fs01/home/ama/workspace/aiac/convo/')
    parser.add_argument("-d", "--dataset-folder", type=str, default = '/fs01/home/ama/workspace/aiac/dataset/')
    parser.add_argument("-r", "--recommend-folder", type=str, default = '/fs01/home/ama/workspace/aiac/recommend/')
    parser.add_argument("-m", "--model-path", type=str, default = '/model-weights/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Setting up logging
    setup_logging(args)

    # Initialization
    image_folder = f'{args.dataset_folder}/images/images/'
    model_path = args.model_path
    system_prompt = "You are an AI Art Curator (AIAC) that interacts with users to recommend paintings. You will engage in a thoughtful dialogue with the user to understand their artistic tastes, focusing on the genre and the country of the artist. The available genres are: 'Expressionism', 'Abstractionism', 'Social Realism', 'Muralism', 'Impressionism', 'Surrealism', 'Realism', 'Byzantine Art', 'Post-Impressionism', 'Symbolism', 'Art Nouveau', 'Northern Renaissance', 'Suprematism', 'Cubism', 'Baroque', 'Romanticism', 'Primitivism', 'Mannerism', 'Proto Renaissance', 'Early Renaissance', 'High Renaissance', 'Neoplasticism', 'Pop Art', 'Abstract Expressionism'. Artists come from the following countries:  'Italian', 'Russian', 'Mexican', 'French', 'Belgian', 'Spanish', 'Dutch', 'Austrian', 'Flemish', 'Greek', 'German', 'British', 'Jewish', 'Belarusian', 'Norwegian', 'Swiss', 'American'. The available artists are: 'Amedeo Modigliani', 'Vasiliy Kandinskiy', 'Diego Rivera', 'Claude Monet', 'Rene Magritte', 'Salvador Dali', 'Edouard Manet', 'Andrei Rublev', 'Vincent van Gogh', 'Gustav Klimt', 'Hieronymus Bosch', 'Kazimir Malevich', 'Mikhail Vrubel', 'Pablo Picasso', 'Peter Paul Rubens', 'Pierre-Auguste Renoir', 'Francisco Goya', 'Frida Kahlo', 'El Greco', 'Albrecht Dürer', 'Alfred Sisley', 'Pieter Bruegel', 'Marc Chagall', 'Giotto di Bondone', 'Sandro Botticelli', 'Caravaggio', 'Leonardo da Vinci', 'Diego Velazquez', 'Henri Matisse', 'Jan van Eyck', 'Edgar Degas', 'Rembrandt', 'Titian', 'Henri de Toulouse-Lautrec', 'Gustave Courbet', 'Camille Pissarro', 'William Turner', 'Edvard Munch', 'Paul Cezanne', 'Eugene Delacroix', 'Henri Rousseau', 'Georges Seurat', 'Paul Klee', 'Piet Mondrian', 'Joan Miro', 'Andy Warhol', 'Paul Gauguin', 'Raphael', 'Michelangelo', 'Jackson Pollock'. If the user asks a question non-related to art, you can politely decline to answer and redirect the conversation back to art. Remember to be polite, respectful, and professional in your responses."

    # Initialize accelerator and load model/tokenizer
    accelerator, current_gpu_id = initialize_accelerator()
    logging.info(f" *loading model: Meta-Llama-3.1-8B-Instruct")

    model, tokenizer = load_model_and_tokenizer(model_path, device='auto')
    logging.info(f' *GPU-{current_gpu_id}: model and tokenizer loaded')

    print_gpu_memory_usage(current_gpu_id)

    # Initialize conversation history
    init_convo = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "How can I help you in your art exploration today?"}
    ]
    convo = copy.deepcopy(init_convo)
    confirmed_genre = None
    confirmed_country = None
    confirmed_artist = None

    # Multi-turn conversation loop
    print(f"===== Beginning of your conversation with AIAC =====")
    print('AIAC: How can I help you in your art exploration today? We will focus on genres and country of the artist to recommend paintings. (Type "Help" for more information.)')
    while True:
        user_prompt = input("User: ")

        # Handle user commands
        command_result = system_command(user_prompt, convo)
        if command_result == 'exit':
            break
        elif command_result in ['clear', 'help']:
            continue

        # Ensure user prompt is not empty
        if not user_prompt.strip():
            print("How can I help you in your art exploration today? Please let me know your preferences.")
            continue

        # Append the user input to conversation history
        convo.append({"role": "user", "content": user_prompt})

        # First thing is to check whether we can get the artist name directly from the conversation history
        if not confirmed_artist:
            confirmed_artist = check_criteria(model, tokenizer, convo, args.max_new_tokens, args.temperature, 'artist')
            if confirmed_artist:
                aiac_output = f'AIAC: I see you are interested in paintings from {confirmed_artist}.'
                print(aiac_output)
                convo.append({"role": "assistant", "content": aiac_output})

        # If we can't get the artist name, we will try to get the genre and country
        if not confirmed_artist and not confirmed_genre:
            confirmed_genre = check_criteria(model, tokenizer, convo, args.max_new_tokens, args.temperature, 'genre')
            if confirmed_genre:
                aiac_output = f'AIAC: I see you are interested in {confirmed_genre} paintings.'
                if not confirmed_country:
                    aiac_output += ' What would be the country of the artist?'
                print(aiac_output)
                convo.append({"role": "assistant", "content": aiac_output})
                if not confirmed_country:
                    continue

        if not confirmed_artist and not confirmed_country:
            confirmed_country = check_criteria(model, tokenizer, convo, args.max_new_tokens, args.temperature, 'country')
            if confirmed_country:
                aiac_output = f'AIAC: I see you are interested in paintings from {confirmed_country} artists.'
                if not confirmed_genre:
                    aiac_output += ' What would be the genre of the painting?'
                print(aiac_output)
                convo.append({"role": "assistant", "content": aiac_output})
                if not confirmed_genre:
                    continue

        # If we have either the artist name or both genre and country, we can make a recommendation:
        if confirmed_artist or (confirmed_genre and confirmed_country):
            if confirmed_artist:
                # Having the artist name is the most specific, so we will use it to make a recommendation
                recommendation = confirmed_artist
            else:
                # Otherwise, we will use the genre and country to make a recommendation
                try:
                    # Get the hierarchy from the saved file
                    hierarchy = torch.load(f'{args.dataset_folder}/hierarchy.pt', weights_only=True)
                except FileNotFoundError:
                    logging.error("Hierarchy file not found. Please run 'python generatetree.py' to generate the hierarchy.")
                    system_command('exit', convo)
                    break
                user_preferences = [confirmed_genre, confirmed_country]
                recommendation = traverse_tree(hierarchy, user_preferences)
                # If no recommendation is found, reset the conversation
                if not recommendation:
                    print(f"AIAC: I apologize, but our dataset does not contain any paintings from the {confirmed_genre} genre by artists from {confirmed_country}. No worries, though! Let's start over and explore other options.")
                    confirmed_genre = None
                    confirmed_country = None
                    # reset convo
                    convo = copy.deepcopy(init_convo)
                    continue
                else:
                    # If a recommendation is found, we will confirm it with the user
                    confirmed_artist = list(recommendation.keys())[0]
            # Make the recommendation
            ending_message = f'Great! I have a couple of pieces from {confirmed_artist} that you might love. Please check out the recommendation folder!'
            print(f'AIAC: {ending_message}')
            try:
                # Create the recommendation folder if it does not exist
                Path(args.recommend_folder).mkdir(parents=True, exist_ok=True)
                # Move some files to the recommendation folder
                image_folder += confirmed_artist.replace(' ', '_')
                image_folder = Path(image_folder)
                random_image = random.choice([f for f in image_folder.iterdir() if f.is_file()])
                new_name = f"{args.recommend_folder}/{confirmed_artist.replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                shutil.copy(random_image, new_name)
            except FileNotFoundError:
                print(f"The source file {random_image} was not found.")

            convo.append({"role": "assistant", "content": ending_message})
            system_command('exit', convo)
            break

        aiac_output = generate_response(model, tokenizer, convo, args.max_new_tokens, args.temperature)

        # Add the system response to the conversation history
        convo.append({"role": "assistant", "content": aiac_output})

        # Display the system's response
        print(f'AIAC: {aiac_output}\n')

# Run the main program
if __name__ == "__main__":
    main()