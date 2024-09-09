import torch
import time
import logging
import os
import ipdb
import argparse
import copy
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

ALL_GENRE = ['Expressionism', 'Abstractionism', 'Social Realism', 'Muralism', 'Impressionism', 'Surrealism', 'Realism', 'Byzantine Art', 'Post-Impressionism', 'Symbolism', 'Art Nouveau', 'Northern Renaissance', 'Suprematism', 'Cubism', 'Baroque', 'Romanticism', 'Primitivism', 'Mannerism', 'Proto Renaissance', 'Early Renaissance', 'High Renaissance', 'Neoplasticism', 'Pop Art', 'Abstract Expressionism']

ALL_COUNTRY = ['Italian', 'Russian', 'Mexican', 'French', 'Belgian', 'Spanish', 'Dutch', 'Austrian', 'Flemish', 'Greek', 'German', 'British', 'Jewish', 'Belarusian', 'Norwegian', 'Swiss', 'American']

ALL_ARTIST = ['Amedeo Modigliani', 'Vasiliy Kandinskiy', 'Diego Rivera', 'Claude Monet', 'Rene Magritte', 'Salvador Dali', 'Edouard Manet', 'Andrei Rublev', 'Vincent van Gogh', 'Gustav Klimt', 'Hieronymus Bosch', 'Kazimir Malevich', 'Mikhail Vrubel', 'Pablo Picasso', 'Peter Paul Rubens', 'Pierre-Auguste Renoir', 'Francisco Goya', 'Frida Kahlo', 'El Greco', 'Albrecht Dürer', 'Alfred Sisley', 'Pieter Bruegel', 'Marc Chagall', 'Giotto di Bondone', 'Sandro Botticelli', 'Caravaggio', 'Leonardo da Vinci', 'Diego Velazquez', 'Henri Matisse', 'Jan van Eyck', 'Edgar Degas', 'Rembrandt', 'Titian', 'Henri de Toulouse-Lautrec', 'Gustave Courbet', 'Camille Pissarro', 'William Turner', 'Edvard Munch', 'Paul Cezanne', 'Eugene Delacroix', 'Henri Rousseau', 'Georges Seurat', 'Paul Klee', 'Piet Mondrian', 'Joan Miro', 'Andy Warhol', 'Paul Gauguin', 'Raphael', 'Michelangelo', 'Jackson Pollock']

def setup_logging(args):
    """Sets up logging to both a file and the console."""
    os.makedirs(args.output_folder, exist_ok=True)
    log_filename = f"{args.output_folder}/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    logging.info("Logging arguments...")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

# Load and setup functions
def initialize_accelerator():
    """Initializes the Accelerator and returns GPU info."""
    accelerator = Accelerator()
    current_gpu_id = accelerator.process_index
    return accelerator, current_gpu_id

def load_model_and_tokenizer(model_path, device):
    """Loads the tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device).eval()

    return model, tokenizer

def print_gpu_memory_usage(current_gpu_id):
    """Prints GPU memory usage."""
    t_in_gb = torch.cuda.get_device_properties(current_gpu_id).total_memory / (1024 ** 3)
    r_in_gb = torch.cuda.memory_reserved(current_gpu_id) / (1024 ** 3)
    a_in_gb = torch.cuda.memory_allocated(current_gpu_id) / (1024 ** 3)
    logging.info(f' *GPU-{current_gpu_id}: total memory: {t_in_gb:.2f} GB, reserved memory: {r_in_gb:.2f} GB, '
                 f'allocated memory: {a_in_gb:.2f} GB, free memory: {(r_in_gb-a_in_gb):.2f} GB')

def generate_response(model, tokenizer, convo, max_new_tokens, temperature):
    """Generates a response from the model based on the entire conversation history."""
    # Tokenize the conversation
    tokenized_chat = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    prompt_len = tokenized_chat.shape[-1]
    output_tokens = model.generate(tokenized_chat, temperature=temperature, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)[0]

    # Generate output
    output_text = tokenizer.decode(output_tokens[prompt_len:], skip_special_tokens=True)

    return output_text

def system_command(user_prompt, conversation_history):
    """Handles user commands like 'exit', 'clear', and 'set'."""
    if user_prompt.lower() == 'exit':
        logging.info("Saving conversation history to the log.")
        for entry in conversation_history:
            logging.info(f"{entry['role']}: {entry['content']}")
        logging.info("Exiting program.")
        return 'exit'
    elif user_prompt.lower() == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        logging.info("Console cleared. Enter your next command.")
        return 'clear'
    elif user_prompt.lower() == 'help':
        logging.info("Available commands:")
        logging.info(" - 'exit': Quit the program.")
        logging.info(" - 'clear': Clear the current console output.")
        return 'help'
    return None