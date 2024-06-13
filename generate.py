from trdg.generators import GeneratorFromStrings
from tqdm.auto import tqdm
import os
import pandas as pd
import random

NUM_IMAGES_TO_SAVE = 1100000
TRAIN_PERCENT = 70
VAL_PERCENT = 20
TEST_PERCENT = 10

OUTPUT_DIRS = {
    "train": "train",
    "val": "val",
    "test": "test"
}

fonts = [
    '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
    '/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf',
    '/System/Library/Fonts/Supplemental/Comic Sans MS.ttf',
    '/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf',
    '/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf',
    '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf',
    '/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf',
    '/System/Library/Fonts/Supplemental/Courier New Bold Italic.ttf',
    '/System/Library/Fonts/Supplemental/Courier New.ttf',
    '/System/Library/Fonts/Supplemental/Courier New Italic.ttf',
    '/System/Library/Fonts/Supplemental/Courier New Bold.ttf',
]

# Load the dictionary
df = pd.read_fwf("./generator/output3.txt", header=None, names=["term"])

# Filter out NaN values
all_words = df["term"].dropna().unique().tolist()
print(len(all_words))

def generate_dark_shades_hex(base_color):
    shades = []
    if base_color == "blue":
        for i in range(50, 200, 10):  # Темные оттенки синего
            shades.append(f'#0000{i:02x}')
    elif base_color == "green":
        for i in range(50, 200, 10):  # Темные оттенки зеленого
            shades.append(f'#00{i:02x}00')
    elif base_color == "red":
        for i in range(50, 200, 10):  # Темные оттенки красного
            shades.append(f'#{i:02x}0000')
    return shades

def get_random_parameters(black_gray_ratio = 0.9):
    gray_shades = [f'#{i:02x}{i:02x}{i:02x}' for i in range(50, 200, 10)]
    blue_shades = generate_dark_shades_hex("blue")
    green_shades = generate_dark_shades_hex("green")
    red_shades = generate_dark_shades_hex("red")
    font  = [random.choice(fonts)]

    def choose_text_color():
        if random.random() < black_gray_ratio:
            return random.choice(["#000000", "#282828"] + gray_shades)
        else:
            return random.choice(blue_shades + green_shades + red_shades)
    
    return {
        'size': random.choice([30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]),
        'skewing_angle': random.randint(0, 3),
        'blur': random.randint(0, 1),
        'random_skew': True,
        'random_blur': True,
        'background_type': random.randint(0, 3),
        'distorsion_type': random.randint(0, 2),
        'distorsion_orientation': random.randint(0, 2),
        'width': random.choice([300, 400, 500]),
        'alignment': random.randint(0, 2),
        'text_color': choose_text_color(),
        # 'orientation': random.randint(0, 1),
        'space_width': random.uniform(0.5, 2.0),
        'character_spacing': random.randint(0, 10),
        'fit': random.choice([True, False]),
        'output_mask': random.choice([True, False]),
        'word_split': random.choice([True, False]),
        'stroke_width': random.randint(0, 3),
        'stroke_fill': random.choice(["#282828", "red", "blue", "green", "black", "white"]),
        'image_mode': random.choice(["RGB", "L"]),
        'output_bboxes': random.randint(0, 1),
        'image_dir': './generator/images',
        'fonts': font
    }

for dir_name in OUTPUT_DIRS.values():
    os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(f'{dir_name}/labels.txt'):
        with open(f"{dir_name}/labels.txt", "w") as f:
            pass

current_indices = {key: len(os.listdir(OUTPUT_DIRS[key])) - 1 for key in OUTPUT_DIRS.keys()}

def get_output_dir():
    rand = random.randint(1, 100)
    if rand <= TRAIN_PERCENT:
        return "train"
    elif rand <= TRAIN_PERCENT + VAL_PERCENT:
        return "val"
    else:
        return "test"

for word in tqdm(all_words[:NUM_IMAGES_TO_SAVE], total=NUM_IMAGES_TO_SAVE):
    parameters = get_random_parameters()
    generator = GeneratorFromStrings(
        [word],
        size=parameters['size'],
        skewing_angle=parameters['skewing_angle'],
        random_skew=parameters['random_skew'],
        blur=parameters['blur'],
        random_blur=parameters['random_blur'],
        distorsion_type=parameters['distorsion_type'],
        distorsion_orientation=parameters['distorsion_orientation'],
        alignment=parameters['alignment'],
        text_color=parameters['text_color'],
        character_spacing=parameters['character_spacing'],
        output_mask=parameters['output_mask'],
        word_split=parameters['word_split'],
        image_mode=parameters['image_mode'],
        output_bboxes=parameters['output_bboxes'],
        image_dir=parameters['image_dir'],
        fonts=parameters['fonts']
    )
    
    for counter, (img, lbl) in enumerate(generator):
        try:
            if img is not None:
                if isinstance(img, tuple):
                    img = img[0]
                if img.size[0] > 0 and img.size[1] > 0:
                    output_dir = get_output_dir()
                    img.save(f'{OUTPUT_DIRS[output_dir]}/image{current_indices[output_dir]}.png')
                    with open(f'{OUTPUT_DIRS[output_dir]}/labels.txt', "a") as f:
                        f.write(f'image{current_indices[output_dir]}.png, {lbl}\n')
                    current_indices[output_dir] += 1
                    break
        except Exception as e:
            print(f"Error processing image: {e}")
