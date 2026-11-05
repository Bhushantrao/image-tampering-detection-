# pyrefly: ignore[missing-import]
from PIL import Image, ImageChops, ImageEnhance
import os

def generate_ela(input_path, output_path, quality=90):

    temp_file = "temp.jpg"

    # Open image
    image = Image.open(input_path).convert('RGB')

    # Save compressed image
    image.save(
        temp_file,
        'JPEG',
        quality=quality
    )

    # Reload compressed image
    compressed_image = Image.open(temp_file)

    # Difference between original and compressed
    ela_image = ImageChops.difference(
        image,
        compressed_image
    )

    extrema = ela_image.getextrema()

    max_difference = max(
        ex[1] for ex in extrema
    )

    if max_difference == 0:
        max_difference = 1

    scale = 255.0/max_difference

    ela_image = ImageEnhance.Brightness(
        ela_image
    ).enhance(scale)

    ela_image.save(output_path)

    os.remove(temp_file)

    return output_path