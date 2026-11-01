from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance

import os


def create_ela_image(
    image_path,
    quality=90
):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_file = os.path.join(BASE_DIR, "temp.jpg")

    image = Image.open(
        image_path
    ).convert("RGB")

    image.save(
        temp_file,
        "JPEG",
        quality=quality
    )

    compressed = Image.open(
        temp_file
    )

    ela_image = ImageChops.difference(
        image,
        compressed
    )

    extrema = ela_image.getextrema()

    max_diff = max(
        ex[1]
        for ex in extrema
    )

    if max_diff == 0:
        max_diff = 1

    scale = 255 / max_diff

    ela_image = ImageEnhance.Brightness(
        ela_image
    ).enhance(scale)

    ela_dir = os.path.join(BASE_DIR, "media", "ela")
    os.makedirs(
        ela_dir,
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    ela_path = os.path.join(
        ela_dir,
        filename
    )

    ela_image.save(
        ela_path
    )

    ela_image.load()
    image.close()
    compressed.close()

    try:
        os.remove(
            temp_file
        )
    except Exception:
        pass

    return ela_path