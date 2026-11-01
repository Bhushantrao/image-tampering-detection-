import os
import shutil

base_folder = "new"

image_folder = os.path.join(
    base_folder,
    "image"
)

target = "raw_dataset"

mapping = {
    "au":"real",
    "tp":"fake"
}


for source_folder,target_folder in mapping.items():

    src = os.path.join(
        image_folder,
        source_folder
    )

    if not os.path.exists(src):

        print(
            f"Folder not found: {src}"
        )

        continue


    dst = os.path.join(
        target,
        target_folder
    )

    os.makedirs(
        dst,
        exist_ok=True
    )


    for file in os.listdir(src):

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):

            shutil.copy(
                os.path.join(
                    src,
                    file
                ),

                os.path.join(
                    dst,
                    file
                )
            )


print(
    "Dataset preparation completed"
)