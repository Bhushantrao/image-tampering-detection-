import os
import random
import shutil

source_root = "new/image"

mapping = {
    "au":"real",
    "tp":"fake"
}

target = "dataset"

random.seed(42)

for source_folder, target_class in mapping.items():

    source = os.path.join(
        source_root,
        source_folder
    )

    files = [

        f for f in os.listdir(source)

        if f.lower().endswith(
            (".jpg",".jpeg",".png")
        )
    ]

    random.shuffle(files)

    total = len(files)

    train_end = int(total*0.7)
    val_end = int(total*0.85)

    splits = {

        "train": files[:train_end],

        "validation":
        files[train_end:val_end],

        "test":
        files[val_end:]
    }


    for split_name,file_list in splits.items():

        destination = os.path.join(
            target,
            split_name,
            target_class
        )

        os.makedirs(
            destination,
            exist_ok=True
        )

        for file in file_list:

            shutil.copy(
                os.path.join(
                    source,
                    file
                ),
                os.path.join(
                    destination,
                    file
                )
            )

print("Dataset split complete")