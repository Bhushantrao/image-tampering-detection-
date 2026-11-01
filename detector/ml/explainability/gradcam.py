import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ..models.vit_model import ViTModel

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = ViTModel()

model.load_state_dict(
    torch.load(
        'saved_models/best_vit_model.pth',
        map_location=device
    )
)

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# ViT reshape function
def reshape_transform(tensor):

    # remove class token
    result = tensor[:,1:,:]

    height = width = 14

    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )

    result = result.transpose(2,3)

    return result


def generate_heatmap(
    image_path,
    output_path
):

    image = Image.open(
        image_path
    ).convert('RGB')

    # Resize to match model input
    image = image.resize(
        (224,224)
    )

    image_np = np.array(
        image
    )

    image_float = image_np.astype(
        np.float32
    ) / 255

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    target_layers = [
        model.model.blocks[-1].norm1
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        image_float,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path