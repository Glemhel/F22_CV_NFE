from PIL import Image
from torch.utils.data import Dataset
import glob


class CustomDataset(Dataset):
    """
    class for image dataset
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.files = glob.glob(root_dir+"/*")
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image