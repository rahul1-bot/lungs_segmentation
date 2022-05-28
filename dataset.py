from __future__ import annotations
import numpy as np
import pandas as pd
from PIL import Image
import os, pydicom, torch, torchvision  



class LungsSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str, leftmask_path: str, 
                                        rightmask_path: str, 
                                        image_transform: Optional[torchvision.transforms.Compose] = None, 
                                        label_transform: Optional[torchvision.transforms.Compose] = None, 
                                        convert_to: Optional[str] = 'L') -> None:
        self.image_path = image_path
        self.leftmask_path = leftmask_path
        self.rightmask_path = rightmask_path
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.convert_to = convert_to
        if self.convert_to not in ['RGB', 'L']:
            return ValueError(f'{self.convert_to} must be either RGB or L')
        
        self.file_names: list[str] = [
            file_name for root_dirs, dirs, files in os.walk(self.image_path) for file_name in files
        ] 
        
    
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
    
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({
            'image_path': self.image_path, 'leftmask_path': self.leftmask_path,
            'rightmask_path': self.rightmask_path, 'image_transform': self.image_transform,
            'label_transform': self.label_transform, 'convert_to': self.convert_to, 'file_names': self.file_names
        })
    
    
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        image_name: str = os.path.join(self.image_path, self.file_names[index])
        leftmask_name: str = os.path.join(self.leftmask_path, self.file_names[index])
        rightmask_name: str = os.path.join(self.rightmask_path, self.file_names[index])
        
        img: PIL.Image = Image.open(image_name).convert(self.convert_to)
        leftmask: PIL.Image = Image.open(leftmask_name)
        rightmask: PIL.Image = Image.open(rightmask_name)
        
        if self.image_transform is not None:
            img: torch.tesnor = self.image_transform(img)
        if self.label_transform is not None:
            leftmask: torch.tesnor = self.label_transform(leftmask)
            rightmask: torch.tesnor = self.label_transform(rightmask)
        
        rightmask: torch.tesnor = rightmask * 2
        rightmask, leftmask = rightmask[0].type(torch.uint8), leftmask[0].type(torch.uint8)
        label: int = rightmask + leftmask
        
        return {
            'image': img, 'label': label, 'filename': self.file_names[index]
        }
        
        




class LungsTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str, image_transform: torchvision.transforms.Compose, convert_to: str) -> None:
        self.image_path = image_path
        self.image_transform = image_transform
        self.convert_to = convert_to
        if self.convert_to not in ['RGB', 'L']:
            return ValueError(f'{self.convert_to} must be either RGB or L')
        
        self.file_names: list[str] = [
            file_name for root_dirs, dirs, files in os.walk(self.image_path) for file_name in files
        ]
    
    
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })    
    
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({
            'image_path': self.image_path, 'image_transform': self.image_transform,
            'convert_to': self.convert_to, 'file_names': self.file_names
        })
    
    
    
    def __getitem__(self, index: str) -> dict[str, Any]:
        image_name: str = os.path.join(self.image_path, self.file_names[index])
        img: PIL.Image = Image.open(image_name).convert(self.convert_to)
        if self.image_transform is not None:
            img: torch.tensor = self.image_transform(img)
        
        return {
            'image': img, 'filename': self.file_names[index]
        }





            
class DicomSegment(torch.utils.data.Dataset):
    def __init__(self, image_path: str, image_transform: torchvision.transform.Compose, convert_to: str) -> None:
        self.image_path = image_path
        self.image_transform = image_transform
        self.convert_to = convert_to
        if self.convert_to not in ['RGB', 'L']:
            return ValueError(f'{self.convert_to} must be either RGB or L')
        
        self.file_names: list[str] = [
            file_name for root_dirs, dirs, files in os.walk(self.image_path) for file_name in files
        ]        
    
    
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })    
    
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({
            'image_path': self.image_path, 'image_transform': self.image_transform,
            'convert_to': self.convert_to, 'file_names': self.file_names
        })
    
    
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        image_name: str = os.path.join(self.image_path, self.file_names[index])
        vector: np.ndarray = pydicom.read_file(image_name).pixel_array
        img: PIL.Image = Image.fromarray(vector).convert(self.convert_to)
        if self.image_transform is not None:
            img: torch.tensor = self.image_transform(img)
        
        return {
            'image': img, 'filename': self.file_names[index]
        }






class JSRTDataset(torch.utils.data.Dataset):
    def __init__(self, original_img_path: str, base_img_path: str, 
                                               image_transform: torchvision.transform.Compose, 
                                               convert_to: str, 
                                               test: Optional[bool] = False) -> None:
        self.original_img_path = original_img_path
        self.base_img_path = base_img_path
        self.image_transform = image_transform
        self.convert_to = convert_to
        if self.convert_to not in ['RGB', 'L']:
            return ValueError(f'{self.convert_to} must be either RGB or L')
        
        self.file_names: list[str] = [
            file_name for root_dirs, dirs, files in os.walk(self.original_img_path) for file_name in files
        ]
        self.test = test
    
    
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
    
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({
            'original_img_path': self.original_img_path, 'base_img_path': self.base_img_path,
            'image_transform': self.image_transform, 'convert_to': self.convert_to,
            'file_names': self.file_names, 'test': self.test
        })
    
    
    
    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor] | torch.tensor:
        original_image_name: str = os.path.join(self.original_img_path, self.file_names[index])
        vector: np.ndarray = np.fromfile(original_image_name, dtype= '>i2').reshape((2048, 2048))
        original_image: PIL.Image = Image.fromarray((vector/ vector.max()) * 255).convert(self.convert_to)
        
        if not self.test:
            base_image_name: str = os.path.join(self.base_img_path, self.file_names[index].replace('.IMG', '.png'))
            bse: np.ndarray = np.array(Image.open(base_image_name))
            bse_img: np.ndarray = Image.fromarray(((bse/bse.max()) * 255).astype('uint8')).convert(self.convert_to)
            original_image: torch.tensor = self.image_transform(original_image)
            bse_img: torch.tensor = self.image_transform(bse_img)
            return original_image, bse_img
        
        return self.image_transform(original_image)
        
         




#@: Driver Code
if __name__.__contains__('__main__'):
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)