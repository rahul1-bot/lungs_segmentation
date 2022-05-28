from __future__ import annotations
import os, torch, torchvision
import numpy as np
import matplotlib.pyplot as plt



class Visualize:
    def __init__(self, sample: dict[str, Any]) -> None:
        self.sample = sample
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
    
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({'sample': self.sample})    
    
    
    
    def image_with_ground(self, idx: int, left: Optional[bool] = False, right: Optional[bool] = False, save: Optional[bool] = False) -> matplotlib.pyplot:
        background: np.ndarray = np.asarray(self.sample[idx]['image'])
        plt.imshow(background[0].astype('float'),cmap='gray')
        filter_: np.ndarray = np.asarray(self.sample[idx]['label'],dtype='uint8')
        
        if left:
            leftfilter = filter_ == 1
            leftfilter.astype('uint8')
            leftfilter = leftfilter * 255
            zerolayer = np.zeros((leftfilter.shape[0], leftfilter.shape[1]))
            leftforeground = np.stack((zerolayer,zerolayer, leftfilter), axis = -1)
            plt.imshow(leftforeground.astype('uint8'), alpha = 0.3)
        
        if right:
            rightfilter = filter_ == 2
            rightfilter.astype('uint8')
            rightfilter = rightfilter * 255
            zerolayer = np.zeros((rightfilter.shape[0], rightfilter.shape[1]))
            rightforeground = np.stack((rightfilter,zerolayer,zerolayer), axis = -1)
            plt.imshow(rightforeground.astype('uint8'), alpha = 0.3)
        
        if save:
            plt.savefig('./image/' + str(idx) + '_groud')
        else:
            plt.show()
    
    
    
    def image_with_mask(self, idx: int, filename: str, mask: str, left: Optional[bool] = False, right: Optional[bool] = False, save: Optional[bool] = False) -> None:
        background: np.ndarray = np.asarray(self.sample[idx]['image'])
        plt.imshow(background[0].astype('float'), cmap = 'gray')
        filter_: np.ndarray = np.asarray(np.argmax(mask, axis = 0))
        
        if left:
            leftfilter = filter_ == 1
            leftfilter.astype('uint8')
            leftfilter = leftfilter * 255
            zerolayer = np.zeros((leftfilter.shape[0], leftfilter.shape[1]))
            leftforeground = np.stack((zerolayer, zerolayer, leftfilter), axis = -1)
            plt.imshow(leftforeground.astype('uint8'), alpha = 0.3)
        
        if right:
            rightfilter = filter_ == 2
            rightfilter.astype('uint8')
            rightfilter = rightfilter*255
            zerolayer = np.zeros((rightfilter.shape[0],rightfilter.shape[1]))
            rightforeground = np.stack((rightfilter,zerolayer,zerolayer), axis = -1)
            plt.imshow(rightforeground.astype('uint8'),alpha = 0.3)
        
        if save:
            plt.savefig('./image/' +  os.path.splitext(os.path.basename(filename))[0] + '_mask.png')
        else:
            plt.show()
    
    
    
    def save_for_preprocessing(self, idx: int, filename: str, mask: str, out_dir: str) -> None:
        background: np.ndarray = np.asarray(self.sample[idx]['image'])
        filter_: np.ndarray = np.asarray(np.argmax(mask, axis=0))
        filter_ = (filter_ > 0).astype('uint8')
        filter_ = np.stack((filter_, filter_, filter_))
        filtered: torch.tensor = torch.Tensor(background * filter_)
        save_image(filtered, os.path.join(out_dir, "{}.png".format(os.path.splitext(os.path.basename(filename))[0])))
    




#@: Driver code
if __name__.__contains__('__main__'):
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)