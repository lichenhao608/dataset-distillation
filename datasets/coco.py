from typing import Any, Callable, Optional, Tuple

from torchvision import datasets


class CustomCOCO(datasets.CocoDetection):

    def __init__(self,
                 root: str,
                 annFile: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        imgid = self.ids[index]
        image = self._load_image(imgid)
        annotation = self._load_target(imgid)
        if self.transform is not None:
            image = self.transform(image)

        return image, len(annotation)
