import json
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os

class COCOCustom:
    def __init__(self, jsonl_path, image_root, coco_caption_path=None, verbose=False, use_bbox=True):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.coco_caption_path = coco_caption_path if coco_caption_path else os.path.join(image_root, 'annotations', 'captions_val2017.json')
        self.data = []

        # Read the JSONL file
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        if verbose:
            print(f"Loaded {len(self.data)} entries from {self.jsonl_path}")
        self.use_bbox = use_bbox

        # Load COCO Captions
        self.coco = COCO(self.coco_caption_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        
        # Get image name and path
        image_name = f"{item}_{entry['file_name']}"
        image_name = image_name.replace('.jpg', '')
        file_name = entry['file_name']
        image_path = self.image_root / Path('val2017') / file_name
        image = Image.open(image_path).convert('RGB')

        # Generate mask
        mask = Image.new('RGB', (entry['width'], entry['height']), (0, 0, 0))
        if not self.use_bbox:
            draw = ImageDraw.Draw(mask)
            for annotation in entry['annotation']:
                if 'segmentation' in annotation:
                    for polygon in annotation['segmentation']:
                        draw.polygon(polygon, fill=(255, 255, 255))

        else:
            draw = ImageDraw.Draw(mask)
            for annotation in entry['annotation']:
                if 'bbox' in annotation:
                    bbox = annotation['bbox']
                    draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], fill=(255, 255, 255))

        # Get prompt_full and prompt_mask
        # prompt_full = entry['caption'][0]
        image_id = entry['image_id']
        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)
        prompt_full = [ann['caption'] for ann in anns][0]
        prompt_mask = entry['caption'][0]

        return image_name, image, mask, prompt_full, prompt_mask

# Example usage
if __name__ == "__main__":
    jsonl_path = "examples/freeinpaint/anno/llava-next_annotations_val2017.jsonl"
    image_root = "MSCOCO"
    coco_caption_path = os.path.join(image_root, 'annotations', 'captions_val2017.json')

    dataset = COCOCustom(jsonl_path, image_root, coco_caption_path, verbose=True, use_bbox=False)
    for i in range(len(dataset)):
        image_name, image, mask, prompt_full, prompt_mask = dataset[i]
        print(f"Image Name: {image_name}")
        print(f"Prompt Full: {prompt_full}")
        print(f"Image Size: {image.size}, Mask Size: {mask.size}")

        # Optionally visualize
        image.save('')
        mask.save('')

        if i == 1:  # Limit to 5 examples for debugging
            break
