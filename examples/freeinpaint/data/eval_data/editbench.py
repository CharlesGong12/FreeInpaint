from pathlib import Path

import pandas as pd
from PIL import Image

    
class EditBench:
    def __init__(self, root_folder, verbose=False, use_prompt_mask=False):
        self.root_folder = Path(root_folder)
        print(f'loading editbench from {self.root_folder}')
        self.verbose = verbose
        self.use_prompt_mask = use_prompt_mask

        natural_captions_path = self.root_folder / 'annotations_natural.csv'
        generated_captions_path = self.root_folder / 'annotations_generated.csv'

        self.natural_captions_df = pd.read_csv(natural_captions_path, header=0)
        self.generated_captions_df = pd.read_csv(generated_captions_path, header=0)

        # row number to aos (concat natural and generated df)
        self.id_to_aos = {}
        for i, row in self.natural_captions_df.iterrows():
            self.id_to_aos[i] = row['aos']
        for i, row in self.generated_captions_df.iterrows():
            self.id_to_aos[i + len(self.natural_captions_df)] = row['aos']

        self.image_ids = sorted(list(self.id_to_aos.keys()))
        if verbose:
            print(f'Number of images: {len(self.image_ids)}')

    def __getitem__(self, item):
        image_id = item
        if item < len(self.natural_captions_df):
            prompt_full = self.natural_captions_df.iloc[item]['prompt_full']
            # TODO: if prompt_full is used to generate, prompt_mask-simple should be used for local clip score
            # if prompt_mask-rich is used to generate, prompt_mask-rich should be used for local clip score
            if not self.use_prompt_mask:
                prompt_mask = self.natural_captions_df.iloc[item]['prompt_mask-simple']
            else:
                prompt_mask = self.natural_captions_df.iloc[item]['prompt_mask-rich']
            image_path = self.root_folder / 'references_natural' / f'{self.id_to_aos[image_id]}.png'
            image = Image.open(image_path).convert('RGB')
            mask_path = self.root_folder / 'masks_natural' / f'{self.id_to_aos[image_id]}.png'
            mask = Image.open(mask_path).convert('RGB')
            image_name = f'{image_id}_natural_{self.id_to_aos[image_id]}'
        else:
            prompt_full = self.generated_captions_df.iloc[item - len(self.natural_captions_df)]['prompt_full']
            if not self.use_prompt_mask:
                prompt_mask = self.generated_captions_df.iloc[item - len(self.natural_captions_df)]['prompt_mask-simple']
            else:
                prompt_mask = self.generated_captions_df.iloc[item - len(self.natural_captions_df)]['prompt_mask-rich']
            image_path = self.root_folder / 'references_generated' / f'{self.id_to_aos[image_id]}.png'
            image = Image.open(image_path).convert('RGB')
            mask_path = self.root_folder / 'masks_generated' / f'{self.id_to_aos[image_id]}.png'
            mask = Image.open(mask_path).convert('RGB')
            image_name = f'{image_id}_generated_{self.id_to_aos[image_id]}'
    
        return image_name, image, mask, prompt_full, prompt_mask

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    ds = EditBench(
        root_folder = '',
        verbose=True
    )
    image_idx = 78
    image_name, image, mask, prompt_full, prompt_mask = ds[image_idx]
    print('Image name:', image_name)
    print('Image size:', image.size)
    print('Mask size:', mask.size)
    print('Prompt full:', prompt_full)
    print('Prompt mask:', prompt_mask)
    image.save('')
    mask.save('')