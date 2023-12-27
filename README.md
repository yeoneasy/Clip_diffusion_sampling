# clip_diffusion_sampling

This is codebase from [openai/CLIP](https://github.com/openai/CLIP), [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). <br/> 
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. <br/><br/>
Our goal is diffusion generative model sampling using CLIP.

## Approach
![CLIP](https://github.com/Yeoneasy/clip_guided_diffusion/assets/129255517/0a8bed9a-00db-4185-b917-8c73367a5c54)

![aas](https://github.com/Yeoneasy/clip_diffusion_sampling/assets/129255517/e32673e0-7a9a-4993-a6ba-2c0be38dbff5)

### Requirements

```
pip install -r requirements.txt
```

### Files

**cars_text** is the text per image folder in the shapenet dataset.
The number of these is 2151. <br/> The image folders each contain 250 photos of the subject from different angles.
We only use images of cars in shapenet.

### Usage

Train and validation

1. Download code zip
2. Download shapenet dataset [here](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90) (cars_train, cars_val)
3. Put datasets seperately in directory (/db/cars_train, /db/cars_val)



