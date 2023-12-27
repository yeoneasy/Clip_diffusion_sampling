# clip_diffusion_sampling

This is codebase from [openai/CLIP](https://github.com/openai/CLIP). <br/> 
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. <br/>
Object is Diffusion generative model sampling using CLIP.

## Approach
![CLIP](https://github.com/Yeoneasy/clip_guided_diffusion/assets/129255517/0a8bed9a-00db-4185-b917-8c73367a5c54)

### Requirements

```
pip install -r requirements.txt
```

### Files

**cars_text** is the text per image folder in the shapenet dataset.

### Usage

Train and validation

1. Download code zip
2. Download shapenet dataset [here](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90) (cars_train, cars_val)
3. Put datasets seperately in directory (/db/cars_train, /db/cars_val)



