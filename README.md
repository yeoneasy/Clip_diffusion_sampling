## Clip 모델을 이용한 확산 생성 모델 샘플링

 This is codebase from [openai/CLIP](https://github.com/openai/CLIP). Also, referd to [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). <br/> 
 CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. <br/><br/>
 Our goal is diffusion generative model sampling using CLIP.

### CLIP Approach
![CLIP](https://github.com/Yeoneasy/clip_guided_diffusion/assets/129255517/0a8bed9a-00db-4185-b917-8c73367a5c54)
 CLIP은 이미지 인코더와 텍스트 인코더를 결합하여 이미지와 텍스트를 모두 처리할 수 있는 모델<br/>
 Contrastive learning은 각 이미지와 텍스트를 결합한 가상의 테이블에서 실제 일치하는 쌍에서<br/> 
 긍정적인 영향이 그렇지 않은 부분에서 부정적인 영향이 극대화되어 학습이 이뤄지도록 하는 학습 기법<br/>
 이를 통해 CLIP에서 image에 맞는 class 정보를 가진 텍스트를 생성

### Diffusion Approach 
![aas](https://github.com/Yeoneasy/clip_diffusion_sampling/assets/129255517/e32673e0-7a9a-4993-a6ba-2c0be38dbff5)
 Diffusion model은 기존의 데이터에 noise를 더해가며 학습하는 모델<br/>
 forward process와 reverse process를 이용하여 이미지를 생성하며 guidance를 활용<br/>
 이를 통해 사용자가 원하는 이미지에 가깝게 생성하는 방법
### Requirements

```
pip install -r requirements.txt
```

### Files

 **cars_text** is the text per image folder in the shapenet dataset.
 The number of these is 2151. <br/> The image folders each contain 250 photos of the subject from different angles. <br/> 
 We only use images of cars in shapenet.

### Usage

 Train and validation

 1. Download code zip
 2. Download shapenet dataset [here](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90) (cars_train, cars_val)
 3. Put datasets seperately in directory (/db/cars_train, /db/cars_val)
 4. Run unnoised_train or ipynb file in notebooks



