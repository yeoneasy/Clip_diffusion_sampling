## Clip 모델을 이용한 확산 생성 모델 샘플링

 이 프로젝트의 목표는 CLIP 모델을 활용하여 확산 모델을 통해 보다 정교하고 의미 있는 이미지를 생성하는 것이다. <br/>
 기존의 확산 모델은 주어진 노이즈를 점진적으로 제거하며 이미지를 생성하지만, 원하는 이미지의 특징을 직접적으로 <br/>
 조정하는 것은 어렵다. 따라서 우리는 CLIP 모델의 guidance를 도입하여 이미지 생성과정에서 사용자가 원하는 특징을 <br/>
 더욱 효과적으로 반영할 수 있도록 한다. 이를 통해 단순한 확산 모델 기반의 이미지 생성이 아니라, 특정한 방향성과 <br/>
 의미를 가진 이미지 생성이 가능해지며, 생성된 이미지를 클래스에 따라 분류하여 정확도를 향상시키고자 한다.
 
### 목표
 1. Shapenet v2 데이터셋 중 car에 관련된 이미지들에 맞게 text를 라벨링
 2. 이미지 데이터셋과 라벨 값을 가지고 CLIP을 학습하고 평가
 3. 이미지 데이터셋을 통해 diffusion model을 학습하고 학습한 CLIP을 guidance로 하여 샘플링
 4. 추가적으로 diffusion model의 각 noise들에 대해서도 CLIP을 학습시키고, 이를 guidance로 사용하여 diffusion model 샘플링
    
### CLIP Approach
![CLIP](https://github.com/Yeoneasy/clip_guided_diffusion/assets/129255517/0a8bed9a-00db-4185-b917-8c73367a5c54)
 CLIP은 이미지 인코더와 텍스트 인코더를 결합하여 이미지와 텍스트를 모두 처리할 수 있는 모델<br/>
 Contrastive learning은 각 이미지와 텍스트를 결합한 가상의 테이블에서 실제 일치하는 쌍에서<br/> 
 긍정적인 영향이 그렇지 않은 부분에서 부정적인 영향이 극대화되어 학습이 이뤄지도록 하는 학습 기법<br/>
 이를 통해 CLIP에서 image에 맞는 class 정보를 가진 텍스트를 생성

### Diffusion Approach 
![aas](https://github.com/Yeoneasy/clip_diffusion_sampling/assets/129255517/e32673e0-7a9a-4993-a6ba-2c0be38dbff5)<br/>
 Diffusion model은 기존의 데이터에 noise를 더해가며 학습하는 모델<br/>
 forward process와 reverse process를 이용하여 이미지를 생성하며 guidance를 활용<br/>
 이를 통해 사용자가 원하는 이미지에 가깝게 생성하는 방법

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
    
### Requirements

```
pip install -r requirements.txt
```

### References
  1. [openai/CLIP](https://github.com/openai/CLIP) <br/>
  2. [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

