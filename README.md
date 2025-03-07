## Clip 모델을 이용한 확산 생성 모델 샘플링

 프로젝트의 목표는 CLIP 모델을 활용하여 확산 모델을 통해 보다 정교하고 의미 있는 이미지를 생성하는 것 입니다. <br/>
 기존의 확산 모델은 주어진 노이즈를 점진적으로 제거하며 이미지를 생성하지만, 원하는 이미지의 특징을 직접적으로 <br/>
 조정하는 것은 어렵습니다. 따라서 CLIP 모델에 guidance를 도입하여 이미지 생성과정에서 사용자가 원하는 특징을 <br/>
 더욱 효과적으로 반영할 수 있도록 합니다. 이를 통해 단순한 확산 모델 기반의 이미지 생성이 아니라, 특정한 방향성과<br/>
 의미를 가진 이미지 생성이 가능해지며, 생성된 이미지를 클래스에 따라 분류하여 정확도를 향상시키고자 합니다.
    
### CLIP Approach
![CLIP](https://github.com/Yeoneasy/clip_guided_diffusion/assets/129255517/0a8bed9a-00db-4185-b917-8c73367a5c54)
 CLIP은 이미지 인코더와 텍스트 인코더를 결합하여 이미지와 텍스트를 모두 처리할 수 있는 모델 입니다. <br/>
 Contrastive learning은 각 이미지와 텍스트를 결합한 가상의 테이블에서 실제 일치하는 쌍에서<br/> 
 긍정적인 영향이 그렇지 않은 부분에서 부정적인 영향이 극대화되어 학습이 이뤄지도록 하는 학습 기법으로<br/>
 이를 통해 CLIP에서 image에 맞는 class 정보를 가진 텍스트를 생성합니다.

### Diffusion Approach 
![aas](https://github.com/Yeoneasy/clip_diffusion_sampling/assets/129255517/e32673e0-7a9a-4993-a6ba-2c0be38dbff5)<br/>
 Diffusion model은 기존의 데이터에 noise를 더해가며 학습하는 모델 입니다.<br/>
 forward process와 reverse process를 이용하여 이미지를 생성하며 guidance를 활용하여<br/>
 사용자가 원하는 이미지에 가깝게 생성하는 방법 입니다.

### 수행 과정
 1. 데이터셋(Shapenet v2) 중 자동차(car)에 관련된 이미지와 텍스트를 1:1로 라벨링
 2. 이미지 데이터셋과 라벨 값을 가지고 CLIP 모델을 사용하여 학습하고 평가, 검증
 3. 이미지 데이터셋을 통해 diffusion model을 학습하고 학습한 CLIP 모델을 guidance로 하여 샘플링
 4. 추가적으로 diffusion model의 각 noise들에 대해서도 CLIP을 학습시키고, 이를 guidance로 사용하여 diffusion model 샘플링

### 라벨링

 **cars_text**는 shapenet 데이터셋의 이미지 폴더에 1:1로 매칭된 텍스트 목록 입니다. <br/>
 이는 2151개 이며, 각 이미지 폴더(ex. red car)에는 다양한 각도에서 찍은 자동차의 이미지 250장이 들어있습니다.<br/>
 그러므로 총 자동차 이미지의 개수는 대략 537750(2151x250)개 입니다.
 
### Requirements

```
pip install -r requirements.txt
```

### References

- [openai/CLIP](https://github.com/openai/CLIP) - OpenAI's CLIP: Learning Transferable Visual Models from Natural Language Supervision.
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - PyTorch implementation of denoising diffusion probabilistic models.
- Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). *arXiv preprint, arXiv:2103.00020*.
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *arXiv preprint, arXiv:2006.11239*.
- Nichol, A., et al. (2022). [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741). *arXiv preprint, arXiv:2112.10741*.
- Sitzmann, V., Zollhöfer, M., & Wetzstein, G. (2020). [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://arxiv.org/abs/1906.01618). *arXiv preprint, arXiv:1906.01618*.


