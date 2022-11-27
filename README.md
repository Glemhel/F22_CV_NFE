# :robot: Neural Face Editor (NFE) :art:

Neural Face Editor - create and edit anime images with power of GAN's and SoTA methods!

Team:
- Mikhail Rudakov, B19-DS, m.rudakov@innopolis.university
- Anna Startseva, B19-DS, a.startseva@innopolis.university
- Andrey Palaev, B19-DS, a.palaev@innopolis.university

## Table of Contents
- [Description](#description)
- [Telegram Bot Demo](#telegram-bot-demo)
- [Project Architecture](#project-architecture)
- [Results](#results)
- [References](#references)

## Description
Neural Face Editor is an application that utilizes GAN latent space manipulation to change anime faces facial attributes. Application is available in form of telegram bot via [@neural_face_editor_bot](https://t.me/neural_face_editor_bot).
To achieve such result, StyleGAN [[1]](#1) is trained on [animefacesdataset](https://www.gwern.net/Danbooru2021) [[2]](#2) for anime faces generation. Then, several SVM are trained for separate attributes. Finally, ideas from [[3]](#3) are used to perform unconditional and conditional attributes manipulation. To let user use their own pictures, hybrid GAN inversion [[4]](#4) is implemented. To assess GAN image generation, FID [[5]](#5) distance is calculated on the dataset of real and fake images. Output quality of generated images has been assessed manually. More detailes are provided in project report.

## Telegram Bot Demo
https://user-images.githubusercontent.com/66643655/204153747-7515023c-4b79-4aa3-b4e6-72e9ce785685.mp4

https://user-images.githubusercontent.com/66643655/204155710-8620c5ee-a8bb-4c03-bfc4-f6f1837d9db0.mp4

## Project Architecture
![Step 1 and 2](images/image_generation_annotation.jpg)
![Step 3](images/svm.jpg)
Mainly adopted from Shen et al. [[3]](#3), our project architecture consists of several models.

Stage 1 is image generation. StyleGAN is used to produce images of anime faces.

Stage 2 is generated image annotation. We use [illustration2Vec](https://github.com/rezoo/illustration2vec) proposed by Saito and Matsui [[6]](#6) to tag images with tags we need.

Stage 3 is image classification. For the each attribute we want to be able to control further, we train a SVM for separating latent codes based on some feature. This would let us know the vector in which this feature changes in the latent space.

## Results
Sample generated images, produced by GAN: <br/>
![samples](images/samples.png)

### Image Manipulation
After getting separating hyperplanes, we implemented the way of attribute manipulation from [[3]](#3). Specifically, given the separating hyperplane n and the latent vector z, we generate several images using $z+an$, where `a = np.linspace(-3, 3, 6)`. Examples of image manipulation:
1. Making hair more (or less) green while keeping the hair length
![change_color_green](images/change_color_green.png)
2. Making hair more (or less) red without controlling the hair length
![change_color](images/change_color.png)
3. Changing the hair length preserving the hair colour
![change_length](images/change_length.png)
4. Changing the hair length without controlling the hair colour
![change_color_length](images/change_color_length.png)

### GAN Inversion
Some results to access quality of GAN inversion:

<img align="left" width="138" height="138" src=images/gi11.png>
<img align="center" width="138" height="138" src=images/gi12.png>

<img align="left" width="138" height="138" src=images/gi21.png>
<img align="center" width="138" height="138" src=images/gi22.png>

## References
<a id="1">[1]</a>
T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019.

<a id="2">[2]</a>
Anonymous, D. community, and G. Branwen, “Danbooru2021: A large-scale crowdsourced and tagged anime illustration dataset,” https://www.gwern.net/Danbooru2021, January 2022. [Online]. Available: https://www.gwern.net/Danbooru2021

<a id="3">[3]</a>
Y. Shen, J. Gu, X. Tang, and B. Zhou, “Interpreting the latent space of gans for semantic face editing,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 9240–9249

<a id="4">[4]</a>
J. Zhu, Y. Shen, D. Zhao, and B. Zhou, “In-domain gan inversion for real image editing,” in ECCV, 2020.

<a id="5">[5]</a>
M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.

<a id="6">[6]</a> 
Saito, Masaki, and Yusuke Matsui. "Illustration2vec: a semantic vector representation of illustrations." SIGGRAPH Asia 2015 Technical Briefs. 2015. 1-4.
