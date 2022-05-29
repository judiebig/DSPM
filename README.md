# DSPM

![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/pytorch-1.6-green)
![](https://img.shields.io/badge/cudatoolkit-10.1-green)
![](https://img.shields.io/badge/cuda-11.0-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)
![](https://img.shields.io/badge/pystoi-0.3.3-green)
![](https://img.shields.io/badge/pypesq-1.2.4-green)

This repo provides a reference implementation of **DSPM** as described in the paper:

> Improving Monaural Speech Enhancement with Dynamic Scene Perception Module

> Accpted by ICME, 2022. 

## data preprosessing

105 types of noise are concatenated for training and validation, while 110 types of noise for testing (add 5 unseen noises).  The ratio of the invisible part to the visible part of our test noise is about 4:1 (5 unseen noises vs. 105 seen noises, 20 minutes vs. 5 minutes)

The  experimental platform  is  Ubuntu  LTS  18.04  with  i7-9700  and  RTX  2060.


## Contact

For any questions please open an issue or drop an email to: `wxtai@std.uestc.edu.cn`
