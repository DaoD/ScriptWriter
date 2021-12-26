# ScriptWriter: Narrative-Guided Script Generation

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- 2021-10-19: We upload the code and data for our new model ScriptWriter-CPre. 
- 2020-06-11: We find a minor error in the data thus we upload a new one. We provide the code for building data file from text file. You can now generate your own data with the Utils.py.
- 2020-06-09: We upload the code and data. Note that we do not share the vocab.txt in the data currentely due to the copyright issue. We will upload it as soon as possible.

## Abstract
This repository contains the source code and datasets for the ACL 2020 paper [ScriptWriter: Narrative-Guided Script Generation](https://www.aclweb.org/anthology/2020.acl-main.765.pdf) by Zhu et al. <br>

It is appealing to have a system that generates a story or scripts automatically from a storyline, even though this is still out of our reach. In dialogue systems, it would also be useful to drive dialogues by a dialogue plan. In this paper, we address a key problem involved in these applications - guiding a dialogue by a narrative. The proposed model ScriptWriter selects the best response among the candidates that fit the context as well as the given narrative. It keeps track of what in the narrative has been said and what is to be said. A narrative plays a different role than the context (i.e., previous utterances), which is generally used in current dialogue systems. Due to the unavailability of data for this new application, we construct a new large-scale data collection GraphMovie from a movie website where end-users can upload their narratives freely when watching a movie. Experimental results on the dataset show that our proposed approach based on narratives significantly outperforms the baselines that simply use the narrative as a kind of context.

Authors: Yutao Zhu, Ruihua Song, Zhicheng Dou, Jian-Yun Nie, Jin Zhou

## Requirements
We test the code with the following packages. <br>
- Python 3.5 <br>
- Tensorflow 1.5 (with GPU support)<br>
- Keras 2.2.4 <br>

## Usage
- Unzip the compressed [data file](https://drive.google.com/file/d/1fJKI9fzUhPM2dKq2zAFWLbtltv6PT2wh/view?usp=sharing) to the data directory. <br>
- python3 ScriptWriter.py (or python3 ScriptWriter-CPre.py)

## Results
| Model             | R2@1  | R10@1 | R10@2 | R10@5 | MRR   | P_strict | P_weak | 
| ----------------- | ----- | ----- | ----- | ----- | ----- | -------- | ------ |
| MVLSTM            | 0.651 | 0.217 | 0.384 | 0.732 | 0.395 | 0.198    | 0.224  |
| DL2R              | 0.643 | 0.210 | 0.321 | 0.638 | 0.314 | 0.230    | 0.243  |
| SMN               | 0.641 | 0.176 | 0.333 | 0.696 | 0.392 | 0.197    | 0.236  |
| DAM               | 0.631 | 0.240 | 0.398 | 0.733 | 0.408 | 0.226    | 0.236  |
| DUA               | 0.654 | 0.237 | 0.403 | 0.736 | 0.396 | 0.223    | 0.251  |
| IMN               | 0.686 | 0.301 | 0.450 | 0.759 | 0.463 | 0.304    | 0.325  |
| IOI               | 0.710 | 0.341 | 0.491 | 0.774 | 0.464 | 0.324    | 0.337  |
| MSN               | 0.724 | 0.329 | 0.511 | 0.794 | 0.464 | 0.314    | 0.346  |
| ScriptWriter      | 0.730 | 0.365 | 0.537 | 0.814 | 0.503 | 0.373    | 0.383  |
| ScriptWriter-CPre | 0.756 | 0.398 | 0.557 | 0.817 | 0.504 | 0.392    | 0.409  | 

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{zhu-etal-2020-scriptwriter,
    title = "{S}cript{W}riter: Narrative-Guided Script Generation",
    author = "Zhu, Yutao  and
      Song, Ruihua  and
      Dou, Zhicheng  and
      Nie, Jian-Yun  and
      Zhou, Jin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.765",
    pages = "8647--8657"
}
```
