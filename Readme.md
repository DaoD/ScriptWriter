# ScriptWriter: Narrative-Guided Script Generation

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- 2020-06-11: We find a minor error in the data thus we upload a new one. We provide the code for building data file from text file. You can now generate your own data with the Utils.py.
- 2020-06-09: We upload the code and data. Note that we do not share the vocab.txt in the data currentely due to the copyright issue. We will upload it as soon as possible.

## Abstract
This repository contains the source code and datasets for the ACL 2020 paper [ScriptWriter: Narrative-Guided Script Generation](https://www.aclweb.org/anthology/2020.acl-main.765.pdf) by Zhu et al. <br>

It is appealing to have a system that generates a story or scripts automatically from a storyline, even though this is still out of our reach. In dialogue systems, it would also be useful to drive dialogues by a dialogue plan. In this paper, we address a key problem involved in these applications - guiding a dialogue by a narrative. The proposed model ScriptWriter selects the best response among the candidates that fit the context as well as the given narrative. It keeps track of what in the narrative has been said and what is to be said. A narrative plays a different role than the context (i.e., previous utterances), which is generally used in current dialogue systems. Due to the unavailability of data for this new application, we construct a new large-scale data collection GraphMovie from a movie website where end-users can upload their narratives freely when watching a movie. Experimental results on the dataset show that our proposed approach based on narratives significantly outperforms the baselines that simply use the narrative as a kind of context.

Authors: Yutao Zhu, Ruihua Song, Zhicheng Dou, Jian-Yun Nie, Jin Zhou

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.5 <br>
- Tensorflow 1.5 (with GPU support)<br>
- Keras 2.2.4 <br>

## Usage
- Unzip the compressed [data file](https://drive.google.com/file/d/1X8qjwAvyc85smlbRHvsWOiLgoeIG8JlN/view?usp=sharing) to the data directory. <br>
- Python3 ScriptWriter.py

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{zhu-etal-2020-script,
    title = "ScriptWriter: Narrative-Guided Script Generation",
    author = "Zhu, Yutao  and
      Song, Ruihua  and
      Dou, Zhicheng  and
      Nie, Jian-Yun  and
      Zhou, Jin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```
