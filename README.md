# MOCA
<a href=""> <b> MOCA: A Modular Object-Centric Approach for Interactive Instruction Following </b> </a>
<br>
<a href=""> Kunal Pratap Singh* </a>,
<a href=""> Suvaansh Bhambri* </a>,
<a href=""> Byeonghwi Kim* </a>,
<a href="http://roozbehm.info/"> Roozbeh Mottaghi </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>

<b> MOCA </b> (<b>M</b>odular <b>O</b>bject-<b>C</b>entric <b>A</b>pproach) is a modular architecture that decouples a task into visual perception and action policy.


<img src="media/moca.png" alt="MOCA">

## Environment
### Clone repo
```
$ git clone https://github.com/gistvision/moca.git moca
$ export ALFRED_ROOT=$(pwd)/moca
```

### Install requirements
```
$ virtualenv -p $(which python3) --system-site-packages moca_env
$ source moca_env/bin/activate

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Dataset
Note that this includes expert trajectories with both original and color-swapped frames.
For details, please refer to the repository of <a href="https://github.com/askforalfred/alfred">ALFRED</a>.
```bash
$ cd $ALFRED_ROOT/data
$ sh download_data.sh
$ ls
download_data.sh  __init__.py  json_2.1.0 json_feat_2.1.0 preprocess.py  README.md  splits
```

## Training
### Cmd
```
python models/train/train_seq2seq.py --dout exp/ --gpu --save_every_epoch
```

### Example
If you want train MOCA and save the weights for all epochs in "exp/moca", you may use the command below.
```
python models/train/train_seq2seq.py --dout exp/moca --gpu --save_every_epoch
```

## Evaluation
### Cmd
```
python models/eval/eval_seq2seq.py --model_path "exp/moca/best_seen.pth" --eval_split valid_seen --gpu --num_threads 4
```

### Example
If you want to evaluate our pretrained model saved in "exp/pretrained/pretrained.pth" in the seen validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```

## Submission
### Cmd
```
python models/eval/leaderboard.py --model_path  --num_threads 4
```

### Example
If you want to submit our pretrained model, "exp/pretrained/pretrained.pth", to the leaderboard, you may use the command below.
```
python models/eval/leaderboard.py --model_path "exp/pretrained/pretrained.pth" --num_threads 4
```

## Hardware 
Tested on:
- **GPU** - GTX 2080 Ti (12GB)
- **CPU** - Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- **RAM** - 32GB
- **OS** - Ubuntu 18.04

## License
MIT License

## Citation
```
@article{MOCA21,
  title ={{MOCA: A Modular Object-Centric Approach for Interactive Instruction Following}},
  author={{Kunal Pratap Singh* and Suvaansh Bhambri* and Byeonghwi Kim*} and Roozbeh Mottaghi and Jonghyun Choi},
  journal = {arXiv},
  year = {2021},
  url  = {https://arxiv.org/abs/}
}
```
