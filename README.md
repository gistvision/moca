# MOCA
<a href=""> <b> MOCA: A Modular Object-Centric Approach for Interactive Instruction Following </b> </a>
<br>
<a href=""> Kunal Pratap Singh* </a>,
<a href=""> Suvaansh Bhambri* </a>,
<a href=""> Byeonghwi Kim* </a>,
<a href="http://roozbehm.info/"> Roozbeh Mottaghi </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>

<b> MOCA </b> (<b>M</b>odular <b>O</b>bject-<b>C</b>entric <b>A</b>pproach) is a modular architecture that decouples a task into visual perception and action policy.
The action policy module (APM) is responsiblefor sequential action prediction, whereas the visual perception module (VPM) generates pixel-wise interaction maskfor the objects of interest for manipulation.

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
Dataset includes visual features extracted by ResNet-18 with natural language annotations.
For details of the ALFRED dataset, please refer to the repository of <a href="https://github.com/askforalfred/alfred">ALFRED</a>.
```bash
$ cd $ALFRED_ROOT/data
$ sh download_data.sh
```
**Note**: Note that this includes expert trajectories with both original and color-swapped frames.


## Training
To train MOCA, run `train_seq2seq.py` with hyper-parameters below. <br>
```
python models/train/train_seq2seq.py --data <path_to_dataset> --model seq2seq_im_mask --dout <path_to_save_weight> --splits data/splits/oct21.json --gpu --batch <batch_size> --pm_aux_loss_wt <pm_aux_loss_wt_coeff> --subgoal_aux_loss_wt <subgoal_aux_loss_wt_coeff> --preprocess
```

For example, if you want train MOCA and save the weights for all epochs in "exp/moca" with all hyperparameters used in the experiments in the paper, you may use the command below. <br>
```
python models/train/train_seq2seq.py --dout exp/moca --gpu --save_every_epoch
```

**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. <br>
**Note**: All hyperparameters used for the experiments in the paper are set as default.
**Note**: The option, `--save_every_epoch`, saves weights for all epochs and therefore could take a lot of space.


## Evaluation
To evaluate MOCA, run `eval_seq2seq.py` with hyper-parameters below. <br>
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num>
```

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```

**Note**: All hyperparameters used for the experiments in the paper are set as default.


## Submission
To evaluate MOCA, run `eval_seq2seq.py` with hyper-parameters below. <br>
```
python models/eval/leaderboard.py --model_path  --num_threads 4
```

If you want to submit our pretrained model, "exp/pretrained/pretrained.pth", to the leaderboard, you may use the command below.
```
python models/eval/leaderboard.py --model_path "exp/pretrained/pretrained.pth" --num_threads 4
```

**Note**: All hyperparameters used for the experiments in the paper are set as default.

## Hardware 
Trained and Tested on:
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
