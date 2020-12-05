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
### Clone repository
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


## Download
### Dataset
Dataset includes visual features extracted by ResNet-18 with natural language annotations.
For details of the ALFRED dataset, see the repository of <a href="https://github.com/askforalfred/alfred">ALFRED</a>.
```
$ cd $ALFRED_ROOT/data
$ sh download_data.sh
```
**Note**: The downloaded data includes expert trajectories with both original and color-swapped frames.

### Pretrained MOCA
We provide our pretrained weight used for the experiments in the paper and the leaderboard submission.
To download the pretrained weight of MOCA, use the command below.
```
$ cd $ALFRED_ROOT/exp/pretrained
$ sh download_pretrained_weight.sh
```

### Pretrained Mask R-CNN
We provide our pretrained Mask R-CNN weight used for the experiments in the paper and the leaderboard submission.
To download the pretrained weight of the Mask R-CNN, use the command below.
```
$ cd $ALFRED_ROOT
$ sh download_maskrcnn.sh


## Training
To train MOCA, run `train_seq2seq.py` with hyper-parameters below. <br>
```
python models/train/train_seq2seq.py --data <path_to_dataset> --model seq2seq_im_mask --dout <path_to_save_weight> --splits data/splits/oct21.json --gpu --batch <batch_size> --pm_aux_loss_wt <pm_aux_loss_wt_coeff> --subgoal_aux_loss_wt <subgoal_aux_loss_wt_coeff> --preprocess
```
**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. <br>
**Note**: All hyperparameters used for the experiments in the paper are set as default.

For example, if you want train MOCA and save the weights for all epochs in "exp/moca" with all hyperparameters used in the experiments in the paper, you may use the command below. <br>
```
python models/train/train_seq2seq.py --dout exp/moca --gpu --save_every_epoch
```
**Note**: The option, `--save_every_epoch`, saves weights for all epochs and therefore could take a lot of space.


## Evaluation
### Task Evaluation
To evaluate MOCA, run `eval_seq2seq.py` with hyper-parameters below. <br>
To evaluate a model in the `seen` or `unseen` environment, pass `valid_seen` or `valid_unseen` to `--eval_split`.
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```

### Subgoal Evaluation
To evaluate MOCA for subgoals, run `eval_seq2seq.py` with with the option `--subgoals <subgoals>`. <br>
The option takes `all` for all subgoals and `GotoLocation`, `PickupObject`, `PutObject`, `CoolObject`, `HeatObject`, `CleanObject`, `SliceObject`, and `ToggleObject` for each subgoal.
The option can take multiple subgoals.
For more details, refer to <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>.
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num> --subgoals <subgoals>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation for all subgoals, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4 --subgoals all
```

<!--
## Submission
To submit MOCA to the leaderboard, run `eval_seq2seq.py` with hyper-parameters below.
This saves `tests_actseqs_dump_<save_time>.json` which is submitted to the leaderboard.
```
python models/eval/leaderboard.py --model_path <path_to_weight> --num_threads 4
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to submit our pretrained model saved in `exp/pretrained/pretrained.pth` to the leaderboard, you may use the command below.
```
python models/eval/leaderboard.py --model_path "exp/pretrained/pretrained.pth" --num_threads 4
```
-->


## Hardware 
Trained and Tested on:
- **GPU** - GTX 2080 Ti (11GB)
- **CPU** - Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- **RAM** - 32GB
- **OS** - Ubuntu 18.04


## License
MIT License


## Citation
```
@article{moca21,
  title ={{MOCA: A Modular Object-Centric Approach for Interactive Instruction Following}},
  author={{Kunal Pratap Singh* and Suvaansh Bhambri* and Byeonghwi Kim*} and Roozbeh Mottaghi and Jonghyun Choi},
  journal = {arXiv},
  year = {2021},
  url  = {https://arxiv.org/abs/}
}
```
