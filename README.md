# MOCA
<a href="http://arxiv.org/abs/2012.03208"> <b> MOCA: A Modular Object-Centric Approach for Interactive Instruction Following </b> </a>
<br>
<a href="https://kunalmessi10.github.io/"> Kunal Pratap Singh* </a>,
<a href="https://www.linkedin.com/in/suvaansh-bhambri-1784bab7/"> Suvaansh Bhambri* </a>,
<a href="https://bhkim94.github.io/"> Byeonghwi Kim* </a>,
<a href="http://roozbehm.info/"> Roozbeh Mottaghi </a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>

<b> MOCA </b> (<b>M</b>odular <b>O</b>bject-<b>C</b>entric <b>A</b>pproach) is a modular architecture that decouples a task into visual perception and action policy.
The action policy module (APM) is responsible for sequential action prediction, whereas the visual perception module (VPM) generates pixel-wise interaction mask for the objects of interest for manipulation.
MOCA addresses long-horizon instruction following tasks based on egocentric RGB observations and natural language instructions on the <a href="https://github.com/askforalfred/alfred">ALFRED</a> benchmark.

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
Dataset includes visual features extracted by ResNet-18 with natural language annotations (~135.5GB after unzipping).
For details of the ALFRED dataset, see the repository of <a href="https://github.com/askforalfred/alfred">ALFRED</a>.
```
$ cd $ALFRED_ROOT/data
$ sh download_data.sh
```
**Note**: The downloaded data includes expert trajectories with both original and color-swapped frames.

### Pretrained Model
We provide our pretrained weight used for the experiments in the paper and the leaderboard submission.
To download the pretrained weight of MOCA, use the command below.
```
$ cd $ALFRED_ROOT
$ sh download_model.sh
```

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

### Expected Validation Result
| Model      | Seen SR(%)                  | Seen GC (%)                 | Unseen SR (%)           | Unseen GC (%)             |
|:----------:|:---------------------------:|:---------------------------:|:-----------------------:|:-------------------------:|
| Reported   | 19.15        (13.60)        | 28.50 (22.30)               | 3.78 (2.00)             | 13.40 (8.30)              |
| Reproduced | 18.66\~19.27 (12.78\~13.63) | 27.79\~28.64 (21.50\~22.14) | 3.65\~3.78 (1.94\~1.99) | 13.40\~13.77 (8.22\~8.69) |

**Note**: "Reproduced" denotes the expected success rates of the pretrained model that we provide.


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
@article{singh2020moca,
  title={MOCA: A Modular Object-Centric Approach for Interactive Instruction Following},
  author={Singh, Kunal Pratap and Bhambri, Suvaansh and Kim, Byeonghwi and Mottaghi, Roozbeh and Choi, Jonghyun},
  journal={arXiv preprint arXiv:2012.03208},
  year={2020}
}
```
