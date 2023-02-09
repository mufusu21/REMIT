# REMIT: Reinforced Multi-Interest Transfer for Cross-domain Recommendation

# **Prerequistes**

You should run the following command to install the necessary libs.

```jsx
pip -r requirements.txt
```

# **Datasets**

We test our algorithms on the Amazon review dataset. Specifically, we pick 3 datasets out of 24, i.e., movies and tv (Movie), cds and vinyl (Music), and books (Book). Three cross-domain recommendation tasks are built upon these three datasets: Movie → Music (Task 1), Book → Movie (Task 2) and Book →Movie (Task 3). 
 To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. The final data will be under `./data/ready`.

```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

For reproducibility, we provide pretrained source embeddings of the source domain in task 1 with $\beta = 80\%$, you can find them in the supplementary files, these embeddings are trained by HIN and stored in csv files. You can put them in the directory ‘./data/ready/_2_8/tgt_CDs_and_Vinyl_src_Movies_and_TV’. 
After this, you can reproduce the results in our paper by following the commands we give in the next section.

# Run

These  arguments are available for users to adjust

```bash
--task, 1:Moive->Music|2:Book->Movie|3:Book->Music
--base_model,  the base model to pretrain user and item embeddings for both domains.
--seed, input random seed
--ratio, split ratio for given task
--gpu, gpu id
--epoch, epoch number
--lr, learning rate
--use_cuda, whether to user gpu
--rl_lr, learning rate for reinforcement learning
--algo,  sscdr|cmf|tgt|ptupcdr|remit
```

To reproduce our results on task 1, you can run the following command.

```bash
 python entry.py --algo remit --base_model MF --task 1 --ratio 2 8 --epoch 15 --use_cuda 1 --lr 0.01
```

Besides, you can specify your own seed with ‘--speed’ option.

```bash
python entry.py --algo remit --base_model MF --task 1 --ratio 2 8 --epoch 15 --use_cuda 0 --lr 0.01 --seed 2020
```