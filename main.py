import argparse
import os
import random
import sys
from datetime import datetime
import numpy
import torch
from dateutil import tz #Deal with timezone flexibility
import scipy
from sklearn.metrics.pairwise import cosine_distances

import config  # import from the file config
from config import TASKS, PERSONALISATION, HUMOR, MIMIC, AROUSAL, VALENCE, PERSONALISATION_DIMS
 # from the file config import these variables
from data_parser import load_data
#  from file data_parser import the function load_data
from dataset import MuSeDataset, custom_collate_fn
# what is the custom_collate_fn
from eval import evaluate, calc_ccc, calc_auc, mean_pearsons
from loss import CCCLoss, BCELossWrapper, MSELossWrapper
from model import Model
from train import train_model
from utils import Logger, seed_worker, log_results


# Just to define and parse command-line arguments passed to the script
# Used when running the task
def parse_args():

    parser = argparse.ArgumentParser(description='MuSe 2023.')

    parser.add_argument('--task', type=str, required=True, choices=TASKS,
                        help=f'Specify the task from {TASKS}.')
    parser.add_argument('--feature', required=True,
                        help='Specify the features used (only one).')
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, only relevant for personalisation (default: {AROUSAL}).')
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    # When no test label, set --predict to True
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    # evaluation only arguments
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--eval_seed', type=str, default=None,
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')

    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args

# retrieve the correct loss function for the target task
def get_loss_fn(task):
    if task == PERSONALISATION:
        return CCCLoss(), 'CCC'
    elif task == HUMOR:
        return BCELossWrapper(), 'Binary Crossentropy'
    elif task == MIMIC:
        return MSELossWrapper(reduction='mean'), 'MSE'


# retrive the correct evaluation function for the target task
def get_eval_fn(task):
    if task == PERSONALISATION:
        return calc_ccc, 'CCC'
    elif task == MIMIC:
        return mean_pearsons, 'Mean Pearsons'
    elif task == HUMOR:
        return calc_auc, 'AUC-Score'

# the seed is set here only when used to ensure reproducibility
def main(args):
    # ensure reproducibility
    numpy.random.seed(10)
    random.seed(10)
    torch.manual_seed(args.seed)

    # emo_dim only relevant for stress/personalisation
    args.emo_dim = args.emo_dim if args.task==PERSONALISATION else ''
    print('Loading data ...')
    
    # load data from specified paths and creates dataset objects
    data = load_data(args.task, args.paths, args.feature, args.emo_dim, args.normalize,
                     args.win_len, args.hop_len, save=args.cache)
    
    train_datasets = data['train']
    devel_datasets = data['devel']
    # test_datasets = data['test']
    
    # Define a dictionary to extract individual 
    Distance_dic_list = {}
    # Use to store the distance dictionaries

    for i in range(0,len(devel_datasets['meta'])):
        distance_dic = {}
        for j in range(0,len(train_datasets['meta'])):
            P_sample_feature = devel_datasets['feature'][i][0:120] # Take the first minute of the training sample
            T_sample_feature = train_datasets['feature'][j][0:120] 
            distances = cosine_distances(P_sample_feature, T_sample_feature)
            Distance = numpy.sum(distances)
            distance_dic[train_datasets['meta'][j][0][0]] = Distance
            sorted_distance_dic = dict(sorted(distance_dic.items(), key=lambda item: item[1]))
        Distance_dic_list[devel_datasets['meta'][i][0][0]] = sorted_distance_dic

    augment_sample_n = 7
    # For every key value pairs in distance_Dic_list, cut 

    # In order to keep the consistency, only test the individuals with out "_devel/_train/_test" in the key
    Distance_dic_list_filtered = {key: value for key, value in Distance_dic_list.items() if "_devel" not in key}

    Distance_dic_list_augment = {}
    Dic_augment = {}
    
    for key, value in Distance_dic_list_filtered.items():
        # Create a new dictionary with only the first 7 items
        Distance_dic_list_augment = {k: v for i, (k, v) in enumerate(value.items()) if i < augment_sample_n}
        # Update the value in the original dictionary
        Dic_augment[key] = Distance_dic_list_augment

    Dic_augment_onlystring =  {key: list(sub_dict.keys()) for key, sub_dict in Dic_augment.items()}

    # Create a New "data"-like item, to store the corresponding data the same structure

        # 名字中不带 _devel：
        # training set: 前120 non _devel + 相似的所有training individual 
        # testing set: 120后 non _devel 分割每一个时刻

    # 新的testing set
    New_validation = {}
    New_validation['feature']=[]
    New_validation['label']=[]
    New_validation['meta']=[]
    New_training = {}
    New_training['feature']=[]
    New_training['label']=[]
    New_training['meta']=[]

    for i in range(0,len(data['devel']['meta'])): 
        if data['devel']['meta'][i][0][0] in list(Dic_augment_onlystring.keys()):
            New_validation['feature'].append(data['devel']['feature'][i][120:])
            New_validation['label'].append(data['devel']['label'][i][120:])
            New_validation['meta'].append(data['devel']['meta'][i][120:]) 
            
            New_training['meta'].append(data['devel']['meta'][i][:120])
            New_training['label'].append(data['devel']['label'][i][:120])
            New_training['feature'].append(data['devel']['feature'][i][:120]) 

    for i in list(Dic_augment_onlystring.keys()) :
        # iterate through all the keys of the distance dict,将augmenting individuals 的 features 和 label append 上去
        target_individual_indice = list(Dic_augment_onlystring.keys()).index(i)
        # Too see where to append
        for j in Dic_augment_onlystring[i]: #iterate through每一个similar individual
            indice = [p for p, sublist in enumerate(data['train']['meta']) if any(j in sublist for sublist in sublist)] #在大data里面找similar individual的位置

            New_training_individual_list = New_training['meta'][target_individual_indice].tolist()
            for n in indice: 
                new_meta = data['train']['meta'][n]
                for ind in new_meta:
                    ind[2] = i
                    New_training_individual_list.append(ind)
                    
                New_training_individual_array = numpy.array(New_training_individual_list)
                New_training['meta'][target_individual_indice] = New_training_individual_array
                
            # print('test it out')

            for n in indice: 
                new_training_individual_list = New_training['feature'][target_individual_indice].tolist()
                new_aug_feature = data['train']['feature'][n]
                for ind in new_aug_feature:
                    new_training_individual_list.append(ind)

                new_training_individual_array = numpy.array(new_training_individual_list, dtype='float64')
                New_training['feature'][target_individual_indice] = new_training_individual_array


            for n in indice: # Append the labels
                new_label = data['train']['label'][n]
                new_training_individual_list = New_training['label'][target_individual_indice].tolist()

                for ind in new_label:
                    new_training_individual_list.append(ind)

                new_training_individual_array = numpy.array(new_training_individual_list,dtype='float64')
                New_training['label'][target_individual_indice] = new_training_individual_array



    print('Data Augmentation Complete!')


    datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}
    
    new_data = {'train':New_training,'devel':New_validation}

    new_datasets = {partition:MuSeDataset(new_data,partition) for partition in new_data.keys()}
    
    # just record the feature_dimensionality for later model initialization
    # args.d_in = datasets['train'].get_feature_dim()
    args.d_in = datasets['train'].get_feature_dim() 

    # args.n_targets = output dimensions (number of targets associated with the specific task)
    args.n_targets = config.NUM_TARGETS[args.task]
    # Whether it is multiple input to one output (no, in personalisation)
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    # if args.eval_model is True, then just go straight to evaluate them, and if not, train and validate for each seed
    if args.eval_model is None:  # Train and validate for each seed

        # the seeds are continuous, and initialize the random number generator for reproducibility
        seeds = range(args.seed, args.seed + args.n_seeds)

        # valudation loss
        # validation scores
        # file that stores the best model
        # file that stores the test_scores (so the committee is supposed to find the test_score from this directory?)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds: 
            # move data initialisation below here...
            
            # sets the seed for the random number generator of PyTorch to the current seed vlue, 
            # ensures the random initialization of the model and the data loader will be consistent for each seed
            torch.manual_seed(seed) 
            data_loader = {}
            
            # Loop over the 'datasets'dictionay, where key is 'train','val','test' and value is the dataset object
            # for partition,dataset in datasets.items():
            for partition,dataset in new_datasets.items():  # one DataLoader for each partition
                # one DataLoader for train
                # one DataLoader for validation
                # one DataLoader for test

                # It uses arg.batch_size if the partition is 'train', 
                # otherwise, it sets the batch size to 1 if personalisation and 2 times the 'args.batch-size' for other task
                batch_size = args.batch_size if partition == 'train' else (
                    1 if args.task == PERSONALISATION else 2 * args.batch_size)
                
                # If partition is 'train', the data will be shuffled
                shuffle = True if partition == 'train' else False  # shuffle only for train partition

                # creates a 'DataLoader' object for each partition using the corresponding dataset, using the parameters determined as before
                # The resulting 'DataLoader' objects are stored in the 'data_loader'dictionary using the partition name as the key
                data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                     num_workers=4,
                                                                     worker_init_fn=seed_worker,
                                                                     collate_fn=custom_collate_fn)
            # creates an instance of the 'Model' class using the 'args' argument
            model = Model(args)

            # 分割符
            print('=' * 50)
            print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')

            # model training
            val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                               args.lr, args.paths['model'], seed,
                                                               loss_fn=loss_fn, eval_fn=eval_fn,use_gpu=args.use_gpu,
                                                               eval_metric_str=eval_str,
                                                               regularization=args.regularization,
                                                               early_stopping_patience=args.early_stopping_patience,
                                                               reduce_lr_patience=args.reduce_lr_patience)
            # restore best model encountered during training
            model = torch.load(best_model_file)

            # run evaluation
            if not args.predict:  # run evaluation only if test labels are available
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                 eval_fn=eval_fn, use_gpu=args.use_gpu)
                test_scores.append(test_score)
                print(f'[Test {eval_str}]:  {test_score:7.4f}')
            val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)
        
        best_idx = val_scores.index(max(val_scores))  # find best performing seed

        print('=' * 50)
        print(f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {eval_str}]: {val_scores[best_idx]:7.4f}'
              f"{f' | [Test {eval_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        print('=' * 50)

        model_file = best_model_files[best_idx]  # best model of all of the seeds
        # if there exists the csv to store results
        # it stores the results in to the result_csv with the following information:
        # params  
        # seds
        # metric
        # where is the mdoel
        # test scores
        # A list of validation scores obtained for each seed
        # the index of the best model in the 'best_model_files' list
        if not args.result_csv is None:
            log_results(args.result_csv, params=args, seeds = list(seeds), metric_name=eval_str,
                        model_files=best_model_files, test_results=test_scores, val_results=val_scores,
                        best_idx=best_idx)

    else:  # Evaluate existing model (No training)
        model_file = os.path.join(args.paths['model'], f'model_{args.eval_seed}.pth')
        model = torch.load(model_file, map_location=torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else (
                1 if args.task == PERSONALISATION else 2 * args.batch_size)
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        best_model = torch.load(model_file, map_location=config.device)
        print("development set of current data setting is being predicted:") #If the new_datasets does not contains the 'devel set' should disable this
        evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv')
        print("test set of the current data setting is on:")
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'], 
                 filename='predictions_test.csv')
        # These two evaluation function are just the same
        print(f'Find predictions in {os.path.join(args.paths["predict"])}')

    print('Done.')


if __name__ == '__main__':
    args = parse_args()

    # Naming the log_file
    args.log_file_name = '{}_{}_[{}]_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format('RNN',
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature, args.emo_dim,
        args.model_dim, args.rnn_n_layers, args.rnn_bi, args.d_fc_out, args.lr, args.batch_size) if args.task == PERSONALISATION else \
        '{}_{}_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format('RNN', datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature.replace(os.path.sep, "-"),
                                                 args.model_dim, args.rnn_n_layers, args.rnn_bi, args.d_fc_out, args.lr,args.batch_size)

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task) if not args.predict else os.path.join(config.LOG_FOLDER, args.task, 'prediction'),
                  'data': os.path.join(config.DATA_FOLDER, args.task),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name if not args.eval_model else args.eval_model)}
    if args.predict:
        if args.eval_model:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.eval_model, args.eval_seed)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    main(args)
