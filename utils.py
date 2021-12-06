'''
Utils script
This script is modified from 'https://github.com/YuejiangLIU/social-lstm-pytorch' by Anirudh Vemula
Author: Pu Zhang
Date: 2019/7/1
'''
import torch
import gc
import os
import pickle
import numpy as np
import scipy.linalg as sl
import random
import sys
import copy


def debug(msg, kill=False):
    print(f'~~~~~~~~ {msg}')
    sys.stdout.flush()
    if kill:
        exit("Killed by debugger")


class DataLoader_bytrajec2():
    def __init__(self, args):

        self.args = args
        if self.args.dataset == 'eth5':

            self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                              './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                              './data/ucy/univ/students001', 'data/ucy/univ/students003',
                              './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']

            # Data directory where the pre-processed pickle file resides
            self.data_dir = './data'
            skip = [6, 10, 10, 10, 10, 10, 10, 10]

            if args.ifvalid:
                self.val_fraction = args.val_fraction
            else:
                self.val_fraction = 0

            train_set = [i for i in range(len(self.data_dirs))]
            if args.test_set == 4 or args.test_set == 5:
                self.test_set = [4, 5]
            else:
                self.test_set = [int(self.args.test_set)]

            for x in self.test_set:
                train_set.remove(x)

            # debug(f'train_set {train_set}, test_set {self.test_set}', kill=True)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        self.train_data_file = os.path.join(self.args.save_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.save_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")

        print("Creating pre-processed data from raw data.")
        self.traject_preprocess('train')
        self.traject_preprocess('test')
        print("Done.")

        # Load the processed data from the pickle file
        print("Preparing data batches.")
        if not (os.path.exists(self.train_batch_cache)):
            self.frameped_dict, self.pedtraject_dict = self.load_dict(self.train_data_file)
            self.data_preprocess('train')
        if True:  ## debugging !!!!!
            # if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
            self.data_preprocess('test')

        self.train_data, self.trainbatchnums, \
        self.val_data, self.valbatchnums = self.load_cache(self.train_batch_cache)
        self.test_data, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)
        print("Done.")

        print('Total number of training batches:', self.trainbatchnums)
        print('Total number of validation batches:', self.valbatchnums)
        print('Total number of test batches:', self.testbatchnums)

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)

    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        all_frame_data = []
        valid_frame_data = []
        traj_len_data = []

        pedlist_data = []
        frame_ped_dict = []  # peds id contained in a certain frame
        ped_trajec_dict = []  # trajectories of a certain ped
        # For each dataset
        for scene_id, directory in enumerate(data_dirs):
            '''Each directory is a dataset scene: hotel, ...'''
            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            '''data: dim (fields, all frames and all peds)'''
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset
            # debug(f'Raw data from one file: shape {data.shape}, data 0 {data[:, 0]}', kill=True)
            '''pedlist: peds in the dataset'''
            pedlist = np.unique(data[1, :]).tolist()
            numPeds = len(pedlist)
            # Add the list of frameIDs to the frameList_data
            pedlist_data.append(pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            '''traj_len_data: length of trajectories in data scenes
            list of list
            dim0: scene
            dim1: ped
            '''
            traj_len_data.append([])
            '''
            frame_ped_dict: peds in frames for all scenes
            list of dict
            dim0: scene
            dict key: frame id
            dict value: list of ped ids
            '''
            frame_ped_dict.append({})
            '''
            ped_trajec_dict: trajectory of peds in all scenes
            list of dict
            dim0: scene
            dict key: ped id
            dict value: trajectory, dim = (len, 3), each point is (frame id, x, y)
            '''
            ped_trajec_dict.append({})

            for idx, ped_id in enumerate(pedlist):
                if idx % 100 == 0:
                    print(idx, len(pedlist))
                # Extract trajectories of one person
                ped_frames = data[:, data[1, :] == ped_id]
                # Extract peds list
                '''ped_frame_id_list: list of time frames for the pedestrian'''
                ped_frame_id_list = ped_frames[0, :].tolist()
                if len(ped_frame_id_list) < 2:
                    continue
                # Add number of frames of this trajectory
                traj_len_data[scene_id].append(len(ped_frame_id_list))
                # Initialize the row of the numpy array
                '''trajectories: 
                dim = (time frames, 3)
                Each point: frame id, x, y  
                '''
                trajectories = []
                # For each ped in the current frame

                for f_idx, frame_id in enumerate(ped_frame_id_list):
                    # Extract their x and y positions
                    current_x = ped_frames[3, ped_frames[0, :] == frame_id][0]
                    current_y = ped_frames[2, ped_frames[0, :] == frame_id][0]
                    # Add their pedID, x, y to the row of the numpy array
                    trajectories.append([int(frame_id), current_x, current_y])
                    if int(frame_id) not in frame_ped_dict[scene_id]:
                        frame_ped_dict[scene_id][int(frame_id)] = []
                    frame_ped_dict[scene_id][int(frame_id)].append(ped_id)
                ped_trajec_dict[scene_id][ped_id] = np.array(trajectories)

        f = open(data_file, "wb")
        pickle.dump((frame_ped_dict, ped_trajec_dict), f, protocol=2)
        f.close()

    def get_data_index(self, frame_ped_dict_list, set_name, ifshuffle=True):
        '''
        Get the dataset sampling index.
        '''
        frame_scene_idx_list = []
        frame_id_list = []
        total_frame = 0
        for scene_idx, frame_ped_dict in enumerate(frame_ped_dict_list):
            frames = sorted(frame_ped_dict)
            # debug(f'frame_ped_dict {frame_ped_dict.keys()}\n'
            #       f'sorted: {frames}', kill=True)
            max_frame = max(frames) - self.args.seq_length
            frames = [x for x in frames if not x > max_frame]
            total_frame += len(frames)
            frame_scene_idx_list.extend(list(scene_idx for i in range(len(frames))))
            frame_id_list.extend(list(frames[i] for i in range(len(frames))))
        frame_idx_list = list(i for i in range(total_frame))
        debug(f'frame_id_list len {len(frame_id_list)},'
              f'frame_scene_idx_list len {len(frame_scene_idx_list)},'
              f'frame_idx_list len {len(frame_idx_list)}')

        data_index = np.concatenate((np.array([frame_id_list], dtype=int), np.array([frame_scene_idx_list], dtype=int),
                                     np.array([frame_idx_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(frame_idx_list)
        data_index = data_index[:, frame_idx_list]

        # to make full use of the data
        if set_name == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)

        debug(f'data_index shape {data_index.shape}')
        return data_index

    def get_seq_from_index(self, frame_ped_dict, ped_traject_dict, data_index, set_name):
        """
        Query the trajectories fragments from data sampling index.
        Inputs:
            frame_ped_dict: peds in frames for all scenes
                list of dict
                dim0: scene
                dict key: frame id
                dict value: list of ped ids
            ped_trajec_dict: trajectory of peds in all scenes
                list of dict
                dim0: scene
                dict key: ped id
                dict value: trajectory, dim = (len, 3), each point is (frame id, x, y)
            data_index
                dim = (3, total number of frames in all scenes)
                each point in dim 0 is (frame_id, scene_id, frame_idx (starting from 0))
        :returns:
            multi_batch_data: list of tuple, len = number of batches, each tuple is (batch_data, batch_key_list)
                batch_data: tuple
                    seq_batch: dim = (seq_len, peds, 2)
                    seq_data_mask_batch: dim = (seq_len, peds)
                    seq_adjacency_matrix_batch: dim = (seq_len, peds, peds)
                    seq_neighbor_num_batch: dim = (seq_len, peds)
                    ped_num_batch: list of int
                    seq_ped_ids: dim = (seq_len, peds)
                batch_key_list: list of len batch size
                    each data is (scene, frame_id)
        """
        batch_data_list = []
        seq_batch = []
        batch_key_list = []
        seq_ped_ids_list = []
        seq_start_frame_list = []

        if set_name == 'train':
            skip = self.trainskip
            batch_size = self.args.batch_size
        else:
            skip = self.testskip
            batch_size = self.args.test_batch_size
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            peds_at_start_frame = set(frame_ped_dict[cur_set][cur_frame])
            try:
                peds_at_end_frame = set(frame_ped_dict[cur_set][cur_frame + self.args.seq_length * skip[cur_set]])
            except:
                continue
            peds_in_clip = peds_at_start_frame | peds_at_end_frame
            if (peds_at_start_frame & peds_at_end_frame).__len__() == 0:
                continue
            clip_trajectories = ()
            ped_ids = ()
            start_frame_ids = ()
            if_full = []
            for ped in peds_in_clip:
                cur_trajectory, if_full, if_exist_obs = self.find_trajectory_fragment(ped_traject_dict[cur_set][ped],
                                                                                      cur_frame,
                                                                                      self.args.seq_length,
                                                                                      skip[cur_set])
                if len(cur_trajectory) == 0:
                    continue
                if not if_exist_obs:
                    # Just ignore trajectories if their data don't exist at the last versed time step
                    continue
                if sum(cur_trajectory[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue

                cur_ped_ids = np.full((cur_trajectory.shape[0],), ped)
                cur_start_frame_ids = np.full((cur_trajectory.shape[0],), cur_frame)

                cur_trajectory = (cur_trajectory[:, 1:].reshape(-1, 1, self.args.input_size),)
                cur_ped_ids = (cur_ped_ids[:].reshape(-1, 1),)
                cur_start_frame_ids = (cur_start_frame_ids[:].reshape(-1, 1),)

                clip_trajectories = clip_trajectories.__add__(cur_trajectory)
                ped_ids = ped_ids.__add__(cur_ped_ids)
                start_frame_ids = start_frame_ids.__add__(cur_start_frame_ids)
                if_full.append(if_full)

            if clip_trajectories.__len__() < 1:
                continue
            if sum(if_full) < 1:
                continue
            '''
            clip_trajectories: a set of trajectories in the clip
            size: num peds
            value: dim = (seq_length, 2)
            ped_ids: ped ids of the set of trajectories in the clip
            size: num peds
            value: dim = (seq_length,)
            '''
            '''
            traject_batch: a batch of trajectories
            dim = (seq_length, num_peds, 2)
            '''
            traject_batch = np.concatenate(clip_trajectories, 1)
            ped_id_batch = np.concatenate(ped_ids, 1)
            start_frame_id_batch = np.concatenate(start_frame_ids, 1)

            seq_batch.append(traject_batch)
            seq_ped_ids_list.append(ped_id_batch)
            seq_start_frame_list.append(start_frame_id_batch)

            # debug(f'traject_batch shape {traject_batch.shape}, ped_id_batch.shape = {ped_id_batch.shape} \n'
            #       f'ped_id_batch = {ped_id_batch}, start_frame_id_batch = {start_frame_id_batch.shape}, '
            #       f'', kill=True)

            if set_name == 'test':
                real_set = self.args.test_set
            else:
                if cur_set >= self.test_set[0]:
                    real_set = cur_set + len(self.test_set)
                else:
                    real_set = cur_set
            batch_key_list.append((real_set, cur_frame,))
            if len(seq_batch) == batch_size:
                batch_data = self.prepare_data_batch(seq_batch, seq_ped_ids_list, seq_start_frame_list)
                batch_data_list.append((batch_data, batch_key_list,))
                seq_batch = []
                batch_key_list = []
                seq_ped_ids_list = []
                seq_start_frame_list = []
        return batch_data_list

    def get_seq_from_index_balance(self, frame_ped_dict, ped_traject_dict, data_index, set_name):
        """
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        Inputs:
            frame_ped_dict: peds in frames for all scenes
                list of dict
                dim0: scene
                dict key: frame id
                dict value: list of ped ids
            ped_trajec_dict: trajectory of peds in all scenes
                list of dict
                dim0: scene
                dict key: ped id
                dict value: trajectory, dim = (len, 3), each point is (frame id, x, y)
            data_index
                dim = (3, total number of frames in all scenes)
                each point in dim 0 is (frame_id, scene_id, frame_idx (starting from 0))
        :returns:
            batch_data_list: list of tuple, len = number of batches, each tuple is (batch_data, batch_key_list)
                batch_data: tuple
                    seq_batch: dim = (seq_len, peds, 2)
                    seq_data_mask_batch: dim = (seq_len, peds)
                    seq_adjacency_matrix_batch: dim = (seq_len, peds, peds)
                    seq_neighbor_num_batch: dim = (seq_len, peds)
                    ped_num_batch: list of int
                    seq_ped_ids: dim = (seq_len, peds)
                batch_key_list: list of len batch size
                    each data is (scene, frame_id)
        """
        batch_data_list = []
        seq_batch = []
        batch_key_list = []
        seq_ped_ids_list = []
        seq_start_frame_list = []

        if set_name == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip

        ped_cnt = 0
        last_frame = 0
        for i in range(data_index.shape[1]):  # for all frames
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            peds_at_start_frame = set(frame_ped_dict[cur_set][cur_frame])
            try:
                peds_at_end_frame = set(frame_ped_dict[cur_set][cur_frame + self.args.seq_length * skip[cur_set]])
            except:
                continue
            peds_in_clip = peds_at_start_frame | peds_at_end_frame
            if (peds_at_start_frame & peds_at_end_frame).__len__() == 0:
                continue
            clip_trajectories = ()
            if_full_list = []
            ped_ids = ()
            start_frame_ids = ()
            for ped in peds_in_clip:

                cur_trajectory, if_full, if_exist_obs = self.find_trajectory_fragment(
                    ped_traject_dict[cur_set][ped], cur_frame, self.args.seq_length, skip[cur_set])
                if len(cur_trajectory) == 0:
                    continue
                if if_exist_obs == False:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajectory[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                '''
                cur_trajectory: frame id omitted, only with xy
                dim = (seq_length, 1 ,2), dim1 is for combining with batch
                '''
                cur_ped_ids = np.full((cur_trajectory.shape[0],), ped)
                cur_start_frame_ids = np.full((cur_trajectory.shape[0],), cur_frame)

                cur_trajectory = (cur_trajectory[:, 1:].reshape(-1, 1, self.args.input_size),)
                cur_ped_ids = (cur_ped_ids[:].reshape(-1, 1),)
                cur_start_frame_ids = (cur_start_frame_ids[:].reshape(-1, 1),)

                clip_trajectories = clip_trajectories.__add__(cur_trajectory)
                ped_ids = ped_ids.__add__(cur_ped_ids)
                start_frame_ids = start_frame_ids.__add__(cur_start_frame_ids)
                if_full_list.append(if_full)
            if clip_trajectories.__len__() < 1:
                continue
            if sum(if_full_list) < 1:
                continue
            '''
            clip_trajectories: a set of trajectories in the clip
            size: num peds
            value: dim = (seq_length, 2)
            '''
            '''
            traject_batch: a batch of trajectories
            dim = (seq_length, num_peds, 2)
            '''
            traject_batch = np.concatenate(clip_trajectories, 1)
            ped_id_batch = np.concatenate(ped_ids, 1)
            start_frame_id_batch = np.concatenate(start_frame_ids, 1)

            # seq_ped_ids_list.append(ped_id_batch)
            # seq_start_frame_list.append(start_frame_id_batch)
            # total number of peds collated in the current batch
            batch_ped_num = sum([i.shape[1] for i in seq_batch]) + traject_batch.shape[1]
            cur_ped_num = traject_batch.shape[1]
            ped_cnt += cur_ped_num
            batch_key = (cur_set, cur_frame,)

            if cur_ped_num >= int(self.args.batch_around_ped) * 2:
                # too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_seq_batch, cur_batch_key_list = [], []
                cur_seq_ped_ids_list, cur_seq_start_frame_list = [], []
                seq_batchs = [traject_batch[:, ind[:cur_ped_num // 2, 0]], traject_batch[:, ind[cur_ped_num // 2:, 0]]]
                pid_batchs = [ped_id_batch[:, ind[:cur_ped_num // 2, 0]], ped_id_batch[:, ind[cur_ped_num // 2:, 0]]]
                sfid_batchs = [start_frame_id_batch[:, ind[:cur_ped_num // 2, 0]], start_frame_id_batch[:, ind[cur_ped_num // 2:, 0]]]

                for (sb, pidb, sfidb) in zip(seq_batchs, pid_batchs, sfid_batchs):
                    cur_seq_batch.append(sb)
                    cur_seq_ped_ids_list.append(pidb)
                    cur_seq_start_frame_list.append(sfidb)
                    cur_batch_key_list.append(batch_key)
                    cur_seq_batch = self.prepare_data_batch(cur_seq_batch, cur_seq_ped_ids_list, cur_seq_start_frame_list)
                    batch_data_list.append((cur_seq_batch, cur_batch_key_list,))
                    cur_seq_batch = []
                    cur_seq_ped_ids_list = []
                    cur_seq_start_frame_list = []
                    cur_batch_key_list = []

                last_frame = i
            elif cur_ped_num >= int(self.args.batch_around_ped):
                # good pedestrian numbers
                cur_seq_batch, cur_batch_key_list = [], []
                cur_seq_ped_ids_list, cur_seq_start_frame_list = [], []
                cur_seq_batch.append(traject_batch)
                cur_seq_ped_ids_list.append(ped_id_batch)
                cur_seq_start_frame_list.append(start_frame_id_batch)

                cur_batch_key_list.append(batch_key)
                cur_seq_batch = self.prepare_data_batch(cur_seq_batch, cur_seq_ped_ids_list, cur_seq_start_frame_list)
                batch_data_list.append((cur_seq_batch, cur_batch_key_list,))

                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_ped_num > int(self.args.batch_around_ped):
                    # enough people in the scene
                    seq_batch.append(traject_batch)
                    seq_ped_ids_list.append(ped_id_batch)
                    seq_start_frame_list.append(start_frame_id_batch)
                    batch_key_list.append(batch_key)

                    seq_batch = self.prepare_data_batch(seq_batch, seq_ped_ids_list, seq_start_frame_list)
                    batch_data_list.append((seq_batch, batch_key_list,))

                    last_frame = i
                    seq_batch = []
                    seq_ped_ids_list = []
                    seq_start_frame_list = []
                    batch_key_list = []
                else:
                    seq_batch.append(traject_batch)
                    seq_ped_ids_list.append(ped_id_batch)
                    seq_start_frame_list.append(start_frame_id_batch)
                    batch_key_list.append(batch_key)
        '''
        seq_batch: a batch of clip trajectories
        size = batch_size
        dim = (seq_length, num_peds_in_clip, 2)
        '''
        if last_frame < data_index.shape[1] - 1 and set_name == 'test' and batch_ped_num > 1:
            batch_data = self.prepare_data_batch(seq_batch, seq_ped_ids_list, seq_start_frame_list)
            batch_data_list.append((batch_data, batch_key_list,))

        return batch_data_list

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]

        # debug(f'Loading raw trajectories: \n'
        #       f'frameped_dict={frameped_dict}\n'
        #       f'pedtraject_dic={pedtraject_dict}', kill=True)

        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def data_preprocess(self, set_name):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if set_name == 'train':
            val_fraction = self.args.val_fraction
            frame_ped_dict = self.frameped_dict
            ped_traject_dict = self.pedtraject_dict
            cache_file = self.train_batch_cache

        else:
            val_fraction = 0
            frame_ped_dict = self.test_frameped_dict
            ped_traject_dict = self.test_pedtraject_dict
            cache_file = self.test_batch_cache

        data_index = self.get_data_index(frame_ped_dict, set_name)
        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]

        train_data = self.get_seq_from_index_balance(frame_ped_dict, ped_traject_dict, train_index, set_name)
        val_data = self.get_seq_from_index_balance(frame_ped_dict, ped_traject_dict, val_index, set_name)

        train_batch_nums = len(train_data)
        val_batch_nums = len(val_data)

        # num_batches=int(train_batch_nums/batch_size)
        # val_num_batches=int(val_batch_nums/self.args.val_batch_size)

        f = open(cache_file, "wb")
        pickle.dump((train_data, train_batch_nums, val_data, val_batch_nums), f, protocol=2)
        f.close()

    def find_trajectory_fragment(self, trajectory, start_frame, seq_length, skip):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
        '''
        return_trajectory = np.zeros((seq_length, 3))
        end_frame = start_frame + seq_length * skip
        # returns the indices of the elements that are True
        start_frame_idx = np.where(trajectory[:, 0] == start_frame)
        end_frame_idx = np.where(trajectory[:, 0] == end_frame)
        # debug(f'trajectory={trajectory}, '
        #       f'start_frame_idx={start_frame_idx[0]}, '
        #       f'end_frame_idx={end_frame_idx[0]}', kill=True)

        if_full = False
        if_exsit_obs = False

        if start_frame_idx[0].shape[0] == 0 and end_frame_idx[0].shape[0] != 0:
            start_frame_idx = 0
            end_frame_idx = end_frame_idx[0][0]
            if end_frame_idx == 0:
                return return_trajectory, if_full, if_exsit_obs

        elif end_frame_idx[0].shape[0] == 0 and start_frame_idx[0].shape[0] != 0:
            start_frame_idx = start_frame_idx[0][0]
            end_frame_idx = trajectory.shape[0]

        elif end_frame_idx[0].shape[0] == 0 and start_frame_idx[0].shape[0] == 0:
            start_frame_idx = 0
            end_frame_idx = trajectory.shape[0]

        else:
            end_frame_idx = end_frame_idx[0][0]
            start_frame_idx = start_frame_idx[0][0]

        candidate_seq = trajectory[start_frame_idx:end_frame_idx]
        offset_start = int((candidate_seq[0, 0] - start_frame) // skip)

        offset_end = self.args.seq_length + int((candidate_seq[-1, 0] - end_frame) // skip)

        return_trajectory[offset_start:offset_end + 1, :3] = candidate_seq

        if return_trajectory[self.args.obs_length - 1, 1] != 0:
            if_exsit_obs = True

        if offset_end - offset_start >= seq_length - 1:
            if_full = True

        return return_trajectory, if_full, if_exsit_obs

    def prepare_data_batch(self, batch_data, seq_ped_ids_list, seq_start_frame_list):
        """
        Massed up data fragements in different time window together to a batch
        :param
            batch_data: a batch of clip trajectories
                size = batch_size
                dim = (seq_length, num_peds_in_clip, 2)
            seq_ped_ids_list: ped ids in a batch of clip trajectories
            seq_start_frame_list: start frame ids in a batch of clip trajectories
        :return:
            seq_batch: dim = (seq_len, peds, 2)
            seq_data_mask_batch: dim = (seq_len, peds)
            seq_adjacency_matrix_batch: dim = (seq_len, peds, peds)
            seq_neighbor_num_batch: dim = (seq_len, peds)
            ped_num_batch: list of int
            seq_ped_ids: dim = (seq_len, peds)
        """

        total_num_peds = 0
        for batch in batch_data:
            total_num_peds += batch.shape[1]

        seq_data_mask_batch = np.zeros((self.args.seq_length, 0))
        seq_batch = np.zeros((self.args.seq_length, 0, 2))

        seq_adjacency_matrix_batch = np.zeros((self.args.seq_length, total_num_peds, total_num_peds))
        seq_neighbor_num_batch = np.zeros((self.args.seq_length, total_num_peds))
        seq_ped_ids = np.zeros((self.args.seq_length, total_num_peds))
        seq_start_frame_ids = np.zeros((self.args.seq_length, total_num_peds))

        num_peds = 0
        ped_num_batch = []
        for batch, ped_ids, start_frames in zip(batch_data, seq_ped_ids_list, seq_start_frame_list):
            '''
            batch: dim = (seq_length, num_peds_in_clip, 2)
            ped_ids: dim = (seq_length, num_peds_in_clip)
            start_frames: dim = (seq_length, num_peds_in_clip)
            '''
            # debug(f'batch shape {batch.shape}, ped_ids shape {ped_ids.shape}', kill=True)
            num_ped_batch = batch.shape[1]
            seq_data_mask, seq_adjacency_matrix, seq_neighbor_num = self.get_social_inputs_numpy(batch)
            seq_batch = np.append(seq_batch, batch, 1)
            seq_data_mask_batch = np.append(seq_data_mask_batch, seq_data_mask, 1)
            seq_adjacency_matrix_batch[:, num_peds:num_peds + num_ped_batch, num_peds:num_peds + num_ped_batch] \
                = seq_adjacency_matrix
            seq_neighbor_num_batch[:, num_peds:num_peds + num_ped_batch] = seq_neighbor_num
            seq_ped_ids[:, num_peds:num_peds + num_ped_batch] = ped_ids
            seq_start_frame_ids[:, num_peds:num_peds + num_ped_batch] = start_frames
            ped_num_batch.append(num_ped_batch)
            num_peds += num_ped_batch

        return seq_batch, seq_data_mask_batch, seq_adjacency_matrix_batch, seq_neighbor_num_batch, \
               ped_num_batch, seq_ped_ids, seq_start_frame_ids

    def get_social_inputs_numpy(self, trajectories):
        """
        :param trajectories: dim = (seq_length, num_peds, 2)
        :return: seq_data_mask: dim = (seq_len, peds_in_clip), value 0 or 1;
            seq_adjacency_matrix: dim = (seq_len, peds_in_clip, peds_in_clip), value 0 or 1;
            seq_neighbor_num: dim = (seq_len, peds_in_clip), value int;
        """
        num_peds = trajectories.shape[1]

        seq_data_mask = np.zeros((trajectories.shape[0], num_peds))
        # denote where the data lies
        for pedi in range(num_peds):
            seq = trajectories[:, pedi]
            seq_data_mask[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        seq_adjacency_matrix = np.zeros((trajectories.shape[0], num_peds, num_peds))
        seq_neighbor_num = np.zeros((trajectories.shape[0], num_peds))

        # seq_adjacency_matrix[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_peds):
            seq_adjacency_matrix[:, pedi, :] = seq_data_mask
            seq_adjacency_matrix[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            seq_neighbor_num[:, pedi] = np.sum(seq_adjacency_matrix[:, pedi, :], 1)
            pedi_seq = trajectories[:, pedi]
            for pedj in range(num_peds):
                pedj_seq = trajectories[:, pedj]
                select = (seq_data_mask[:, pedi] > 0) & (seq_data_mask[:, pedj] > 0)

                relative_cord = pedi_seq[select, :2] - pedj_seq[select, :2]

                # invalid data index
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred)

                seq_neighbor_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                seq_adjacency_matrix[select, pedi, pedj] = 0
        # debug(f'seq_data_mask shape {seq_data_mask.shape}, '
        #       f'seq_adjacency_matrix shape {seq_adjacency_matrix.shape}, '
        #       f'seq_neighbor_num shape {seq_neighbor_num.shape}', kill=True)
        return seq_data_mask, seq_adjacency_matrix, seq_neighbor_num

    def rotate_shift_batch(self, batch_data, ifrotate=True):
        '''
        Random ration and zero shifting.
        '''
        seq_batch, seq_data_mask_batch, seq_adjacency_matrix_batch, nei_num, batch_pednum, \
        seq_ped_ids, seq_start_frame_ids = batch_data

        # rotate seq_batch
        if ifrotate:
            th = random.random() * np.pi
            cur_ori = seq_batch.copy()
            seq_batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            seq_batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)
        # get shift value
        pred_start_xys = seq_batch[self.args.obs_length - 1]
        # debug(f'pred_start_xys shape {pred_start_xys.shape}, pred_start_xys.reshape((1, -1, 2) shape {pred_start_xys.reshape((1, -1, 2)).shape}')

        translations = np.repeat(pred_start_xys.reshape((1, -1, 2)), self.args.seq_length, 0)
        # debug(f'seq_batch shape {seq_batch.shape}, translations shape={translations.shape}', kill=True)

        seq_batch_translated = seq_batch - translations
        batch_data = seq_batch, seq_batch_translated, translations, seq_data_mask_batch, \
                     seq_adjacency_matrix_batch, nei_num, batch_pednum, seq_ped_ids, seq_start_frame_ids
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.train_data[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)

        return batch_data, batch_id

    def get_val_batch(self, idx):
        batch_data, batch_id = self.val_data[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)
        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_key_list = self.test_data[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)

        return batch_data, batch_key_list

    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0


def get_loss_masks(output_seq, seq_data_masks_first, seq_data_mask_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    :returns:
        loss_masks: dim = (seq_length, num_peds)
    '''
    seq_length = output_seq.shape[0]
    node_pre = seq_data_masks_first
    loss_masks = torch.zeros(seq_length, seq_data_mask_list.shape[1])
    if using_cuda:
        loss_masks = loss_masks.cuda()
    for frame in range(seq_length):
        loss_masks[frame] = seq_data_mask_list[frame] * node_pre
        if frame > 0:
            node_pre = seq_data_mask_list[frame - 1]
    return loss_masks, sum(sum(loss_masks))


def L2forTest(outputs, targets, obs_length, lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[obs_length - 1:, pedi_full]
    error = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()

    return error.item(), error_cnt, final_error.item(), final_error_cnt, error_full


def call_errors(outputs, targets, obs_length, loss_masks, seq_data_mask_list, nl_thred):
    '''
    Evaluation including non-linear ade/fde.
    '''
    nl_list = torch.zeros(loss_masks.shape).cuda()
    pednum = targets.shape[1]
    for ped in range(pednum):
        gt_traj = targets[seq_data_mask_list[:, ped] > 0, ped]
        second = torch.zeros(gt_traj.shape).cuda()
        first = gt_traj[:-1] - gt_traj[1:]
        second[1:-1] = first[:-1] - first[1:]
        tmp = abs(second) > nl_thred
        nl_list[seq_data_mask_list[:, ped] > 0, ped] = (torch.sum(tmp, 1) > 0).float()
    seq_length = outputs.shape[0]

    error = torch.norm(outputs - targets, p=2, dim=2)
    error_nl = error * nl_list
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(loss_masks, dim=0) == seq_length
    error_nl = error_nl[obs_length - 1:, pedi_full]
    error_full = error[obs_length - 1:, pedi_full]
    error_sum = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()
    error_nl = error_nl[error_nl > 0]
    return error_sum.item(), error_cnt, final_error.item(), final_error_cnt, torch.sum(
        error_nl).item(), error_nl.numel(), error_full


from geometry_utils import collision_check
from cross_utils import get_ped_type_from_cross


def cal_col_rate(output_seq, translations, start_frame_xys, gt_seq, seq_pedids, seq_start_frames, loss_masks, batch_key_list):
    # debug(f'output_seq shape {output_seq.shape}\n'
    #       f'translations shape {translations.shape}\n'
    #       f'cur_frame_xys shape {start_frame_xys.shape}\n'
    #       f'seq_pedids shape {seq_pedids.shape}\n'
    #       f'seq_frames shape {seq_start_frames.shape}\n'
    #       f'loss_masks shape {loss_masks.shape}\n'
    #       # f'batch_key_list {batch_key_list}\n'
    #       # f'seq_start_frames {seq_start_frames}'
    #       )

    seq_t = output_seq + translations
    seq_t = seq_t.cpu().detach().numpy()
    gt_seq = gt_seq.cpu().detach().numpy()

    seq_start_frames = seq_start_frames.cpu().detach().numpy()

    # debug(f'start_frame_xys shape {start_frame_xys.shape}\n')
    start_frame_xys = start_frame_xys.reshape(1, -1, 2).cpu().detach().numpy()
    # debug(f'start_frame_xys shape {start_frame_xys.shape}\n'
          # f'seq_t[:-1] shape {seq_t[:-1].shape}\n')
    seq_last_t = np.concatenate([start_frame_xys, seq_t[:-1]], axis=0)
    # debug(f'seq_t shape {seq_t.shape}\n'
          # f'seq_last_t shape {seq_last_t.shape}\n')

    col_rates = []
    for step in range(seq_t.shape[0]):
        ped_xys = seq_t[step]
        ped_gt_xys = gt_seq[step]

        ped_last_xys = seq_last_t[step]
        ped_headings = ped_xys - ped_last_xys
        ped_ids = seq_pedids[step]
        start_frames = seq_start_frames[step]
        step_masks = loss_masks[step]

        # debug(f'ped_last_xys {ped_last_xys[0]}, '
        #       f'ped_xys {ped_xys[0]}, '
        #       f'heading {ped_headings[0]}')
        debug(f'ped_gt_xys {ped_gt_xys[0]}, '
              f'ped_xys {ped_xys[0]}, '
              f'heading {ped_headings[0]}, '
              f'step_masks {step_masks[0]}')
        # group ped_idx by frame id.
        for (scene, frame_id) in batch_key_list:
            frame_id = int(frame_id)
            indices = np.where(start_frames == frame_id)
            frame_ped_xys = ped_xys[indices]
            frame_ped_headings = ped_headings[indices]
            frame_ped_ids = ped_ids[indices]
            # debug(f'indices {indices}\n'
            #       f'frame_ped_xys {frame_ped_xys}\n'
            #       f'frame_ped_headings {frame_ped_headings}\n')
            # debug(f'frame_ped_ids {frame_ped_ids}')

            indices = indices[0]
            num_peds = indices.shape[0]
            # debug(f'num_peds = {num_peds}')
            for i in range(num_peds):
                if np.linalg.norm(frame_ped_headings[i]) == 0:
                    continue
                if step_masks[indices[i]] == 0:
                    continue
                # debug(f'i={i}')
                ped1_key = str(frame_ped_ids[i])
                for j in range(i + 1, num_peds):
                    if np.linalg.norm(frame_ped_headings[j]) == 0:
                        continue
                    if step_masks[indices[j]] == 0:
                        continue
                    # debug(f'j={j}')
                    ped2_key = str(frame_ped_ids[j])
                    col = collision_check(frame_ped_xys[i], frame_ped_headings[i], get_ped_type_from_cross(ped1_key),
                                          frame_ped_xys[j], frame_ped_headings[j], get_ped_type_from_cross(ped2_key))
                    col_rates.append(float(col))
    col_rate = sum(col_rates) / len(col_rates)
    debug(f'batch col rate = {col_rate}')
    return col_rates


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
