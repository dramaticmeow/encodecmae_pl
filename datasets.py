import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np
from tqdm import tqdm
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import copy, json, os
import torch
import re, random
from pytorch_lightning import LightningDataModule
from functools import partial


def ProcessorReadAudio(x,
                       state,
                       input=None,
                       output=None,
                       max_length=None,
                       mono=True
):  
    def read_sample(x,state,max_length,mono):
        if max_length is not None:
            audio_info = sf.info(x[input])
            desired_frames = int(max_length*audio_info.samplerate)
            total_frames = audio_info.frames
            if total_frames > desired_frames:
                start = random.randint(0,total_frames - desired_frames)
                stop = start + desired_frames
            else:
                start = 0
                stop = None
            if 'chunk_idx' in state:
                #This is for ordered reading in chunks when doing evals
                start = int(state['chunk_idx']*desired_frames)
                stop = start + desired_frames
        else:
            start = 0
            stop = None
        if 'start' in x:
            start = x['start']
        if 'stop' in x:
            stop = x['stop']
        x['start'] = start
        x['stop'] = stop
        wav, fs = sf.read(x[input], start=start, stop=stop, dtype=np.float32)
        if (wav.ndim == 2) and mono:
            wav = np.mean(wav,axis=-1)
        return wav
    try:
        wav = read_sample(x, state, max_length, mono)
    except:
        print('Failed reading {}'.format(x))
        wav = None
    if output is None:
        output = input
    x[output] = wav
    
    return x, state

def ProcessorLoadNumpy(x, state, input, output):
    x[output] = np.load(x[input])
    return x,state

# def load_dataset(state, reader_fn, 
#                  cache=True, 
#                  filters=[], 
#                  key_out='dataset_metadata',
#                  rename=None):
    
#     if not (cache and key_out in state):
#         if not isinstance(reader_fn, list):
#             reader_fn = [reader_fn]
#         dfs = [fn() for fn in reader_fn]
#         df = pd.concat(dfs).reset_index()
#         state[key_out] = df
#     else:
#         logger.info('Caching dataset metadata from state')
    
#     for f in filters:
#         state[key_out] = f(state[key_out])
#     if rename is not None:
#         for r in rename:
#             state[key_out][r['column']] = state[key_out][r['column']].apply(lambda x: r['new_value'] if x == r['value'] else x)
    
#     return state

def read_audiodir(dataset_path, subsample=None, dataset=None, regex_groups=None, filter_list=None, partition_lists=None,filter_mode='include', cache_dict_path=None):
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]
    all_files = []
    for p in dataset_path:
        all_files_i = list(Path(p).rglob('*.wav')) + list(Path(p).rglob('*.flac'))
        all_files.extend(all_files_i)
    if filter_list is not None:
        with open(filter_list, 'r') as f:
            keep_values = set(f.read().splitlines())
        n_slashes = len(next(iter(keep_values)).split('/')) - 1
        stem_to_f = {'/'.join(v.parts[-n_slashes-1:]): v for v in all_files}
        if filter_mode == 'include':
            all_files = [stem_to_f[k] for k in keep_values]
        elif filter_mode == 'discard':
            all_files = [v for k,v in stem_to_f.items() if k not in keep_values]
        else:
            raise Exception("Unrecognized filter_mode {}".format(filter_mode))
    rows = []
    if subsample is not None:
        subsample_idx = np.random.choice(np.arange(len(all_files)),size=subsample,replace=False)
        all_files = np.array(all_files)[subsample_idx]
    print(f'Found {len(all_files)} files')
    cache_dict = {}
    if cache_dict_path is not None:
        print(f'Loading metadata cache from {cache_dict_path}')
        with open(cache_dict_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                p = obj['path']
                meta = {}
                meta['sr'] = obj['sample_rate']
                meta['channels'] = obj['channels']
                meta['frames'] = obj['sample_points']
                meta['duration'] = obj['duration']
                cache_dict[os.path.basename(p)] = meta

    for f in tqdm(all_files):
        try:
            basename = f.stem+f.suffix
            if basename in cache_dict:
                metadata = cache_dict[basename]
                metadata['filename'] = str(f.resolve())
            else:
                finfo = sf.info(f)
                metadata = {'filename': str(f.resolve()),
                        'sr': finfo.samplerate,
                        'channels': finfo.channels,
                        'frames': finfo.frames,
                        'duration': finfo.duration}
            
            if regex_groups is not None:
                regex_data = re.match(regex_groups,str(f.relative_to(dataset_path[0]))).groupdict()
                metadata.update(regex_data)
            rows.append(metadata)
        except Exception as e:
            print(f'Failed reading {f}. {e}')
    df = pd.DataFrame(rows)
    if dataset is not None:
        df['dataset'] = dataset
    df['rel_path'] = df['filename'].apply(lambda x: str(Path(x).relative_to(dataset_path[0])))
    if partition_lists is not None:
        remainder = None
        map_to_partitions={}
        for k,v in partition_lists.items():
            if v is not None:
                list_path = Path(dataset_path[0],v)
                with open(list_path,'r') as f:
                    list_files = f.read().splitlines()
                for l in list_files:
                    map_to_partitions[str(l)] = k
            else:
                remainder = k
        df['partition'] = df['rel_path'].apply(lambda x: map_to_partitions[x] if x in map_to_partitions else remainder)
        df = df.drop('rel_path', axis=1)
    return df

# def get_dataloaders(state, split_function=None, 
#                            dataset_cls=None, 
#                            dataloader_cls=None, 
#                            dataset_key_in='dataset_metadata',
#                            dataset_key_out='datasets',
#                            partitions_key_out='partitions',
#                            dataloaders_key_out='dataloaders'):

#     if split_function is not None:
#         partitions = split_function(state[dataset_key_in])
#     else:
#         partitions = {'train': state[dataset_key_in]}

#     datasets = {k: dataset_cls[k](v, state) for k,v in partitions.items() if k in dataset_cls}
#     dataloaders = {k: dataloader_cls[k](v) for k,v in datasets.items() if k in dataloader_cls}

#     state[partitions_key_out] = partitions
#     state[dataset_key_out] = datasets
#     state[dataloaders_key_out] = dataloaders

#     return state
    
def dataset_random_split(df, proportions={}):
    idxs = df.index
    prop_type = [v for k,v in proportions.items() if v>1]
    if len(prop_type)>0:
        prop_type = 'n'
    else:
        prop_type = 'prop'
    remainder_k = [k for k,v in proportions.items() if v==-1]
    if len(remainder_k) > 1:
        raise Exception("-1 can't be used in more than one entry")
    elif len(remainder_k) == 1:
        remainder_k = remainder_k[0]
    else:
        remainder_k = None
    partitions = {}
    for k,v in proportions.items():
        if k != remainder_k:
            if prop_type == 'prop':
                v = int(len(df)*v)
            sampled_idxs = np.random.choice(idxs, v, replace=False)
            idxs = [i for i in idxs if i not in sampled_idxs]
            partitions[k] = df.loc[sampled_idxs]
    if remainder_k is not None:
        partitions[remainder_k] = df.loc[idxs]
    return partitions
    
def remove_long_audios(df, limit=10000):
    df = df.loc[df['duration']<limit]
    return df

def dynamic_pad_batch(x):
    def not_discarded(x):
        if x is None:
            return False
        else:
            return not any([xi is None for xi in x.values()])

    def get_len(x):
        if x.ndim == 0:
            return 1
        else:
            return x.shape[0]

    def pad_to_len(x, max_len):
        if x.ndim == 0:
            return x
        else:
            pad_spec = ((0,max_len-x.shape[0]),) + ((0,0),)*(x.ndim - 1)
            return np.pad(x,pad_spec)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            if x.dtype in [np.float64, np.float32, np.float16, 
                        np.complex64, np.complex128, 
                        np.int64, np.int32, np.int16, np.int8,
                        np.uint8, np.bool]:

                return torch.from_numpy(x)
            else:
                return x
            
    x_ = x
    x = [xi for xi in x if not_discarded(xi)]

    batch = {k: [np.array(xi[k]) for xi in x] for k in x[0]}
    batch_lens = {k: [get_len(x) for x in batch[k]] for k in batch.keys()}
    batch_max_lens = {k: max(v) for k,v in batch_lens.items()}
    batch = {k: np.stack([pad_to_len(x, batch_max_lens[k]) for x in batch[k]]) for k in batch.keys()}
    batch_lens = {k+'_lens': np.array(v) for k,v in batch_lens.items()}
    batch.update(batch_lens)
    batch = {k: to_torch(v) for k,v in batch.items()}

    return batch

def compensate_lengths(df, chunk_length=None):
    if chunk_length is not None:
        map_idx = []
        for i, (idx, row) in enumerate(df.iterrows()):
            map_idx.extend([i]*int(max(1,row['duration']//chunk_length)))
        return map_idx
    else:
        return list(range(len(df)))

class DictDataset(Dataset):
    def __init__(self, metadata, out_cols, preprocessors=None, index_mapper=None):
        self._metadata = metadata
        self._out_cols = out_cols
        self._state = {}
        self._state['metadata'] = metadata

        self._preprocessors = preprocessors
        if index_mapper is not None:
            self._idx_map = index_mapper(self._metadata)
        else:
            self._idx_map = list(range(len(self._metadata)))

    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[self._idx_map[idx]])
        for p in self._preprocessors:
            row, self._state = p(row, self._state)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._idx_map)

# def read_selflearning_dataset(dataset_path):
#     df = pd.read_csv(Path(dataset_path, 'metadata_selftrain_dataset.csv'), names=['start','stop','filename'])
#     df = df.reset_index()
#     df = df.rename({'index':'filename_audio','filename':'filename_targets'},axis=1)
#     return df

class EncodecMAEDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str = None) -> None:
        df = read_audiodir(self.args.dataset.audio_dir, dataset='emo', subsample=None, cache_dict_path=self.args.dataset.cache)
        df = remove_long_audios(df, limit=self.args.dataset.filter_audio_length)
        partitions = dataset_random_split(df, proportions={'train':-1,'validation':self.args.dataset.val_set_size})
        datasets = {}
        for k, v in partitions.items():
            datasets[k] = DictDataset(
                v, 
                out_cols=['wav'], 
                preprocessors=[partial(ProcessorReadAudio, input='filename', output='wav', max_length=self.args.dataset.max_audio_length)], 
                index_mapper=partial(compensate_lengths, chunk_length=self.args.dataset.max_audio_length)
            )
        self.datasets = datasets

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'], 
            self.args.dataset.train_batch_size, 
            shuffle=True, 
            num_workers=self.args.dataset.train_num_workers, 
            drop_last=False,
            pin_memory=True,
            collate_fn=dynamic_pad_batch
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.datasets['validation'], 
            self.args.dataset.val_batch_size, 
            shuffle=False, 
            num_workers=self.args.dataset.val_num_workers, 
            drop_last=False,
            pin_memory=True,
            collate_fn=dynamic_pad_batch
            )
    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    dm = EncodecMAEDataModule(OmegaConf.load('/data41/private/dongyuanliang/encodecmae/encodecmae_pl/config/encodecmae_base.yaml'))
    dm.setup()
