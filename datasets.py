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
from typing import Dict, List, Callable, Any, Optional, Union, Type
from processors import SequentialProcessor, ReadAudioProcessor, MelspectrogramProcessor



def read_audiodir(dataset_path: List[str], 
                  subsample: Optional[int] = None, 
                  dataset: Optional[str] = None, 
                  regex_groups: Optional[str] = None, 
                  filter_list: Optional[str] = None, 
                  partition_lists: Optional[Dict[str, Optional[str]]] = None, 
                  filter_mode: str = 'include', 
                  cache_dict_path: str = None) -> pd.DataFrame:
    """
    Reads audio files from directories and generates metadata DataFrame.

    Args:
        dataset_path (list): List of paths to directories containing audio files.
        subsample (int, optional): Number of files to subsample. Defaults to None.
        dataset (str, optional): Name of the dataset. Defaults to None.
        regex_groups (str, optional): Regular expression to extract metadata from filenames. Defaults to None.
        filter_list (str, optional): Path to a file containing a list of filenames to filter. Defaults to None.
        partition_lists (dict, optional): Dictionary mapping partitions to filenames. Defaults to None.
        filter_mode (str, optional): Filtering mode, either 'include' or 'discard'. Defaults to 'include'.

    Returns:
        pandas.DataFrame: Metadata DataFrame containing information about audio files.

    Raises:
        Exception: If an unrecognized filter mode is provided.
    """

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

    
def dataset_random_split(df: pd.DataFrame, 
                         proportions: Dict[str, float] = {}) -> Dict[str, pd.DataFrame]:
    """
    Splits a DataFrame into partitions randomly based on given proportions.

    Args:
        df (pandas.DataFrame): The DataFrame to be split.
        proportions (dict, optional): Dictionary containing proportions of split for each partition. 
            If value is greater than 1, it's treated as the number of samples to include in the partition. 
            If value is between 0 and 1, it's treated as the proportion of the DataFrame to include in the partition.
            If -1 is provided for any partition, remaining samples will be assigned to this partition.
            Defaults to an empty dictionary.

    Returns:
        dict: Dictionary containing partitions as DataFrames.

    Raises:
        Exception: If -1 is used in more than one entry in the proportions dictionary.
    """
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

def dynamic_pad_batch(x: Union[list, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Dynamically pads a batch of sequences with variable lengths and converts them to PyTorch tensors.

    Args:
        x (Union[list, dict]): List or dictionary containing sequences to be padded.

    Returns:
        dict: Dictionary containing padded sequences converted to PyTorch tensors.
    """
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

def compensate_lengths(df: pd.DataFrame, chunk_length: Optional[float] = None) -> List[int]:
    """
    Compensates for varying lengths of elements in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing elements with varying lengths.
        chunk_length (float, optional): The length of each chunk in seconds. 
            If provided, elements will be divided into chunks of approximately equal length. 
            Defaults to None.

    Returns:
        list: A list of indices corresponding to elements in the DataFrame, accounting for varying lengths.

    Note:
        If chunk_length is not provided, each element is represented by a single index.
        If chunk_length is provided, elements are divided into chunks, and each chunk is represented by its element's index.
    """
    if chunk_length is not None:
        map_idx = []
        for i, (idx, row) in enumerate(df.iterrows()):
            map_idx.extend([i]*int(max(1,row['duration']//chunk_length)))
        return map_idx
    else:
        return list(range(len(df)))

class DictDataset(Dataset):
    """
    Dataset class to handle data stored in a dictionary-like format.

    Args:
        metadata (pandas.DataFrame): DataFrame containing metadata of the dataset.
        state (dict): Dictionary containing additional state information.
        out_cols (list): List of columns to be included in the output.
        preprocessor (optional): Callable to apply to a dataframe row before returning the item. Defaults to None.
        index_mapper (callable, optional): A function to map indices of metadata. Defaults to None.
        state_keys (list, optional): List of keys from the state dictionary to be included in the dataset's state. Defaults to None.
    """
    def __init__(self, 
                 metadata: pd.DataFrame, 
                 out_cols: List[str], 
                 preprocessor: Callable[[Any, Dict[str, Any]], Any] = None, 
                 index_mapper: Optional[Callable[[pd.DataFrame], List[int]]] = None):

        self._metadata = metadata
        self._out_cols = out_cols
        self._state = {}
        self._state['metadata'] = metadata
        self._preprocessor = preprocessor()
        if index_mapper is not None:
            self._idx_map = index_mapper(self._metadata)
        else:
            self._idx_map = list(range(len(self._metadata)))

    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[self._idx_map[idx]])
        if self._preprocessor is not None:
            row = self._preprocessor(row)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._idx_map)


class EncodecMAEDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str = None) -> None:
        if stage == 'fit':
            df = read_audiodir(self.args.dataset.audio_dir, dataset='emo', subsample=None, cache_dict_path=self.args.dataset.cache)
            df = remove_long_audios(df, limit=self.args.dataset.filter_audio_length)
            partitions = dataset_random_split(df, proportions={'train':-1,'validation':self.args.dataset.val_set_size})
            datasets = {}
            read_audio_proc= partial(ReadAudioProcessor, key_in='filename', key_out='wav', max_length=self.args.dataset.max_audio_length)
            if self.args.input == 'encodec':
                out_cols = ['wav']
                preprocessor = partial(SequentialProcessor, processors=[read_audio_proc])
            elif self.args.input == 'mel':
                out_cols = ['wav', 'wav_features']
                mel_spec_proc = partial(MelspectrogramProcessor, key_in='wav', key_out='wav_features', sample_frequency=self.args.wav_encoder.fs, frame_shift=self.args.mel.frame_shift, frame_length=self.args.mel.frame_length, htk_compat=True, use_energy=False, window_type=self.args.mel.window_type, num_mel_bins=self.args.mel.num_bins, dither=0.0, norm_stats=[self.args.mel.mean, self.args.mel.std])
                preprocessor = partial(SequentialProcessor, processors=[read_audio_proc, mel_spec_proc])
            
            for k, v in partitions.items():
                datasets[k] = DictDataset(
                    v, 
                    out_cols=out_cols, 
                    preprocessor=preprocessor, 
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
