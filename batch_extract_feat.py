import os
import json
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from models.builder import build_EncodecMAE
from processors import SequentialProcessor, ReadAudioProcessor
import copy
import sys, torch
from functools import partial

def encodecmae_30s_mean_torch(arr, batch_size=2250):
    original_shape = arr.shape
    n = original_shape[1]
    num_full_sections = n // batch_size
    
    # 处理完整的30秒段
    full_sections = arr[:, :num_full_sections*batch_size, :]
    reshaped_full = full_sections.reshape(original_shape[0], num_full_sections, batch_size, original_shape[2])
    mean_full = reshaped_full.mean(dim=2)
    
    # 处理剩余的不足30秒的段
    if n % batch_size != 0:
        remaining = arr[:, num_full_sections*batch_size:, :]
        mean_remaining = remaining.mean(dim=1, keepdim=True)
        
        # 合并完整段和剩余段的平均值
        mean_arr = torch.cat([mean_full, mean_remaining], dim=1)
    else:
        mean_arr = mean_full
    
    return mean_arr.cpu().numpy()

def process_directory(model, input_dir, output_dir, files_to_process, batch_size=2250):
    # 保留文件的相对路径
    def preserve_relative_path(file, input_dir, output_dir):
        rel_path = os.path.relpath(file, input_dir)
        new_path = os.path.join(output_dir, rel_path)
        return os.path.splitext(new_path)[0] + ".mp3.npy"

    # 处理单个文件
    def process_file(model, file, input_dir, output_dir):
        output_file = preserve_relative_path(file, input_dir, output_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if os.path.exists(output_file):
            print(f"跳过 {file}，输出已存在")
            return

        try:
            with torch.no_grad():
                features = model.extract_features_from_file_simple(file, 64)
            features = encodecmae_30s_mean_torch(features, batch_size=batch_size)
            np.save(output_file, features)
            with open("pass.txt", "a") as f:
                f.write(file + "\n")
        except Exception as e:
            print(f"处理 {file} 失败: {e}")
            with open("log.txt", "a") as f:
                f.write(f"处理 {file} 失败: {str(e)}\n")

    # 处理目录中的所有文件
    for file in tqdm(files_to_process):
        process_file(model, file, input_dir, output_dir)
    
# input_path_root = "/2214/guozhancheng/MARBLE-Benchmark/data/MTG/audio-low"
# output_path_root = "/2214/guozhancheng/MARBLE-Benchmark/data/MTG/encodecmae_large_32K_mean30s"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', type=str, required=True)
    parser.add_argument('--input_path_root', type=str, required=True)
    parser.add_argument('--output_path_root', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--global_rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    with open(args.meta_path, "r") as f:
        files = json.load(f)[args.global_rank::args.world_size]
    print(f'进程 {args.global_rank} 处理的总音频数量:', len(files))
    config_path = f'/2214/dongyuanliang/encodecmae_pl/config/{args.model_name}.yaml'
    train_args = OmegaConf.load(config_path)
    model = build_EncodecMAE(train_args)
    model = model.to(f'cuda:{args.gpu}')
    model.eval()
    ckpt_path = f'/2214/dongyuanliang/encodecmae_pl/encodecMAE/{args.model_name}/{args.ckpt_path}'
    checkpoint = torch.load(ckpt_path, map_location=f'cuda:{args.gpu}')
    model.load_state_dict(checkpoint['state_dict'])
    read_audio_proc= partial(ReadAudioProcessor, key_in='filename', key_out='wav', max_length=train_args.dataset.max_audio_length)
    model.processor = SequentialProcessor(processors=[read_audio_proc])
    process_directory(model, args.input_path_root, args.output_path_root, files, batch_size=args.batch_size)


