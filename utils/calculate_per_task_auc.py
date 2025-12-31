import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os
import json
import seaborn as sns
import numpy as np
from sklearn.metrics import auc
from functools import reduce

generation_tasks = ['sciq','typescript_chunks','polish_sequence_labeling','coqa','qa_wikidata','squad_v2','disfl_qa']

def calculate_auc(df):
    """Calculate the AUC given the normalized data and accuracy"""
        
    df = df.sort_values('data_size_normalized')
    auc_val = auc(np.array(df['data_size_normalized']), np.array(df['accuracy_normalized']))

    return auc_val


def clean_curve(df, data_cutoff, human_eval):
    """Extrapolate the curve for missing values (performance at data size <=2500) or non-increasing accuracy"""

    for i in [50, 100, 200, 500, 1000, 2500, 5000]:
        if i not in df['data_size'].unique():
            df = pd.concat([df, pd.DataFrame({'data_size':[i], 'accuracy':[0]})])

    df= df[(df['data_size'] <=data_cutoff)]
    df= df.sort_values('data_size').set_index('data_size')

    prev_acc = 0
    for i in df.index:
        curr_acc = float(df[df.index==i]['accuracy'].values[0])
        if prev_acc > curr_acc:
            df.loc[i] = prev_acc
        else:
            prev_acc = curr_acc

    df = df.reset_index()

    return df


def run_data_processing(task_json_name, data_cutoff, is_generation):

    print(task_json_name, "to open")
    with open(f"../results/finetune_results/{task_json_name}") as f:
        ff = json.load(f)
        
    # data preprocessing
    df_full = pd.DataFrame(ff).T.reset_index().rename(columns={'index':'data_size'})

    print(df_full.head())
    if 'exact_string_match_accuracy' in df_full.columns and 'accuracy' in df_full.columns:
        df_full['accuracy'] = df_full['accuracy'].fillna(df_full['exact_string_match_accuracy'])
        df_full = df_full.drop('exact_string_match_accuracy', axis=1)
    elif 'exact_string_match_accuracy' in df_full.columns:
        df_full = df_full.rename(columns={'exact_string_match_accuracy':'accuracy'})

    df_full['data_size'] = df_full['data_size'].astype(int)
    df_full['accuracy'] = df_full['accuracy'].astype(float)
    df_full = df_full[(df_full['data_size']>=0) & (df_full['data_size']<=data_cutoff)]
    
    if is_generation:
        df_full = df_full.drop(['accuracy'], axis=1)
        df_full = df_full.rename(columns={'f1':'accuracy'})

    df_full = df_full[(df_full['data_size']==0) | (df_full['data_size']>=50)].reset_index(drop=True)

    return df_full


def load_run_full_results(model_name, tasks, data_cutoff=2500, use_log=False, tags=['org']):
    """Load each fine-tuning experiment result and calculate AUC"""

    d = {}

    with open("../results/human_eval.json", "r") as f:
        d_human_eval = json.load(f)
         
    ls_dfs = []
    for task_json in tasks:
        
        task_name=task_json.split('_full')[0].split(f'{model_name}_')[-1]
        human_eval = None if task_name not in d_human_eval else d_human_eval[task_name]['score']

        # average the runs across seeds
        runs = []
        for tag in tags:
            try:
                print(f"tag {tag}")
                if tag in ['48', '37']:
                    task_json_name = task_json.replace('_v2.json', f'_v2_seed{tag}.json')
                elif tag == 'org':
                    task_json_name = task_json

                df_run = run_data_processing(task_json_name=task_json_name, data_cutoff=data_cutoff, is_generation=task_name in generation_tasks)
                # calculate metrics to get clean curves
                max_acc = df_run['accuracy'].max().item()
                min_acc = df_run[df_run['data_size']>=0]['accuracy'].min().item()
                human_eval = max(human_eval, max_acc) if human_eval is not None else max_acc

                if df_run.shape[0] < 2:
                    continue
                
                df_run_clean = clean_curve(df_run, data_cutoff=data_cutoff, human_eval=human_eval)
                df_run_clean['reached_max'] = np.where(df_run_clean['accuracy'] + 0.02 < max_acc, 0, 1)
                df_run_clean = df_run_clean[['data_size','accuracy','reached_max']].rename(columns={f'accuracy':f'accuracy_{tag}', 'reached_max': f'reached_max_{tag}'})
                runs.append(df_run_clean)
            
            except:
                print(f'{tag} run not found for the task: {task_name}')
                continue
        
        if len(runs) == 0:
            print(f"No result found for {task_name}")
            continue

        # combine random seed runs
        df_clean = reduce(lambda left, right: pd.merge(left, right, on='data_size'), runs)
        df_clean['accuracy'] = df_clean[[i for i in df_clean.columns if 'accuracy' in i and 'normalize' not in i]].mean(axis=1) # average the raw acc across random runs
        
        print(task_json)
        print(df_clean.head())
                
        # calculate the accuracy across runs from extrapolated curves
        df_clean['accuracy_normalized'] = (df_clean['accuracy'] - df_clean['accuracy'].min()) /(human_eval - df_clean['accuracy'].min())
        df_clean['data_size_normalized'] = df_clean['data_size'] / df_clean['data_size'].max()

        if use_log:
            df_clean['data_size_org'] = df_clean['data_size']
            df_clean['data_size'] = np.where(df_clean['data_size'] == 0, 1, df_clean['data_size'])
            df_clean['data_size'] = np.log2(df_clean['data_size'].astype(int))
            df_clean['data_size_normalized'] = df_clean['data_size'] / df_clean['data_size'].max()

        # calculate misc metrics
        auc_val_clean = calculate_auc(df_clean)

        d[task_name] = {'max_acc': max_acc,
                        'min_acc': min_acc,
                        'extrapolation_auc': auc_val_clean,
                        'human_eval': human_eval
                        }
        df_clean['task'] = task_name
        ls_dfs.append(df_clean)
        print('extrapolation auc: ', auc_val_clean)
            
    return ls_dfs, d


def run_auc_calculation(model_prefix, data_cutoff):
    """Run the script end-to-end for all tasks,
        using the specified model results and data-cutoffs for maximum data budget"""

    all_tasks = [i for i in os.listdir('../results/finetune_results') if 'full_result_v2.json' in i and model_prefix in i]
    
    dfs, auc_res = load_run_full_results(model_name=model_prefix,
                                tasks=all_tasks,
                                use_log=True,
                                data_cutoff=data_cutoff,
                                tags=['org']
                                )
    
    with open(f"../results/auc_res/{model_prefix}_auc_logscale_by_task_{data_cutoff}.json", "w") as f:
        json.dump(auc_res, f, indent=4)

if __name__=='__main__':

    run_auc_calculation("llama",5000)
    run_auc_calculation("mistral",5000)
    run_auc_calculation("qwen",5000)

