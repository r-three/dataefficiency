from pynvml import *
import pynvml as nvml
import psutil
import numpy as np
import json
import pandas as pd
from statsmodels.formula.api import ols


def get_mem_usage_stats(msg):
    nvml.nvmlInit()
    for idx in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = info.total
        total_mem = total_mem / (1024*1024*1024)
        free_mem = info.free
        free_mem = free_mem / (1024*1024*1024)
        print(f"State {msg} Device {idx}: total mem {total_mem:.2f}, free mem {free_mem:.2f}")
    
    print(f"State {msg} Total virtual memory available {psutil.virtual_memory().available//1024**3} GB | Virtual memory % used: {np.round(psutil.virtual_memory().percent)}% | CPU memory % used: {np.round(psutil.cpu_percent())}%") 


def load_auc(path, tasks_to_exclude):

    with open(path) as f:
        ff = json.load(f)
    
    df_auc = pd.DataFrame(ff).T.sort_values('extrapolation_auc').reset_index().rename(columns={'extrapolation_auc':'auc','index':'task'})
    df_auc = df_auc[~df_auc['task'].isin(tasks_to_exclude)].reset_index(drop=True)

    return df_auc

def run_leave_one_out(df_in, depvar, predvar):
    
	# fit 
	d = {}
	tasks = list(df_in['task'].unique())

	for t in tasks:    
		d[t] = {}
		train = df_in[df_in['task'] != t].reset_index(drop=True)
		test_heldout = df_in[df_in['task'] == t].reset_index(drop=True) # one heldout task

		# fit the simple regression model
		formula = f"""{depvar} ~ """
		for p in predvar:
			formula += f"{p} + "
		formula = formula.strip(" +").strip()
		lm = ols(formula,train).fit()

		test_heldout['pred'] = 0
		for p in predvar:
			test_heldout['pred'] += test_heldout[p] * lm.params[p].item()
			d[t][f'coeff_{p}'] = lm.params[p].item()
			d[t][f'pval_{p}'] = lm.pvalues[p].item()
			d[t][f'param_raw_{p}'] = test_heldout[p]
		
		test_heldout['pred'] += lm.params['Intercept'].item()
		
		d[t]['intercept'] = lm.params['Intercept'].item()
		d[t]['pval_intercept'] = lm.pvalues['Intercept'].item()
		d[t]['pred'] =  test_heldout['pred'].item()
		d[t]['depvar'] = test_heldout[depvar].item()
		d[t]['abs_diff'] =  abs(test_heldout[depvar] - test_heldout['pred']).item()
		d[t]['diff'] = (test_heldout[depvar] - test_heldout['pred']).item()
		d[t]['sqrd_err'] = ((test_heldout[depvar] - test_heldout['pred'])**2).item()

	res =pd.DataFrame(d).T.reset_index().rename(columns={'index':'task'})

	return res

def get_model_prefix(model_name):
    
    model_prefix=None
    if "mistral" in model_name.lower():
        model_prefix = "mistral"
    elif "llama" in model_name.lower():
        model_prefix = "llama"
    elif "qwen" in model_name.lower():
        model_prefix = "qwen"
    elif "smollm" in model_name.lower():
        model_prefix = "smollm"

    return model_prefix 
