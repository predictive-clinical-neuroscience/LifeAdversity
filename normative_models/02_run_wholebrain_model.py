import os
from pcntoolkit.normative_parallel import execute_nm, collect_nm, delete_nm

# globals
root_dir = '/project/3022054.01/projects/linschl/'
proc_dir = os.path.join(root_dir,'results','Run_8_inv')
w_dir = os.path.join(proc_dir,'vox/')
os.makedirs(w_dir, exist_ok=True)

py_path = '/home/preclineu/linschl/.conda/envs/spyder-env-new/bin/python'
log_path = os.path.join(proc_dir,'logs/')
os.makedirs(log_path, exist_ok=True)
job_name = 'CA_run_8_inv'
batch_size = 400 
memory = '30gb'
duration = '03:00:00'
alg = 'blr'
warp = 'WarpSinArcsinh'
warp_reparam = 'True'
cluster = 'slurm'

resp_file_tr = os.path.join(proc_dir,'resp_tr.pkl')
resp_file_te = os.path.join(proc_dir,'resp_te.pkl')

cov_file_tr = os.path.join(proc_dir, 'cov_bspline_tr.txt')
cov_file_te = os.path.join(proc_dir, 'cov_bspline_te.txt')

os.chdir(w_dir)
execute_nm(processing_dir = w_dir,
          python_path = py_path, 
          job_name = job_name,
          covfile_path = cov_file_tr,
          respfile_path = resp_file_tr, 
          batch_size = batch_size,
          memory = memory,
          duration = duration,
          alg = 'blr',
          savemodel='True',
          optimizer = 'powell',
          warp = warp,
          warp_reparam = warp_reparam,
          testcovfile_path = cov_file_te,
          testrespfile_path = resp_file_te,
          cluster_spec = cluster,
          log_path = log_path,
          binary = True)



collect_nm(w_dir, job_name, collect=True, binary=True)