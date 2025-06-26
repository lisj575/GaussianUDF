import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

datapath = 'data/dtu'
scenes = [24,37,40,55,63,69,83,97,65, 105, 106, 110, 114, 118, 122]
MAX_WORKER = 8
target_name = 'release_dtu'



total_iterations = 10

excluded_gpus = set()
mem_threshold = 0.2

jobs = scenes.copy()
current_date = time.strftime("%Y%m%d", time.localtime())

def train_block(gpu_id, scene):
    # training
    out_path = f"output_dtu/{current_date}-{target_name}/scan{scene}/"
    if not os.path.exists(f"{out_path}/chkpnt30000.pth"):
        cmd = f'python train_dtu.py -s data/dtu/scan{scene} --gpu {gpu_id} -m {target_name}/scan{scene} --depth_ratio 1.0 -r 2 --lambda_dist 1000 --outdir output_dtu'
        print(cmd)
        os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python eval_dtu/evaluate_single_scene.py --input_mesh {out_path}/mesh_udf/mesh_res512_thred0.0000_iter30000.ply --scan_id {scene}'
    print(cmd)
    os.system(cmd)

def worker(gpu_id, scene):
    print(f"Starting job on GPU {gpu_id} with scene {scene}\n")
    train_block(gpu_id, scene)
    print(f"Finished job on GPU {gpu_id} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
            # Get the list of available GPUs, not including those that are reserved.
            all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxLoad=0.9, maxMemory=mem_threshold))
            available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

            # Launch new jobs on available GPUs
            while available_gpus and jobs:
                    gpu = available_gpus.pop(0)
                    job = jobs.pop(0)
                    future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
                    future_to_job[future] = (gpu, job)
                    reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

            # Check for completed jobs and remove them from the list of running jobs.
            # Also, release the GPUs they were using.
            done_futures = [future for future in future_to_job if future.done()]
            for future in done_futures:
                    job = future_to_job.pop(future)  # Remove the job associated with the completed future
                    gpu = job[0]  # The GPU is the first element in each job tuple
                    reserved_gpus.discard(gpu)  # Release this GPU
                    print(f"Job {job} has finished., rellasing GPU {gpu}")
            # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
            # when there are no GPUs available.
            if len(jobs) > 0:
                    print("No GPU available at the moment. Retrying in 1 minutes.")
                    time.sleep(60)
            else:
                    time.sleep(10)


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
    dispatch_jobs(jobs, executor)

