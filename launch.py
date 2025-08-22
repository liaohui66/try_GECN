import os
import subprocess
import sys
import uuid # 导入 uuid 库来生成唯一的文件名

def main():
    """
    最终版 DDP 启动器。
    它会创建一个唯一的同步文件名，并将其作为命令行参数传递给所有子进程。
    """
    # --- 配置 ---
    num_gpus = 2
    script_to_run = "train.py"
    
    python_executable = r"C:\Users\LH\miniconda3\envs\sen_pytorch\python.exe"
    if not os.path.exists(python_executable):
        print(f"FATAL ERROR: Python executable not found at '{python_executable}'")
        return
        
    # ==================== 关键修改: 创建唯一的、共享的同步文件 ====================
    # 使用 uuid 确保每次运行的文件名都是唯一的，避免与旧的运行冲突
    sync_file = os.path.join(os.getcwd(), f"ddp_sync_{uuid.uuid4()}.tmp")
    # =========================================================================

    print("="*60)
    print(f"Manual DDP Launcher for Windows")
    print(f"  - Python: {python_executable}")
    print(f"  - Script: {script_to_run}")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Sync File: {sync_file}")
    print("="*60)
    
    processes = []
    
    try:
        for local_rank in range(num_gpus):
            env = os.environ.copy()
            # 这些环境变量仍然是必须的，train.py 会读取它们
            env["WORLD_SIZE"] = str(num_gpus)
            env["RANK"] = str(local_rank)
            env["LOCAL_RANK"] = str(local_rank)
            
            # ==================== 关键修改: 将文件名作为参数传递 ====================
            command = [
                python_executable,
                script_to_run,
                "--ddp_sync_file", # 参数名
                sync_file          # 参数值
            ]
            # =====================================================================
            
            print(f"\nLaunching process for GPU {local_rank}...")
            proc = subprocess.Popen(command, env=env)
            processes.append(proc)
            
        for proc in processes:
            proc.wait()

    finally:
        # ==================== 关键修改: 确保同步文件被清理 ====================
        # 无论成功还是失败，最后都尝试删除这个临时文件
        if os.path.exists(sync_file):
            print(f"\nCleaning up sync file: {sync_file}")
            os.remove(sync_file)
        # ===================================================================

    print("\nAll processes finished.")

if __name__ == "__main__":
    main()