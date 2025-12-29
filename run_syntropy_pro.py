import subprocess
import time
import os
import signal
import sys

def run_pro_trading():
    print("🚀 SYNTROPY QUANT: PRO EXECUTION INITIALIZED")
    print(">>> MODE: AGGRESSIVE TURBO (SIGMA/CITADEL STYLE)")
    print(">>> PREVENTING MAC SLEEP VIA CAFFEINATE...")

    # Define commands
    # 1. Turbo MFT (High Frequency Streaming)
    turbo_cmd = ["caffeinate", "-is", "python3", "turbo_mft.py"]
    
    # 2. Main System (Portfolio Management & Risk Control - Run Once every 30 mins)
    main_cmd = ["python3", "trading_system.py", "--mode", "paper", "--run", "once"]

    processes = []
    
    try:
        # Start Turbo MFT with unbuffered output
        print(">>> Launching Turbo MFT Engine (Tick-by-Tick)...")
        turbo_proc = subprocess.Popen(
            ["python3", "-u", "turbo_mft.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1
        )
        processes.append(turbo_proc)
        
        # Periodic Main Cycle
        while True:
            # We don't want a silent sleep. We want to check if turbo is alive.
            for _ in range(30): # 30 * 60 seconds = 30 minutes
                if turbo_proc.poll() is not None:
                    print("⚠️ Turbo MFT crashed! Restarting...")
                    turbo_proc = subprocess.Popen(["python3", "-u", "turbo_mft.py"], stdout=sys.stdout, stderr=sys.stderr, bufsize=1)
                
                time.sleep(60) # Heartbeat check every minute
                
            print("\n[SYSTEM] Triggering Scheduled Portfolio Rebalancing...")
            subprocess.run(["python3", "trading_system.py", "--mode", "paper", "--run", "once"])
            
    except KeyboardInterrupt:
        print("\n🛑 SHUTDOWN SIGNAL RECEIVED. CLOSING ALL KERNELS...")
        for p in processes:
            p.terminate()
        sys.exit(0)

if __name__ == "__main__":
    run_pro_trading()
